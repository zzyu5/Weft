from __future__ import annotations

import ctypes
import json
import math
import os
import tempfile
import threading
import weakref
from _ctypes import dlclose
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from weft.api import KernelDefinition
from weft.frontend import lower_to_mlir

from .buffer import Buffer
from .driver import CompilationError, NativeTarget, Toolchain, run_compiler


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _release_artifact(handle: int, directory: tempfile.TemporaryDirectory) -> None:
    dlclose(handle)
    directory.cleanup()


@dataclass(frozen=True)
class CompileOptions:
    meta: Mapping[str, int] = field(default_factory=dict)
    lmul_eighths: int = 8
    unroll: int = 1
    pipeline_depth: int = 1
    scalar_load_prime: int = 0
    partial_combine_policy: str = "independent-multilevel"
    record_axis_policy: str = "within-record"
    max_widening_combine_groups: int = 2
    matrix_extension: str = "none"

    def __post_init__(self) -> None:
        values = dict(self.meta)
        if any(not isinstance(name, str) or not name.isidentifier() or
               type(value) is not int or value <= 0 for name, value in values.items()):
            raise ValueError("source bindings must be named positive integers")
        object.__setattr__(self, "meta", MappingProxyType(values))

    def arguments(self) -> list[str]:
        arguments = [f"--meta={name}={value}" for name, value in sorted(self.meta.items())]
        arguments.extend((
            f"--auto-lmul-eighths={self.lmul_eighths}",
            f"--auto-unroll={self.unroll}",
            f"--auto-pipeline-depth={self.pipeline_depth}",
            f"--auto-scalar-load-prime={self.scalar_load_prime}",
            f"--partial-combine-policy={self.partial_combine_policy}",
            f"--record-axis-policy={self.record_axis_policy}",
            f"--max-widening-combine-groups={self.max_widening_combine_groups}",
            f"--matrix-extension={self.matrix_extension}",
        ))
        return arguments


class CompiledKernel:
    def __init__(self, definition: KernelDefinition, artifact: dict,
                 directory: tempfile.TemporaryDirectory, target: NativeTarget,
                 options: CompileOptions) -> None:
        if artifact["kind"] != "weft-riscv-artifact" or len(artifact["kernels"]) != 1:
            raise CompilationError("a callable kernel requires one compiler-emitted ABI")
        self.metadata = _freeze(artifact["kernels"][0])
        if (self.metadata["march"], self.metadata["abi"], self.metadata["vlen_bits"]) != (
                target.march, target.abi, target.vlen_bits):
            raise CompilationError("compiled artifact disagrees with its discovered target")
        self.target = target
        self.options = options
        self.riscv_ir = artifact["riscv_ir"]
        self.source = artifact["intrinsic_c"]
        self.library_path = Path(directory.name) / "kernel.so"
        self._directory = directory
        self._lock = threading.RLock()
        self._signature = definition.signature
        self._library = ctypes.CDLL(str(self.library_path), mode=os.RTLD_LOCAL)
        self._release = weakref.finalize(self, _release_artifact,
                                        self._library._handle, directory)
        self._function = getattr(self._library, self.metadata["symbol"])
        self._function.argtypes = (
            [ctypes.c_void_p] * len(self.metadata["arguments"]) +
            [ctypes.c_size_t] * len(self.metadata["shape_parameters"])
        )
        self._function.restype = None
        self.closed = False

    def __call__(self, *args: object, **kwargs: object) -> None:
        with self._lock:
            if self.closed:
                raise RuntimeError("the compiled kernel has been closed")
            self.target.check_execution()
            dimensions = {}
            for name in self.metadata["shape_parameters"]:
                if name in kwargs and name not in self._signature.parameters:
                    value = kwargs.pop(name)
                    if type(value) is not int or not 0 <= value < 1 << self.target.xlen:
                        raise ValueError(f"shape parameter {name} must fit the target unsigned index ABI")
                    dimensions[name] = value
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            buffers = []
            spans = []
            for parameter in self.metadata["arguments"]:
                value = bound.arguments[parameter["name"]]
                buffer = value if isinstance(value, Buffer) else Buffer(value)
                if buffer.encoding != parameter["encoding"]:
                    raise TypeError(
                        f"{parameter['name']} requires {parameter['encoding']}, got {buffer.encoding}"
                    )
                if len(buffer.shape) != len(parameter["shape"]):
                    raise ValueError(f"{parameter['name']} has the wrong logical rank")
                for spelling, extent in zip(parameter["shape"], buffer.shape):
                    if extent >= 1 << self.target.xlen:
                        raise ValueError(f"{parameter['name']} extent cannot be represented by the target ABI")
                    if spelling.isdecimal():
                        expected = int(spelling)
                    elif spelling in dimensions:
                        expected = dimensions[spelling]
                    else:
                        dimensions[spelling] = extent
                        expected = extent
                    if extent != expected:
                        raise ValueError(f"{parameter['name']} disagrees with extent {spelling}={expected}")
                elements = parameter["record_elements"]
                if buffer.shape and buffer.shape[-1] % elements:
                    raise ValueError(f"{parameter['name']} does not contain complete Encoding records")
                required = math.prod(buffer.shape) // elements * parameter["storage_bytes"]
                if required > buffer.nbytes:
                    raise ValueError(f"{parameter['name']} storage is shorter than its logical View")
                if required and buffer.address % parameter["alignment"]:
                    raise ValueError(f"{parameter['name']} does not meet its Encoding alignment")
                if parameter["writable"] and buffer.readonly:
                    raise TypeError(f"{parameter['name']} requires writable storage")
                for other, begin, end, alias_set in spans:
                    if required and parameter["alias_set"] != alias_set and (
                            buffer.address < end and begin < buffer.address + required):
                        raise ValueError(f"{parameter['name']} aliases {other} across declared alias groups")
                if required:
                    spans.append((parameter["name"], buffer.address,
                                  buffer.address + required, parameter["alias_set"]))
                buffers.append(buffer)
            self._function(*(buffer.address for buffer in buffers),
                            *(dimensions[name] for name in self.metadata["shape_parameters"]))

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self._function = None
            self._library = None
            self._release()


class _Compiler:
    def __init__(self, toolchain: Toolchain) -> None:
        self.toolchain = toolchain
        self.target = NativeTarget.discover(toolchain)
        self._cache: dict[tuple, CompiledKernel] = {}
        self._lock = threading.RLock()

    def compile(self, definition: KernelDefinition, options: CompileOptions) -> CompiledKernel:
        if not isinstance(definition, KernelDefinition):
            raise TypeError("compile expects an @weft.kernel definition")
        if options.matrix_extension != "none":
            raise NotImplementedError("native discovery does not expose vendor matrix capabilities")
        canonical = lower_to_mlir(definition)
        arguments = options.arguments()
        key = (definition, canonical, tuple(arguments))
        with self._lock:
            self.target.check_execution()
            cached = self._cache.get(key)
            if cached is not None and not cached.closed:
                return cached
            result = run_compiler([
                self.toolchain.compiler, "--emit=artifact",
                f"--march={self.target.march}", f"--abi={self.target.abi}",
                f"--vlen-bits={self.target.vlen_bits}", *arguments,
            ], canonical)
            artifact = json.loads(result)
            cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "weft"
            cache_root.mkdir(parents=True, exist_ok=True)
            directory = tempfile.TemporaryDirectory(prefix="jit-", dir=cache_root)
            try:
                root = Path(directory.name)
                source = root / "kernel.c"
                source.write_text(artifact["intrinsic_c"], encoding="utf-8")
                run_compiler([
                    *self.toolchain.cc, *self.toolchain.cflags, "-O3", "-std=c11",
                    "-Wall", "-Wextra", "-Werror", "-ffp-contract=fast", "-fPIC", "-shared",
                    f"-march={self.target.march}", f"-mabi={self.target.abi}",
                    str(source), "-lm", "-o", str(root / "kernel.so"),
                ])
                kernel = CompiledKernel(definition, artifact, directory, self.target, options)
            except BaseException:
                directory.cleanup()
                raise
            self._cache[key] = kernel
            return kernel


_compilers: dict[tuple, _Compiler] = {}
_compilers_lock = threading.RLock()


def compile(definition: KernelDefinition, *, options: CompileOptions | None = None,
            toolchain: Toolchain | None = None) -> CompiledKernel:
    selected = (Toolchain() if toolchain is None else toolchain).resolve()
    stamp = os.stat(selected.compiler)
    identity = (selected, stamp.st_dev, stamp.st_ino, stamp.st_mtime_ns, stamp.st_size,
                frozenset(os.sched_getaffinity(0)))
    with _compilers_lock:
        compiler = _compilers.get(identity)
        if compiler is None:
            compiler = _Compiler(selected)
            _compilers[identity] = compiler
    return compiler.compile(definition, CompileOptions() if options is None else options)


class JITKernel:
    def __init__(self, definition: KernelDefinition, options: CompileOptions,
                 toolchain: Toolchain) -> None:
        if not isinstance(definition, KernelDefinition):
            raise TypeError("jit expects an @weft.kernel definition")
        self.definition = definition
        self.options = options
        self.toolchain = toolchain
        self.compiled: CompiledKernel | None = None
        self._lock = threading.RLock()

    def __call__(self, *args: object, **kwargs: object) -> None:
        with self._lock:
            if self.compiled is None:
                self.compiled = compile(self.definition, options=self.options, toolchain=self.toolchain)
            compiled = self.compiled
        compiled(*args, **kwargs)


def jit(definition: KernelDefinition, *, options: CompileOptions | None = None,
        toolchain: Toolchain | None = None) -> JITKernel:
    return JITKernel(definition, CompileOptions() if options is None else options,
                     Toolchain() if toolchain is None else toolchain)
