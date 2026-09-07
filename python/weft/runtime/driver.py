from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from weft.diagnostics import WeftError


_libc = ctypes.CDLL(None, use_errno=True)


class CompilationError(WeftError):
    pass


@dataclass(frozen=True)
class Toolchain:
    compiler: str = "weft-compile"
    cc: tuple[str, ...] = ("clang",)
    cflags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cc", tuple(self.cc))
        object.__setattr__(self, "cflags", tuple(self.cflags))
        if not self.compiler or not self.cc:
            raise ValueError("the Weft compiler and system C compiler must be explicit")

    def resolve(self) -> Toolchain:
        compiler = shutil.which(self.compiler)
        cc = shutil.which(self.cc[0])
        if compiler is None or cc is None:
            raise CompilationError(
                f"native toolchain is unavailable: Weft={self.compiler}, C={self.cc[0]}"
            )
        return Toolchain(os.path.abspath(compiler),
                         (os.path.abspath(cc), *self.cc[1:]), self.cflags)


def run_compiler(command: list[str], source: str | None = None) -> str:
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if result.returncode != 0:
        raise CompilationError(
            f"compiler exited with status {result.returncode}: {command!r}\n{result.stderr}"
        )
    return result.stdout


@dataclass(frozen=True)
class NativeTarget:
    march: str
    abi: str
    xlen: int
    vlen_bits: int
    cpus: frozenset[int]

    @classmethod
    def discover(cls, toolchain: Toolchain) -> NativeTarget:
        facts = json.loads(run_compiler([toolchain.compiler, "--query-native-target"]))
        target = cls(facts["march"], facts["abi"], facts["xlen"],
                     facts["vlen_bits"], frozenset(facts["cpus"]))
        if target.vlen_bits <= 0 or not target.cpus:
            raise CompilationError("native discovery returned incomplete hardware facts")
        target.check_execution()
        return target

    def check_execution(self) -> None:
        if sys.platform != "linux" or sys.byteorder != "little":
            raise CompilationError("native execution requires little-endian RISC-V Linux")
        if ctypes.sizeof(ctypes.c_void_p) * 8 != self.xlen:
            raise CompilationError("process pointer ABI disagrees with the compiled target")
        # The exec'ed compiler probe cannot establish the calling thread's state.
        control = _libc.prctl(ctypes.c_int(70), *(ctypes.c_ulong(0) for _ in range(4)))
        if control < 0:
            raise CompilationError(
                f"cannot query the calling thread's vector state: {os.strerror(ctypes.get_errno())}"
            )
        if (control & 3) == 1:
            raise CompilationError("RISC-V vector state is disabled in the calling thread")
        if not os.sched_getaffinity(0).issubset(self.cpus):
            raise CompilationError("current CPU affinity exceeds the discovered execution target")
