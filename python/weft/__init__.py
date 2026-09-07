from .api import (
    DerivedEncodingDefinition,
    EncodingDefinition,
    KernelDefinition,
    derive,
    encoding,
    kernel,
    overloads,
)
from .diagnostics import DefinitionError, FrontendError, LanguageUseError, WeftError


def compile(definition, *, options=None, toolchain=None):
    from .runtime import compile as native_compile

    return native_compile(definition, options=options, toolchain=toolchain)


def jit(definition, *, options=None, toolchain=None):
    from .runtime import jit as native_jit

    return native_jit(definition, options=options, toolchain=toolchain)


def lower_to_mlir(definition: KernelDefinition[object, object]) -> str:
    from .frontend import lower_to_mlir as lower

    return lower(definition)


__all__ = [
    "DefinitionError",
    "DerivedEncodingDefinition",
    "EncodingDefinition",
    "FrontendError",
    "KernelDefinition",
    "LanguageUseError",
    "WeftError",
    "derive",
    "compile",
    "encoding",
    "kernel",
    "jit",
    "lower_to_mlir",
    "overloads",
]
