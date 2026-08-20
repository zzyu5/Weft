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
    "encoding",
    "kernel",
    "lower_to_mlir",
    "overloads",
]
