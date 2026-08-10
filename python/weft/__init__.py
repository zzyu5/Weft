from .api import KernelDefinition
from .api import kernel
from .diagnostics import DefinitionError
from .diagnostics import FrontendError
from .diagnostics import LanguageUseError
from .diagnostics import WeftError


def lower_to_mlir(definition: KernelDefinition[object, object]) -> str:
    from .frontend import lower_to_mlir as lower

    return lower(definition)


__all__ = [
    "DefinitionError",
    "FrontendError",
    "KernelDefinition",
    "LanguageUseError",
    "WeftError",
    "kernel",
    "lower_to_mlir",
]
