from __future__ import annotations

import functools
import inspect
import textwrap
from dataclasses import dataclass
from types import FunctionType
from typing import Any
from typing import Callable
from typing import Generic
from typing import ParamSpec
from typing import TypeVar

from weft.diagnostics import DefinitionError
from weft.diagnostics import LanguageUseError


P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class DefinitionSource:
    filename: str
    first_line: int


class KernelDefinition(Generic[P, R]):
    """Captured Python text for one worker-local Weft DSL kernel."""

    def __init__(self, function: Callable[P, R]) -> None:
        if not isinstance(function, FunctionType):
            raise DefinitionError("Weft kernels must decorate a Python function")
        self.python_function = function
        try:
            _, source_start = inspect.getsourcelines(function)
        except (OSError, TypeError) as error:
            raise DefinitionError(
                "Weft must be able to inspect the decorated DSL kernel"
            ) from error
        self.source = DefinitionSource(
            filename=inspect.getsourcefile(function) or function.__code__.co_filename,
            first_line=source_start,
        )
        functools.update_wrapper(self, function)

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.python_function, eval_str=True)

    def source_text(self) -> str:
        return textwrap.dedent(inspect.getsource(self.python_function))

    def lower(self) -> str:
        from weft.frontend import lower_to_mlir

        return lower_to_mlir(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise LanguageUseError(
            f"kernel {self.__name__} is an AOT Weft DSL kernel and cannot run as Python"
        )

    def __repr__(self) -> str:
        return f"<weft kernel {self.__module__}.{self.__qualname__}>"


def kernel(function: Callable[P, R] | None = None) -> Any:
    if function is None:
        return KernelDefinition
    return KernelDefinition(function)
