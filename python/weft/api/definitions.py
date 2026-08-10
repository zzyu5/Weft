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
    def __init__(self, function: Callable[P, R], *, grid_rank: int = 1) -> None:
        if not isinstance(function, FunctionType):
            raise DefinitionError("Weft kernels must decorate a Python function")
        if isinstance(grid_rank, bool) or not isinstance(grid_rank, int) or grid_rank <= 0:
            raise DefinitionError("Weft kernel grid_rank must be a positive integer")
        self.python_function = function
        self.grid_rank = grid_rank
        _, source_start = inspect.getsourcelines(function)
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

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise LanguageUseError(
            f"kernel {self.__name__} is compile-time Weft code and cannot run as Python"
        )

    def __repr__(self) -> str:
        return f"<weft kernel {self.__module__}.{self.__qualname__}>"


def kernel(
    function: Callable[P, R] | None = None,
    *,
    grid_rank: int = 1,
) -> Any:
    if function is None:
        return lambda nested: KernelDefinition(nested, grid_rank=grid_rank)
    return KernelDefinition(function, grid_rank=grid_rank)
