from __future__ import annotations

import builtins as python_builtins
import functools
import inspect
from dataclasses import dataclass
from types import FunctionType
from typing import Callable

from weft.diagnostics import DefinitionError
from weft.diagnostics import LanguageUseError


@dataclass(frozen=True, slots=True)
class Intrinsic:
    name: str
    canonical_name: str

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"W.{self.name} is valid only while lowering an @weft.kernel"
        )

    def __repr__(self) -> str:
        return f"W.{self.name}"


@dataclass(frozen=True, slots=True)
class InvalidValue:
    def __repr__(self) -> str:
        return "W.invalid"


invalid = InvalidValue()


class HelperDefinition:
    def __init__(self, function: Callable[..., object], effects: tuple[str, ...]) -> None:
        if not isinstance(function, FunctionType):
            raise DefinitionError("Weft helpers must decorate a Python function")
        allowed = {"read", "write", "atomic", "fence"}
        if any(effect not in allowed for effect in effects):
            raise DefinitionError(
                "helper effects must be read, write, atomic, or fence"
            )
        self.python_function = function
        self.effects = python_builtins.tuple(dict.fromkeys(effects))
        functools.update_wrapper(self, function)

    @property
    def pure(self) -> bool:
        return not self.effects

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.python_function, eval_str=True)

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"helper {self.__name__} is AOT Weft source and cannot run as Python"
        )


def pure(function: Callable[..., object]) -> HelperDefinition:
    return HelperDefinition(function, ())


def helper(
    function: Callable[..., object] | None = None,
    *,
    effects: tuple[str, ...] = (),
) -> object:
    if function is None:
        return lambda nested: HelperDefinition(nested, effects)
    return HelperDefinition(function, effects)


def _intrinsic(name: str, canonical_name: str | None = None) -> Intrinsic:
    return Intrinsic(name, canonical_name or f"weft_kernel.{name}")


range = _intrinsic("range")
vla = _intrinsic("vla")
select = _intrinsic("select")
load = _intrinsic("load")
store = _intrinsic("store")
prefetch = _intrinsic("prefetch")
atomic_add = _intrinsic("atomic_add")
fence = _intrinsic("fence")
valid = _intrinsic("valid")
fill = _intrinsic("fill")
block_axis = _intrinsic("block_axis")
full = _intrinsic("full")
zeros = _intrinsic("zeros")
expand_dims = _intrinsic("expand_dims")
broadcast_to = _intrinsic("broadcast_to")
reshape = _intrinsic("reshape")
transpose = _intrinsic("transpose")
reduce = _intrinsic("reduce")
scan = _intrinsic("scan")
summary_fold = _intrinsic("summary_fold")
contract = _intrinsic("contract")
dot = _intrinsic("dot")
permute = _intrinsic("permute")
lookup = _intrinsic("lookup")
decode = _intrinsic("decode")
widen = _intrinsic("widen")
narrow = _intrinsic("narrow")
cast = _intrinsic("cast")
bitcast = _intrinsic("bitcast")
tuple = _intrinsic("tuple")
maximum = _intrinsic("maximum")
minimum = _intrinsic("minimum")
exp = _intrinsic("exp")
exp2 = _intrinsic("exp2")
log = _intrinsic("log")
rsqrt = _intrinsic("rsqrt")
neg_inf = _intrinsic("neg_inf")
block_scaled_contract = _intrinsic(
    "block_scaled_contract", "weft_ext.block_scaled_contract"
)
