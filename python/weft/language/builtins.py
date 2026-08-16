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
        if function.__annotations__:
            raise DefinitionError(
                "Weft helper annotations do not declare types; omit them"
            )
        allowed = {"read", "write"}
        if any(effect not in allowed for effect in effects):
            raise DefinitionError(
                "helper effects must be read or write"
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
            f"helper {self.__name__} belongs to an AOT Weft DSL kernel and cannot run as Python"
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


def _intrinsic(name: str) -> Intrinsic:
    return Intrinsic(name)


range = _intrinsic("range")
vla = _intrinsic("vla")
select = _intrinsic("select")
load = _intrinsic("load")
load_f16_le = _intrinsic("load_f16_le")
store = _intrinsic("store")
storage = _intrinsic("storage")
sort_indices = _intrinsic("sort_indices")
block = _intrinsic("block")
full = _intrinsic("full")
zeros = _intrinsic("zeros")
reduce = _intrinsic("reduce")
scan = _intrinsic("scan")
argmax = _intrinsic("argmax")
online_softmax_summary = _intrinsic("online_softmax_summary")
dot = _intrinsic("dot")
matmul = _intrinsic("matmul")
lookup = _intrinsic("lookup")
decode = _intrinsic("decode")
narrow = _intrinsic("narrow")
cast = _intrinsic("cast")
bitcast = _intrinsic("bitcast")
tuple = _intrinsic("tuple")
maximum = _intrinsic("maximum")
minimum = _intrinsic("minimum")
exp = _intrinsic("exp")
tanh = _intrinsic("tanh")
log = _intrinsic("log")
sin = _intrinsic("sin")
cos = _intrinsic("cos")
floor = _intrinsic("floor")
sqrt = _intrinsic("sqrt")
rsqrt = _intrinsic("rsqrt")
neg_inf = _intrinsic("neg_inf")
affine_i4_i8_dot = _intrinsic("affine_i4_i8_dot")
symmetric_i4_i8_dot = _intrinsic("symmetric_i4_i8_dot")
grouped_affine_i4_i8_dot = _intrinsic("grouped_affine_i4_i8_dot")
sign_bit_i8_dot = _intrinsic("sign_bit_i8_dot")
e2m1_e8m0_i8_dot = _intrinsic("e2m1_e8m0_i8_dot")
packed_i4_i8_dot = _intrinsic("packed_i4_i8_dot")
packed_i5_i8_dot = _intrinsic("packed_i5_i8_dot")
base3_ternary_i8_dot = _intrinsic("base3_ternary_i8_dot")
packed_i2_ternary_i8_dot = _intrinsic("packed_i2_ternary_i8_dot")
signed_codebook_i8_dot = _intrinsic("signed_codebook_i8_dot")
packed_u9_u7_codebook_i8_dot = _intrinsic("packed_u9_u7_codebook_i8_dot")
packed_u11_grid_delta_i8_dot = _intrinsic("packed_u11_grid_delta_i8_dot")
iq2_s_i8_dot = _intrinsic("iq2_s_i8_dot")
iq3_s_i8_dot = _intrinsic("iq3_s_i8_dot")
iq1_m_i8_dot = _intrinsic("iq1_m_i8_dot")
q6_k_i8_dot = _intrinsic("q6_k_i8_dot")
