from __future__ import annotations

from dataclasses import dataclass

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


task_id = Intrinsic("task_id")
range = Intrinsic("range")
arange = Intrinsic("arange")
expand_dims = Intrinsic("expand_dims")
full_like = Intrinsic("full_like")
zeros_like = Intrinsic("zeros_like")
load = Intrinsic("load")
store = Intrinsic("store")
contract = Intrinsic("contract")
dot = Intrinsic("dot")
reduce = Intrinsic("reduce")
sum = Intrinsic("sum")
max = Intrinsic("max")
min = Intrinsic("min")
cast = Intrinsic("cast")
exp = Intrinsic("exp")
exp2 = Intrinsic("exp2")
log = Intrinsic("log")
rsqrt = Intrinsic("rsqrt")
maximum = Intrinsic("maximum")
minimum = Intrinsic("minimum")
