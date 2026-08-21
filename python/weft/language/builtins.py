from __future__ import annotations

from dataclasses import dataclass

from weft.diagnostics import LanguageUseError


@dataclass(frozen=True, slots=True)
class Intrinsic:
    name: str

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"W.{self.name} is valid only while lowering a Weft definition"
        )


@dataclass(frozen=True, slots=True)
class EngineRole:
    name: str

    def __rmatmul__(self, value: object) -> object:
        raise LanguageUseError("engine-role binding is compile-time Weft syntax")


scalar = EngineRole("scalar")
wide = EngineRole("wide")
matrix = EngineRole("matrix")
transfer = EngineRole("transfer")


@dataclass(frozen=True, slots=True)
class LevelConstructor:
    relation: str

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError("Weft levels exist only while lowering a kernel")


class LevelNamespace:
    rows = LevelConstructor("rows")
    cols = LevelConstructor("cols")
    tiles = LevelConstructor("tiles")
    blocks = LevelConstructor("blocks")
    subs = LevelConstructor("subs")


L = LevelNamespace()


def _intrinsic(name: str) -> Intrinsic:
    return Intrinsic(name)


new = _intrinsic("new")
materialize = _intrinsic("materialize")
admit = _intrinsic("admit")
commit = _intrinsic("commit")
handoff = _intrinsic("handoff")

mac_pairs = _intrinsic("mac_pairs")
mac_groups = _intrinsic("mac_groups")
widen = _intrinsic("widen")
narrow = _intrinsic("narrow")
reduce = _intrinsic("reduce")
fold2 = _intrinsic("fold2")
dot = _intrinsic("dot")
contract = _intrinsic("contract")
outer_contract = _intrinsic("outer_contract")
lookup = _intrinsic("lookup")
pack = _intrinsic("pack")
interleave = _intrinsic("interleave")

maximum = _intrinsic("maximum")
minimum = _intrinsic("minimum")
abs = _intrinsic("abs")
exp = _intrinsic("exp")
