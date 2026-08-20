from __future__ import annotations

from dataclasses import dataclass

from .dtypes import DType


@dataclass(frozen=True, slots=True)
class LayoutAtom:
    category: str
    value: str


class _BitOrder:
    lsb_first = LayoutAtom("bit_order", "lsb_first")
    msb_first = LayoutAtom("bit_order", "msb_first")


class _ByteOrder:
    little = LayoutAtom("byte_order", "little")
    big = LayoutAtom("byte_order", "big")


bitorder = _BitOrder()
byteorder = _ByteOrder()
lo_first = "lo_first"
hi_first = "hi_first"


@dataclass(frozen=True, slots=True)
class PackingSpec:
    kind: str
    argument: str


def packed(storage_bytes: int) -> PackingSpec:
    if isinstance(storage_bytes, bool) or not isinstance(storage_bytes, int) or storage_bytes <= 0:
        raise TypeError("packed expects a positive byte count")
    return PackingSpec("packed", str(storage_bytes))


def nibble(order: str) -> PackingSpec:
    if order not in {lo_first, hi_first}:
        raise TypeError("nibble order must be lo_first or hi_first")
    return PackingSpec("nibble", order)


@dataclass(frozen=True, slots=True)
class PaddingSpec:
    bytes: int
    value: int = 0


def padding(bytes: int, value: int = 0) -> PaddingSpec:
    if isinstance(bytes, bool) or not isinstance(bytes, int) or bytes <= 0:
        raise TypeError("padding expects a positive byte count")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise TypeError("padding fill must be a byte")
    return PaddingSpec(bytes, value)


@dataclass(frozen=True, slots=True)
class ArraySpec:
    dtype: DType
    shape: tuple[object, ...]
    packing: PackingSpec | None = None

    def __matmul__(self, packing: object) -> ArraySpec:
        if not isinstance(packing, PackingSpec):
            raise TypeError("encoding field @ annotation must be packed(...) or nibble(...)")
        return ArraySpec(self.dtype, self.shape, packing)


@dataclass(frozen=True, slots=True)
class ViewSpec:
    encoding: object
    shape: tuple[object, ...]


class View:
    @classmethod
    def __class_getitem__(cls, parameters: object) -> ViewSpec:
        if not isinstance(parameters, tuple) or len(parameters) != 2:
            raise TypeError("View expects View[Encoding, shape]")
        encoding, shape = parameters
        return ViewSpec(encoding, shape if isinstance(shape, tuple) else (shape,))


@dataclass(frozen=True, slots=True)
class AutoSpec:
    choices: tuple[object, ...]

    @property
    def spelling(self) -> str:
        return "|".join(str(choice) for choice in self.choices)


def auto(*choices: object) -> AutoSpec:
    if not choices:
        raise TypeError("auto expects at least one candidate or one symbolic parameter")
    return AutoSpec(tuple(choices))
