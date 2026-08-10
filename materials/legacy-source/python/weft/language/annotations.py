from __future__ import annotations

from dataclasses import dataclass

from .dtypes import DType
from .dtypes import index


@dataclass(frozen=True, slots=True)
class PtrSpec:
    dtype: DType
    address_space: str = "global"

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise TypeError("ptr element type must be a Weft dtype")
        if not isinstance(self.address_space, str) or not self.address_space:
            raise TypeError("ptr address space must be a non-empty string")


class ptr:
    @classmethod
    def __class_getitem__(cls, parameters: object) -> PtrSpec:
        if isinstance(parameters, DType):
            return PtrSpec(parameters)
        if isinstance(parameters, tuple) and len(parameters) == 2:
            return PtrSpec(parameters[0], parameters[1])
        raise TypeError("ptr annotation expects W.ptr[dtype] or W.ptr[dtype, address_space]")


@dataclass(frozen=True, slots=True)
class ConstexprSpec:
    dtype: DType

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise TypeError("constexpr value type must be a Weft dtype")


class Constexpr:
    @classmethod
    def __class_getitem__(cls, dtype: object) -> ConstexprSpec:
        if not isinstance(dtype, DType):
            raise TypeError("Constexpr annotation expects a Weft dtype")
        return ConstexprSpec(dtype)


constexpr = ConstexprSpec(index)
