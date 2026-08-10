from __future__ import annotations

import json
from dataclasses import dataclass

from weft.language import DType
from weft.language import DTypeCategory


class ValueType:
    pass


@dataclass(frozen=True, slots=True)
class ScalarType(ValueType):
    dtype: DType


@dataclass(frozen=True, slots=True)
class PointerType(ValueType):
    element_type: ScalarType
    address_space: str


@dataclass(frozen=True, slots=True)
class BlockType(ValueType):
    shape: tuple[int, ...]
    element_type: ValueType


@dataclass(frozen=True, slots=True)
class ConstexprType(ValueType):
    value_type: ScalarType


def emit_type(value_type: ValueType) -> str:
    if isinstance(value_type, ScalarType):
        if value_type.dtype.name == "bool":
            return "i1"
        if value_type.dtype.category is DTypeCategory.INTEGER:
            prefix = "s" if value_type.dtype.signedness == "signed" else "u"
            return f"{prefix}i{value_type.dtype.bits}"
        return value_type.dtype.name
    if isinstance(value_type, PointerType):
        return (
            f"!weft_kernel.ptr<{emit_type(value_type.element_type)}, "
            f"{json.dumps(value_type.address_space)}>"
        )
    if isinstance(value_type, BlockType):
        shape = ", ".join(str(dimension) for dimension in value_type.shape)
        return f"!weft_kernel.block<[{shape}], {emit_type(value_type.element_type)}>"
    if isinstance(value_type, ConstexprType):
        return f"!weft_kernel.constexpr<{emit_type(value_type.value_type)}>"
    raise TypeError(f"cannot emit unknown Weft type {value_type!r}")


def element_type(value_type: ValueType) -> ValueType:
    if isinstance(value_type, BlockType):
        return value_type.element_type
    return value_type


def block_shape(value_type: ValueType) -> tuple[int, ...] | None:
    if isinstance(value_type, BlockType):
        return value_type.shape
    return None


def with_element_type(value_type: ValueType, new_element: ValueType) -> ValueType:
    if isinstance(value_type, BlockType):
        return BlockType(value_type.shape, new_element)
    return new_element
