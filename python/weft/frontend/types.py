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
    access: str
    noalias: bool
    alignment: int
    restrict_like: bool
    storage_class: str
    storage_format: str


@dataclass(frozen=True, slots=True)
class ConstexprType(ValueType):
    value_type: ScalarType


@dataclass(frozen=True, slots=True)
class BlockType(ValueType):
    shape: tuple[int, ...]
    axes: tuple[int, ...]
    element_type: ValueType


@dataclass(frozen=True, slots=True)
class RegionType(ValueType):
    shape: tuple[int, ...]
    axes: tuple[int, ...]
    element_type: ValueType


@dataclass(frozen=True, slots=True)
class MaskedType(ValueType):
    value_type: ValueType


@dataclass(frozen=True, slots=True)
class TupleType(ValueType):
    fields: tuple[ValueType, ...]


@dataclass(frozen=True, slots=True)
class NoneType(ValueType):
    pass


NONE_TYPE = NoneType()


def emit_type(value_type: ValueType) -> str:
    if isinstance(value_type, ScalarType):
        dtype = value_type.dtype
        if dtype.category is DTypeCategory.BOOL:
            return "i1"
        if dtype.category is DTypeCategory.INDEX:
            return "index"
        if dtype.category is DTypeCategory.INTEGER:
            prefix = "s" if dtype.signedness == "signed" else "u"
            return f"{prefix}i{dtype.bits}"
        return dtype.name
    if isinstance(value_type, PointerType):
        return (
            f"!weft_kernel.ptr<{emit_type(value_type.element_type)}, "
            f"{json.dumps(value_type.address_space)}, {json.dumps(value_type.access)}, "
            f"{str(value_type.noalias).lower()}, {value_type.alignment}, "
            f"{str(value_type.restrict_like).lower()}, "
            f"{json.dumps(value_type.storage_class)}, "
            f"{json.dumps(value_type.storage_format)}>"
        )
    if isinstance(value_type, ConstexprType):
        return f"!weft_kernel.constexpr<{emit_type(value_type.value_type)}>"
    if isinstance(value_type, BlockType):
        shape = ", ".join(str(dimension) for dimension in value_type.shape)
        axes = ", ".join(str(axis) for axis in value_type.axes)
        return (
            f"!weft_kernel.block<[{shape}], [{axes}], "
            f"{emit_type(value_type.element_type)}>"
        )
    if isinstance(value_type, RegionType):
        shape = ", ".join(str(dimension) for dimension in value_type.shape)
        axes = ", ".join(str(axis) for axis in value_type.axes)
        return (
            f"!weft_kernel.region<[{shape}], [{axes}], "
            f"{emit_type(value_type.element_type)}>"
        )
    if isinstance(value_type, MaskedType):
        return f"!weft_kernel.masked<{emit_type(value_type.value_type)}>"
    if isinstance(value_type, TupleType):
        fields = ", ".join(emit_type(field) for field in value_type.fields)
        return f"!weft_kernel.tuple<[{fields}]>"
    if isinstance(value_type, NoneType):
        return "none"
    raise TypeError(f"cannot emit unknown Weft type {value_type!r}")


def bare_type(value_type: ValueType) -> ValueType:
    return value_type.value_type if isinstance(value_type, MaskedType) else value_type


def is_masked(value_type: ValueType) -> bool:
    return isinstance(value_type, MaskedType)


def element_type(value_type: ValueType) -> ValueType:
    value_type = bare_type(value_type)
    if isinstance(value_type, (BlockType, RegionType)):
        return value_type.element_type
    return value_type


def shape_of(value_type: ValueType) -> tuple[int, ...] | None:
    value_type = bare_type(value_type)
    if isinstance(value_type, (BlockType, RegionType)):
        return value_type.shape
    return None


def axes_of(value_type: ValueType) -> tuple[int, ...] | None:
    value_type = bare_type(value_type)
    if isinstance(value_type, (BlockType, RegionType)):
        return value_type.axes
    return None


def shape_kind(value_type: ValueType) -> str:
    value_type = bare_type(value_type)
    if isinstance(value_type, RegionType):
        return "region"
    if isinstance(value_type, BlockType):
        return "block"
    return "scalar"


def shaped_type(
    kind: str,
    shape: tuple[int, ...],
    axes: tuple[int, ...],
    element: ValueType,
) -> ValueType:
    if kind == "region":
        return RegionType(shape, axes, element)
    if kind == "block":
        return BlockType(shape, axes, element)
    if shape or axes:
        raise ValueError("scalar type cannot carry a logical domain")
    return element


def with_element_type(value_type: ValueType, new_element: ValueType) -> ValueType:
    masked = is_masked(value_type)
    bare = bare_type(value_type)
    if isinstance(bare, BlockType):
        result: ValueType = BlockType(bare.shape, bare.axes, new_element)
    elif isinstance(bare, RegionType):
        result = RegionType(bare.shape, bare.axes, new_element)
    else:
        result = new_element
    return MaskedType(result) if masked else result
