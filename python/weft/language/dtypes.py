from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DTypeCategory(Enum):
    BOOL = "bool"
    INTEGER = "integer"
    INDEX = "index"
    FLOAT = "float"
    BFLOAT = "bfloat"


@dataclass(frozen=True, slots=True)
class DType:
    name: str
    category: DTypeCategory
    bits: int | None
    signedness: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dtype name must not be empty")
        if self.category is DTypeCategory.INDEX:
            if self.bits is not None:
                raise ValueError("index has target-dependent width")
        elif self.bits is None or self.bits <= 0:
            raise ValueError("non-index dtype requires a positive bit width")
        if self.category is DTypeCategory.INTEGER:
            if self.signedness not in {"signed", "unsigned"}:
                raise ValueError("integer dtype requires signed or unsigned semantics")
        elif self.signedness is not None:
            raise ValueError("only integer dtypes carry signedness")

    def __repr__(self) -> str:
        return f"W.{self.name}"


bool = DType("bool", DTypeCategory.BOOL, 1)
index = DType("index", DTypeCategory.INDEX, None)
i8 = DType("i8", DTypeCategory.INTEGER, 8, "signed")
i16 = DType("i16", DTypeCategory.INTEGER, 16, "signed")
i32 = DType("i32", DTypeCategory.INTEGER, 32, "signed")
i64 = DType("i64", DTypeCategory.INTEGER, 64, "signed")
u8 = DType("u8", DTypeCategory.INTEGER, 8, "unsigned")
u16 = DType("u16", DTypeCategory.INTEGER, 16, "unsigned")
u32 = DType("u32", DTypeCategory.INTEGER, 32, "unsigned")
u64 = DType("u64", DTypeCategory.INTEGER, 64, "unsigned")
f16 = DType("f16", DTypeCategory.FLOAT, 16)
bf16 = DType("bf16", DTypeCategory.BFLOAT, 16)
f32 = DType("f32", DTypeCategory.FLOAT, 32)
f64 = DType("f64", DTypeCategory.FLOAT, 64)
