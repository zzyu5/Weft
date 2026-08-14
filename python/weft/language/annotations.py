from __future__ import annotations

from dataclasses import dataclass

from .dtypes import DType


@dataclass(frozen=True, slots=True)
class PointerQualifier:
    kind: str
    value: object = True


readonly = PointerQualifier("access", "read")
writeonly = PointerQualifier("access", "write")
noalias = PointerQualifier("noalias")
restrict = PointerQualifier("restrict")
external = PointerQualifier("storage", ("external", ""))
workspace = PointerQualifier("storage", ("workspace", ""))


def persistent(storage_format: str) -> PointerQualifier:
    if not isinstance(storage_format, str) or not storage_format:
        raise TypeError("W.persistent expects a non-empty storage format identity")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if any(character not in allowed for character in storage_format):
        raise TypeError(
            "W.persistent storage format identity uses only letters, digits, '_', '-', or '.'"
        )
    return PointerQualifier("storage", ("persistent", storage_format))


def aligned(minimum: int) -> PointerQualifier:
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        raise TypeError("W.aligned expects a positive integer")
    if minimum & (minimum - 1):
        raise TypeError("W.aligned expects a power of two")
    return PointerQualifier("alignment", minimum)


def address_space(name: str) -> PointerQualifier:
    if not isinstance(name, str) or not name:
        raise TypeError("W.address_space expects a non-empty string")
    return PointerQualifier("address_space", name)


@dataclass(frozen=True, slots=True)
class PtrSpec:
    dtype: DType
    address_space: str = "global"
    access: str = "readwrite"
    noalias: bool = False
    alignment: int = 0
    restrict_like: bool = False
    storage_class: str = "external"
    storage_format: str = ""


class ptr:
    @classmethod
    def __class_getitem__(cls, parameters: object) -> PtrSpec:
        items = parameters if isinstance(parameters, tuple) else (parameters,)
        if not items or not isinstance(items[0], DType):
            raise TypeError("W.ptr expects a Weft dtype as its first parameter")
        address = "global"
        access = "readwrite"
        alias = False
        alignment = 0
        restrict_like = False
        storage_class = "external"
        storage_format = ""
        seen: set[str] = set()
        for qualifier in items[1:]:
            if not isinstance(qualifier, PointerQualifier):
                raise TypeError("W.ptr qualifiers must be Weft qualifier objects")
            if qualifier.kind in seen and qualifier.kind != "noalias":
                raise TypeError(f"duplicate pointer qualifier {qualifier.kind!r}")
            seen.add(qualifier.kind)
            if qualifier.kind == "address_space":
                address = str(qualifier.value)
            elif qualifier.kind == "access":
                access = str(qualifier.value)
            elif qualifier.kind == "noalias":
                alias = True
            elif qualifier.kind == "alignment":
                alignment = int(qualifier.value)
            elif qualifier.kind == "restrict":
                restrict_like = True
            elif qualifier.kind == "storage":
                storage_class, storage_format = qualifier.value
            else:
                raise TypeError(f"unknown pointer qualifier {qualifier.kind!r}")
        if storage_class == "workspace" and not alias:
            raise TypeError("W.workspace pointers must also declare W.noalias")
        return PtrSpec(
            dtype=items[0],
            address_space=address,
            access=access,
            noalias=alias,
            alignment=alignment,
            restrict_like=restrict_like,
            storage_class=storage_class,
            storage_format=storage_format,
        )


@dataclass(frozen=True, slots=True)
class ConstexprSpec:
    dtype: DType


class constexpr:
    @classmethod
    def __class_getitem__(cls, dtype: object) -> ConstexprSpec:
        if not isinstance(dtype, DType):
            raise TypeError("W.constexpr expects a Weft dtype")
        return ConstexprSpec(dtype)
