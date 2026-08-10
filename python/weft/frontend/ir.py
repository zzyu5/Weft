from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field

from weft.diagnostics import SourceLocation

from .types import ValueType
from .types import emit_type


# This is a generic, one-way MLIR assembly builder. Operation names, operands,
# result types and regions map directly to generic MLIR syntax; this module owns
# no parallel Weft op schema or verifier. The registered C++ weft_kernel dialect
# remains the canonical semantic authority.
@dataclass(frozen=True, slots=True)
class Value:
    id: int
    type: ValueType
    name_hint: str | None = None

    @property
    def name(self) -> str:
        return f"%v{self.id}"


@dataclass(slots=True)
class Region:
    arguments: tuple[Value, ...]
    operations: list[Operation] = field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return bool(self.operations) and self.operations[-1].name in {
            "weft_kernel.return",
            "weft_kernel.yield",
        }


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: tuple[tuple[str, str], ...]
    regions: tuple[Region, ...]
    location: SourceLocation


class IRBuilder:
    def __init__(self) -> None:
        self._next_value_id = 0

    def value(self, value_type: ValueType, name_hint: str | None = None) -> Value:
        value = Value(self._next_value_id, value_type, name_hint)
        self._next_value_id += 1
        return value

    def region(
        self,
        argument_types: tuple[ValueType, ...],
        argument_names: tuple[str | None, ...] = (),
    ) -> Region:
        if argument_names and len(argument_names) != len(argument_types):
            raise ValueError("region argument names and types must match")
        names = argument_names or (None,) * len(argument_types)
        return Region(
            tuple(
                self.value(value_type, name)
                for value_type, name in zip(argument_types, names)
            )
        )

    def emit(
        self,
        block: Region,
        name: str,
        location: SourceLocation,
        *,
        operands: tuple[Value, ...] = (),
        result_types: tuple[ValueType, ...] = (),
        attributes: dict[str, str] | None = None,
        regions: tuple[Region, ...] = (),
        result_names: tuple[str | None, ...] = (),
    ) -> tuple[Value, ...]:
        if block.terminated:
            raise ValueError("cannot emit after a block terminator")
        if result_names and len(result_names) != len(result_types):
            raise ValueError("result names and types must match")
        names = result_names or (None,) * len(result_types)
        results = tuple(
            self.value(value_type, hint)
            for value_type, hint in zip(result_types, names)
        )
        block.operations.append(
            Operation(
                name=name,
                operands=tuple(operands),
                results=results,
                attributes=tuple((attributes or {}).items()),
                regions=tuple(regions),
                location=location,
            )
        )
        return results

    def module(
        self,
        module_name: str,
        kernel: Operation,
    ) -> str:
        lines = [
            f"module attributes {{weft_kernel.source_module = {json.dumps(module_name)}}} {{"
        ]
        lines.extend(_render_operation(kernel, 1))
        lines.append("}")
        return "\n".join(lines) + "\n"


def _render_operation(operation: Operation, indent: int) -> list[str]:
    prefix = "  " * indent
    result_prefix = ""
    if operation.results:
        result_prefix = ", ".join(value.name for value in operation.results) + " = "
    operands = ", ".join(value.name for value in operation.operands)
    lines = [f'{prefix}{result_prefix}"{operation.name}"({operands})']
    if operation.regions:
        lines[0] += " ("
        for region_index, region in enumerate(operation.regions):
            if region_index:
                lines[-1] += ","
            lines.extend(_render_region(region, indent + 1))
        lines.append(f"{prefix})")
    if operation.attributes:
        attributes = ", ".join(
            f"{name} = {value}" for name, value in operation.attributes
        )
        lines[-1] += f" {{{attributes}}}"
    operand_types = ", ".join(emit_type(value.type) for value in operation.operands)
    result_types = _emit_result_types(tuple(value.type for value in operation.results))
    location = operation.location
    lines[-1] += (
        f" : ({operand_types}) -> {result_types} "
        f"loc({json.dumps(location.filename)}:{location.line}:{location.column + 1})"
    )
    return lines


def _render_region(region: Region, indent: int) -> list[str]:
    prefix = "  " * indent
    arguments = ", ".join(
        f"{value.name}: {emit_type(value.type)}" for value in region.arguments
    )
    lines = [f"{prefix}{{", f"{prefix}  ^bb0({arguments}):"]
    for operation in region.operations:
        lines.extend(_render_operation(operation, indent + 2))
    lines.append(f"{prefix}}}")
    return lines


def _emit_result_types(types: tuple[ValueType, ...]) -> str:
    if not types:
        return "()"
    if len(types) == 1:
        return emit_type(types[0])
    return "(" + ", ".join(emit_type(value_type) for value_type in types) + ")"
