from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Iterable
from typing import Sequence

from weft.api import KernelDefinition
from weft.diagnostics import FrontendError
from weft.diagnostics import SourceLocation
from weft.language import ConstexprSpec
from weft.language import DType
from weft.language import DTypeCategory
from weft.language import HelperDefinition
from weft.language import Intrinsic
from weft.language import PtrSpec
from weft.language import f16
from weft.language import f32
from weft.language import i1
from weft.language import i16
from weft.language import i32
from weft.language import i8
from weft.language import index
from weft.language import u8
from weft.language import u16
from weft.language import u32
from weft.language.builtins import InvalidValue

from .ir import IRBuilder
from .ir import Operation
from .ir import Region
from .ir import Value
from .source import HelperSource
from .source import SourceUnit
from .types import NONE_TYPE
from .types import BlockType
from .types import ConstexprType
from .types import MaskedType
from .types import NoneType
from .types import PointerType
from .types import RegionType
from .types import ScalarType
from .types import TupleType
from .types import ValueType
from .types import axes_of
from .types import bare_type
from .types import element_type
from .types import emit_type
from .types import is_masked
from .types import shape_kind
from .types import shape_of
from .types import shaped_type
from .types import with_element_type


@dataclass(frozen=True, slots=True)
class _ShapeSpec:
    dimensions: tuple[int, ...]
    extents: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class _BlockShapeSpec:
    dimensions: tuple[int, ...]
    axis_ids: tuple[int, ...]
    coordinates: tuple[Value, ...]


class _AssignedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


class _LoadedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _assigned_names(statements: Sequence[ast.stmt]) -> set[str]:
    visitor = _AssignedNames()
    for statement in statements:
        visitor.visit(statement)
    return visitor.names


def _loaded_names(statements: Sequence[ast.stmt]) -> set[str]:
    visitor = _LoadedNames()
    for statement in statements:
        visitor.visit(statement)
    return visitor.names


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for element in target.elts:
            result.update(_target_names(element))
        return result
    return set()


def _statement_uses_defs(statement: ast.stmt) -> tuple[set[str], set[str]]:
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        value = statement.value
        uses = _loaded_names([ast.Expr(value)]) if value is not None else set()
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        definitions: set[str] = set()
        for target in targets:
            definitions.update(_target_names(target))
        return uses, definitions
    if isinstance(statement, ast.AugAssign):
        names = _target_names(statement.target)
        return names | _loaded_names([ast.Expr(statement.value)]), names
    if isinstance(statement, ast.Expr):
        return _loaded_names([statement]), set()
    if isinstance(statement, ast.With):
        context_uses = set()
        bound = set()
        for item in statement.items:
            context_uses.update(_loaded_names([ast.Expr(item.context_expr)]))
            if item.optional_vars is not None:
                bound.update(_target_names(item.optional_vars))
        body_uses = _names_needed_before_definition(statement.body) - bound
        return context_uses | body_uses, _assigned_names(statement.body) | bound
    if isinstance(statement, ast.For):
        target = _target_names(statement.target)
        uses = _loaded_names([ast.Expr(statement.iter)])
        uses.update(_names_needed_before_definition(statement.body) - target)
        return uses, _assigned_names(statement.body) | target
    if isinstance(statement, ast.If):
        uses = _loaded_names([ast.Expr(statement.test)])
        uses.update(_names_needed_before_definition(statement.body))
        uses.update(_names_needed_before_definition(statement.orelse))
        then_defs = _assigned_names(statement.body)
        else_defs = _assigned_names(statement.orelse)
        return uses, then_defs & else_defs
    if isinstance(statement, ast.While):
        uses = _loaded_names([ast.Expr(statement.test)])
        uses.update(_names_needed_before_definition(statement.body))
        return uses, _assigned_names(statement.body)
    return _loaded_names([statement]), _assigned_names([statement])


def _names_needed_before_definition(statements: Sequence[ast.stmt]) -> set[str]:
    needed: set[str] = set()
    defined: set[str] = set()
    for statement in statements:
        uses, definitions = _statement_uses_defs(statement)
        needed.update(uses - defined)
        defined.update(definitions)
    return needed


def _dense_i64(values: Iterable[int]) -> str:
    values = tuple(values)
    if not values:
        return "array<i64>"
    return "array<i64: " + ", ".join(str(value) for value in values) + ">"


def _string(value: str) -> str:
    return json.dumps(value)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _dtype_type(dtype: DType) -> ScalarType:
    return ScalarType(dtype)


def _annotation_type(annotation: object, location: SourceLocation) -> tuple[ValueType, str]:
    if isinstance(annotation, DType):
        return ScalarType(annotation), "scalar"
    if isinstance(annotation, PtrSpec):
        return (
            PointerType(
                ScalarType(annotation.dtype),
                annotation.access,
                annotation.noalias,
                annotation.alignment,
                annotation.restrict_like,
                annotation.storage_class,
                annotation.storage_format,
            ),
            "pointer",
        )
    if isinstance(annotation, ConstexprSpec):
        return ConstexprType(ScalarType(annotation.dtype)), "constexpr"
    raise FrontendError(
        "kernel parameters require a W scalar, W.ptr, or W.constexpr annotation",
        location,
    )


def _is_scalar(value_type: ValueType) -> bool:
    return isinstance(bare_type(value_type), ScalarType)


def _is_index(value_type: ValueType) -> bool:
    if is_masked(value_type):
        return False
    element = element_type(value_type)
    return isinstance(element, ScalarType) and element.dtype is index


def _is_predicate(value_type: ValueType) -> bool:
    if is_masked(value_type):
        return False
    element = element_type(value_type)
    return isinstance(element, ScalarType) and element.dtype is i1


def _is_pointer(value_type: ValueType) -> bool:
    return isinstance(element_type(value_type), PointerType)


def _join_shapes(
    lhs: ValueType, rhs: ValueType, location: SourceLocation
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    lhs_kind = shape_kind(lhs)
    rhs_kind = shape_kind(rhs)
    lhs_shape = shape_of(lhs)
    rhs_shape = shape_of(rhs)
    lhs_axes = axes_of(lhs)
    rhs_axes = axes_of(rhs)
    if lhs_kind == "scalar":
        return rhs_kind, rhs_shape or (), rhs_axes or ()
    if rhs_kind == "scalar":
        return lhs_kind, lhs_shape or (), lhs_axes or ()
    if (
        lhs_shape is None
        or rhs_shape is None
        or lhs_axes is None
        or rhs_axes is None
        or len(lhs_shape) != len(rhs_shape)
    ):
        raise FrontendError("logical ranks are not broadcast-compatible", location)
    result: list[int] = []
    result_axes: list[int] = []
    for left, right, left_axis, right_axis in zip(
        lhs_shape, rhs_shape, lhs_axes, rhs_axes
    ):
        if left == 1 and left_axis == 0:
            result.append(right)
            result_axes.append(right_axis)
        elif right == 1 and right_axis == 0:
            result.append(left)
            result_axes.append(left_axis)
        elif left == right and left_axis == right_axis:
            result.append(left)
            result_axes.append(left_axis)
        else:
            raise FrontendError(
                "logical values do not share the same block axes", location
            )
    kind = "region" if "region" in {lhs_kind, rhs_kind} else "block"
    return kind, tuple(result), tuple(result_axes)


class FrontendCompiler:
    def __init__(self, definition: KernelDefinition[object, object]) -> None:
        self.unit = SourceUnit.from_definition(definition)
        self.source: SourceUnit | HelperSource = self.unit
        self.builder = IRBuilder()
        self.block: Region | None = None
        self.entry_block: Region | None = None
        self.env: dict[str, Value] = {}
        self.active_vla = False
        self.vla_outer_names: set[str] | None = None
        self.helper_effects: tuple[str, ...] | None = None
        self.constant_values: dict[Value, object] = {}
        self.storage_shapes: dict[Value, _ShapeSpec] = {}
        self.block_axes: dict[Value, tuple[int, Value]] = {}
        self.next_axis_id = 1

    def compile(self) -> str:
        signature = self.unit.definition.signature
        function_args = self.unit.function.args
        if (
            function_args.posonlyargs
            or function_args.kwonlyargs
            or function_args.vararg is not None
            or function_args.kwarg is not None
            or function_args.defaults
            or function_args.kw_defaults
        ):
            raise FrontendError(
                "kernel ABI uses plain annotated positional parameters without defaults",
                self.unit.location(self.unit.function),
            )
        return_annotation = signature.return_annotation
        if return_annotation in {None, type(None)}:
            return_type: ValueType = NONE_TYPE
        elif isinstance(return_annotation, DType):
            return_type = ScalarType(return_annotation)
        else:
            raise FrontendError(
                "kernel return annotation must be None or a Weft scalar dtype",
                self.unit.location(self.unit.function),
            )
        parameters = tuple(signature.parameters.values())
        argument_types: list[ValueType] = []
        argument_kinds: list[str] = []
        for parameter, ast_parameter in zip(parameters, self.unit.function.args.args):
            if parameter.annotation is parameter.empty:
                raise FrontendError(
                    f"kernel parameter {parameter.name!r} has no Weft annotation",
                    self.unit.location(ast_parameter),
                )
            value_type, kind = _annotation_type(
                parameter.annotation, self.unit.location(ast_parameter)
            )
            argument_types.append(value_type)
            argument_kinds.append(kind)

        body = self.builder.region(
            tuple(argument_types), tuple(parameter.name for parameter in parameters)
        )
        self.block = body
        self.entry_block = body
        for parameter, argument, kind in zip(parameters, body.arguments, argument_kinds):
            if kind == "constexpr":
                constexpr_type = argument.type
                assert isinstance(constexpr_type, ConstexprType)
                materialized = self._emit(
                    "weft_kernel.meta_value",
                    self.unit.function,
                    operands=(argument,),
                    result_types=(constexpr_type.value_type,),
                    result_names=(parameter.name,),
                )[0]
                self.env[parameter.name] = materialized
            else:
                self.env[parameter.name] = argument

        if isinstance(return_type, NoneType):
            self._compile_statements(self.unit.function.body)
            if not body.terminated:
                self._emit("weft_kernel.return", self.unit.function)
        else:
            if not self.unit.function.body or not isinstance(
                self.unit.function.body[-1], ast.Return
            ):
                raise FrontendError(
                    "a value-returning kernel must end in one scalar return",
                    self.unit.location(self.unit.function),
                )
            self._compile_statements(self.unit.function.body[:-1])
            return_node = self.unit.function.body[-1]
            assert isinstance(return_node, ast.Return)
            if return_node.value is None:
                raise FrontendError(
                    "kernel return requires a value", self.unit.location(return_node)
                )
            result = self._expect_value(
                self._compile_expr(return_node.value), return_node.value
            )
            if result.type != return_type:
                raise FrontendError(
                    "kernel return value must match its scalar annotation",
                    self.unit.location(return_node.value),
                )
            self._emit("weft_kernel.return", return_node, operands=(result,))

        location = self.unit.location(self.unit.function)
        source_name = f"{self.unit.filename}:{self.unit.first_line}"
        kernel = self.builder.operation(
            "weft_kernel.kernel",
            location,
            attributes={
                "sym_name": _string(self.unit.definition.__name__),
                "arg_names": "["
                + ", ".join(_string(parameter.name) for parameter in parameters)
                + "]",
                "arg_kinds": "["
                + ", ".join(_string(kind) for kind in argument_kinds)
                + "]",
                "return_type": emit_type(return_type),
                "source": _string(source_name),
            },
            regions=(body,),
        )
        return self.builder.module(self.unit.definition.__module__, kernel)

    def _location(self, node: ast.AST) -> SourceLocation:
        return self.source.location(node)

    def _resolve_static(self, node: ast.AST) -> object:
        if isinstance(node, ast.Name):
            if node.id in self.source.bindings:
                return self.source.bindings[node.id]
            raise FrontendError(
                f"unknown static Python binding {node.id!r}", self._location(node)
            )
        if isinstance(node, ast.Attribute):
            owner = self._resolve_static(node.value)
            try:
                return getattr(owner, node.attr)
            except AttributeError as error:
                raise FrontendError(
                    f"Python binding has no attribute {node.attr!r}",
                    self._location(node),
                ) from error
        raise FrontendError("expected a static Python binding", self._location(node))

    def _emit(
        self,
        name: str,
        node: ast.AST,
        *,
        operands: tuple[Value, ...] = (),
        result_types: tuple[ValueType, ...] = (),
        attributes: dict[str, str] | None = None,
        regions: tuple[Region, ...] = (),
        result_names: tuple[str | None, ...] = (),
    ) -> tuple[Value, ...]:
        if self.block is None:
            raise AssertionError("compiler has no insertion block")
        return self.builder.emit(
            self.block,
            name,
            self._location(node),
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            regions=regions,
            result_names=result_names,
        )

    def _require_effect(self, effect: str, node: ast.AST) -> None:
        if self.helper_effects is not None and effect not in self.helper_effects:
            declared = ", ".join(self.helper_effects) or "none"
            raise FrontendError(
                f"helper performs {effect} but declares effects {declared}",
                self._location(node),
            )

    def _compile_statements(self, statements: Sequence[ast.stmt]) -> None:
        for index_in_body, statement in enumerate(statements):
            live_after = _names_needed_before_definition(
                statements[index_in_body + 1 :]
            )
            self._compile_statement(statement, live_after)

    def _compile_statement(self, statement: ast.stmt, live_after: set[str]) -> None:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1:
                raise FrontendError(
                    "chained assignment is not part of the Weft DSL",
                    self._location(statement),
                )
            value = self._expect_value(self._compile_expr(statement.value), statement.value)
            self._bind_target(statement.targets[0], value)
            return
        if isinstance(statement, ast.AnnAssign):
            if statement.value is None:
                raise FrontendError(
                    "declaration without a value is not supported",
                    self._location(statement),
                )
            value = self._expect_value(self._compile_expr(statement.value), statement.value)
            self._bind_target(statement.target, value)
            return
        if isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name) or statement.target.id not in self.env:
                raise FrontendError(
                    "augmented assignment requires an existing local",
                    self._location(statement),
                )
            synthetic = ast.BinOp(statement.target, statement.op, statement.value)
            ast.copy_location(synthetic, statement)
            value = self._expect_value(self._compile_expr(synthetic), statement)
            self._bind_name(statement.target.id, value, statement.target)
            return
        if isinstance(statement, ast.Expr):
            self._compile_expr(statement.value)
            return
        if isinstance(statement, ast.For):
            self._compile_for(statement)
            return
        if isinstance(statement, ast.With):
            self._compile_with(statement, live_after)
            return
        if isinstance(statement, ast.If):
            self._compile_if(statement, live_after)
            return
        if isinstance(statement, ast.While):
            self._compile_while(statement)
            return
        if isinstance(statement, ast.Pass):
            return
        if isinstance(statement, ast.Return):
            raise FrontendError(
                "kernel entries return implicitly; return is only valid in helpers",
                self._location(statement),
            )
        raise FrontendError(
            f"unsupported Weft statement {type(statement).__name__}",
            self._location(statement),
        )

    def _bind_name(self, name: str, value: Value, node: ast.AST) -> None:
        if self.vla_outer_names is not None and name in self.vla_outer_names:
            raise FrontendError(
                f"VLA body cannot mutate outer state {name!r}; use state algebra",
                self._location(node),
            )
        self.env[name] = value

    def _bind_target(self, target: ast.expr, value: Value) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, value, target)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            tuple_type = bare_type(value.type)
            if not isinstance(tuple_type, TupleType) or len(target.elts) != len(tuple_type.fields):
                raise FrontendError(
                    "tuple destructuring must match a canonical tuple value",
                    self._location(target),
                )
            for index_in_tuple, (element, field_type) in enumerate(
                zip(target.elts, tuple_type.fields)
            ):
                field = self._emit(
                    "weft_kernel.tuple_get",
                    target,
                    operands=(value,),
                    result_types=(field_type,),
                    attributes={"index": str(index_in_tuple)},
                )[0]
                self._bind_target(element, field)
            return
        raise FrontendError("assignment target must be a local name", self._location(target))

    def _compile_expr(self, expression: ast.expr) -> Value | None:
        if isinstance(expression, ast.Name):
            if expression.id in self.env:
                return self.env[expression.id]
            static = self._resolve_static(expression)
            if isinstance(static, InvalidValue):
                return self._invalid(expression)
            raise FrontendError(
                f"static binding {expression.id!r} is not a runtime value",
                self._location(expression),
            )
        if isinstance(expression, ast.Constant):
            if expression.value is None:
                return self._invalid(expression)
            return self._constant(expression.value, None, expression)
        if isinstance(expression, ast.Call):
            return self._compile_call(expression)
        if isinstance(expression, ast.BinOp):
            return self._compile_binary(expression)
        if isinstance(expression, ast.BoolOp):
            kind = "and" if isinstance(expression.op, ast.And) else "or"
            values = [self._expect_value(self._compile_expr(item), item) for item in expression.values]
            result = values[0]
            for value in values[1:]:
                result = self._emit_binary_values(kind, result, value, expression)
            return result
        if isinstance(expression, ast.UnaryOp):
            if isinstance(expression.op, ast.USub) and isinstance(expression.operand, ast.Constant):
                value = expression.operand.value
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return self._constant(-value, None, expression)
            operand = self._expect_value(self._compile_expr(expression.operand), expression.operand)
            if isinstance(expression.op, ast.USub):
                return self._emit(
                    "weft_kernel.unary",
                    expression,
                    operands=(operand,),
                    result_types=(operand.type,),
                    attributes={
                        "kind": _string("neg"),
                        "math": _string("strict"),
                    },
                )[0]
            if isinstance(expression.op, ast.Invert):
                one = self._constant(True, i1, expression)
                return self._emit_binary_values("xor", operand, one, expression)
            raise FrontendError("unsupported unary operator", self._location(expression))
        if isinstance(expression, ast.Compare):
            if len(expression.ops) != 1 or len(expression.comparators) != 1:
                raise FrontendError(
                    "chained comparisons are not part of the Weft DSL",
                    self._location(expression),
                )
            lhs, rhs = self._compile_binary_operands(
                expression.left, expression.comparators[0]
            )
            predicates = {
                ast.Eq: "eq",
                ast.NotEq: "ne",
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Gt: "gt",
                ast.GtE: "ge",
            }
            predicate = predicates.get(type(expression.ops[0]))
            if predicate is None:
                raise FrontendError("unsupported comparison", self._location(expression))
            kind, shape, axes = _join_shapes(
                lhs.type, rhs.type, self._location(expression)
            )
            result_type: ValueType = shaped_type(
                kind, shape, axes, ScalarType(i1)
            )
            if is_masked(lhs.type) or is_masked(rhs.type):
                result_type = MaskedType(result_type)
            return self._emit(
                "weft_kernel.compare",
                expression,
                operands=(lhs, rhs),
                result_types=(result_type,),
                attributes={"predicate": _string(predicate)},
            )[0]
        if isinstance(expression, ast.Subscript):
            return self._compile_subscript(expression)
        if isinstance(expression, (ast.Tuple, ast.List)):
            values = tuple(
                self._expect_value(self._compile_expr(element), element)
                for element in expression.elts
            )
            return self._emit_tuple(values, expression)
        raise FrontendError(
            f"unsupported Weft expression {type(expression).__name__}",
            self._location(expression),
        )

    def _expect_value(self, value: Value | None, node: ast.AST) -> Value:
        if value is None:
            raise FrontendError("expression does not produce a value", self._location(node))
        return value

    def _constant(
        self, value: object, dtype: DType | None, node: ast.AST
    ) -> Value:
        if dtype is None:
            if isinstance(value, bool):
                dtype = i1
            elif isinstance(value, int):
                dtype = index
            elif isinstance(value, float):
                dtype = f32
            else:
                raise FrontendError(
                    "only bool, integer, and floating literals are runtime values",
                    self._location(node),
                )
        value_type = ScalarType(dtype)
        if dtype.category is DTypeCategory.BOOL:
            if not isinstance(value, bool):
                raise FrontendError("i1 constant expects bool", self._location(node))
            spelling = _bool(value)
        elif dtype.category in {DTypeCategory.INTEGER, DTypeCategory.INDEX}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise FrontendError("integer constant expects int", self._location(node))
            spelling = f"{value} : {emit_type(value_type)}"
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FrontendError("floating constant expects int or float", self._location(node))
            literal = f"{float(value):.17g}"
            if not any(marker in literal for marker in (".", "e", "E")):
                literal += ".0"
            spelling = f"{literal} : {emit_type(value_type)}"
        result = self._emit(
            "weft_kernel.constant",
            node,
            result_types=(value_type,),
            attributes={"value": spelling},
        )[0]
        self.constant_values[result] = value
        return result

    def _invalid(self, node: ast.AST) -> Value:
        return self._emit(
            "weft_kernel.invalid", node, result_types=(NONE_TYPE,)
        )[0]

    def _is_true_value(self, value: Value) -> bool:
        return self.constant_values.get(value) is True

    def _same_index_extent(self, lhs: Value, rhs: Value) -> bool:
        if lhs == rhs:
            return True
        lhs_constant = self.constant_values.get(lhs)
        rhs_constant = self.constant_values.get(rhs)
        return (
            isinstance(lhs_constant, int)
            and not isinstance(lhs_constant, bool)
            and isinstance(rhs_constant, int)
            and not isinstance(rhs_constant, bool)
            and lhs_constant == rhs_constant
        )

    def _require_footprint(
        self, value: Value, footprint: Value, role: str, node: ast.AST
    ) -> None:
        value_kind = shape_kind(value.type)
        footprint_kind = shape_kind(footprint.type)
        if footprint_kind == "scalar":
            if value_kind != "scalar":
                raise FrontendError(
                    f"{role} cannot widen a scalar pointer footprint",
                    self._location(node),
                )
            return
        if value_kind == "scalar":
            return
        kind, shape, axes = _join_shapes(
            value.type, footprint.type, self._location(node)
        )
        if (
            kind != footprint_kind
            or shape != shape_of(footprint.type)
            or axes != axes_of(footprint.type)
        ):
            raise FrontendError(
                f"{role} does not broadcast to the pointer domain",
                self._location(node),
            )

    def _compile_binary_operands(self, lhs_node: ast.expr, rhs_node: ast.expr) -> tuple[Value, Value]:
        if isinstance(lhs_node, ast.Constant) and not isinstance(rhs_node, ast.Constant):
            rhs = self._expect_value(self._compile_expr(rhs_node), rhs_node)
            element = element_type(rhs.type)
            dtype = element.dtype if isinstance(element, ScalarType) else None
            lhs = self._constant(lhs_node.value, dtype, lhs_node)
            return lhs, rhs
        lhs = self._expect_value(self._compile_expr(lhs_node), lhs_node)
        if isinstance(rhs_node, ast.Constant):
            element = element_type(lhs.type)
            dtype = element.dtype if isinstance(element, ScalarType) else None
            rhs = self._constant(rhs_node.value, dtype, rhs_node)
        else:
            rhs = self._expect_value(self._compile_expr(rhs_node), rhs_node)
        return lhs, rhs

    def _compile_binary(self, expression: ast.BinOp) -> Value:
        lhs, rhs = self._compile_binary_operands(expression.left, expression.right)
        if isinstance(expression.op, ast.Add) and (_is_pointer(lhs.type) or _is_pointer(rhs.type)):
            if _is_pointer(rhs.type) and not _is_pointer(lhs.type):
                lhs, rhs = rhs, lhs
            if not _is_pointer(lhs.type) or not _is_index(rhs.type):
                raise FrontendError(
                    "pointer addition requires an index offset", self._location(expression)
                )
            kind, shape, axes = _join_shapes(
                lhs.type, rhs.type, self._location(expression)
            )
            result_type = shaped_type(
                kind, shape, axes, element_type(lhs.type)
            )
            return self._emit(
                "weft_kernel.ptr_add",
                expression,
                operands=(lhs, rhs),
                result_types=(result_type,),
            )[0]
        operations = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.FloorDiv: "div",
            ast.Mod: "mod",
            ast.BitAnd: "and",
            ast.BitOr: "or",
            ast.BitXor: "xor",
            ast.LShift: "shl",
            ast.RShift: "shr",
        }
        kind = operations.get(type(expression.op))
        if kind is None:
            raise FrontendError("unsupported binary operator", self._location(expression))
        return self._emit_binary_values(kind, lhs, rhs, expression)

    def _emit_binary_values(
        self, kind: str, lhs: Value, rhs: Value, node: ast.AST
    ) -> Value:
        if element_type(lhs.type) != element_type(rhs.type):
            raise FrontendError(
                "binary operands must have the same element type", self._location(node)
            )
        if kind in {"and", "or", "xor", "shl", "shr"}:
            operand_type = element_type(lhs.type)
            if (
                not isinstance(operand_type, ScalarType)
                or operand_type.dtype.category
                not in {
                    DTypeCategory.BOOL,
                    DTypeCategory.INTEGER,
                    DTypeCategory.INDEX,
                }
            ):
                raise FrontendError(
                    "bitwise operands must contain integer elements",
                    self._location(node),
                )
        result_kind, result_shape, result_axes = _join_shapes(
            lhs.type, rhs.type, self._location(node)
        )
        result_type: ValueType = shaped_type(
            result_kind, result_shape, result_axes, element_type(lhs.type)
        )
        if is_masked(lhs.type) or is_masked(rhs.type):
            result_type = MaskedType(result_type)
        return self._emit(
            "weft_kernel.binary",
            node,
            operands=(lhs, rhs),
            result_types=(result_type,),
            attributes={"kind": _string(kind)},
        )[0]

    def _compile_subscript(self, expression: ast.Subscript) -> Value:
        value = self._expect_value(self._compile_expr(expression.value), expression.value)
        bare = bare_type(value.type)
        if isinstance(bare, TupleType):
            if not isinstance(expression.slice, ast.Constant) or not isinstance(
                expression.slice.value, int
            ):
                raise FrontendError("tuple index must be a constant integer", self._location(expression))
            index_in_tuple = expression.slice.value
            if index_in_tuple < 0 or index_in_tuple >= len(bare.fields):
                raise FrontendError("tuple index is out of range", self._location(expression))
            return self._emit(
                "weft_kernel.tuple_get",
                expression,
                operands=(value,),
                result_types=(bare.fields[index_in_tuple],),
                attributes={"index": str(index_in_tuple)},
            )[0]
        if shape_kind(value.type) == "scalar":
            raise FrontendError("only logical blocks support slicing sugar", self._location(expression))
        items = expression.slice.elts if isinstance(expression.slice, ast.Tuple) else [expression.slice]
        current = value
        consumed = 0
        for item in items:
            if isinstance(item, ast.Constant) and item.value is None:
                current = self._expand_dims(current, consumed, expression)
                consumed += 1
            elif isinstance(item, ast.Slice) and item.lower is None and item.upper is None and item.step is None:
                consumed += 1
            else:
                raise FrontendError(
                    "block slicing supports only full slices and None",
                    self._location(expression),
                )
        if consumed - sum(isinstance(item, ast.Constant) and item.value is None for item in items) != len(shape_of(value.type) or ()):
            raise FrontendError("slice must mention every original block axis", self._location(expression))
        return current

    def _compile_call(self, call: ast.Call) -> Value | None:
        callee = self._resolve_static(call.func)
        if isinstance(callee, DType):
            if len(call.args) != 1 or call.keywords:
                raise FrontendError("typed constant expects one positional argument", self._location(call))
            static = self._eval_static(call.args[0])
            return self._constant(static, callee, call)
        if isinstance(callee, HelperDefinition):
            return self._inline_helper(callee, call)
        if not isinstance(callee, Intrinsic):
            raise FrontendError("call target is not a Weft intrinsic/helper", self._location(call))
        method = getattr(self, f"_intrinsic_{callee.name}", None)
        if method is None:
            raise FrontendError(
                f"W.{callee.name} has no canonical lowering", self._location(call)
            )
        return method(call)

    def _positional_and_keywords(
        self, call: ast.Call, positional: tuple[str, ...], defaults: dict[str, object]
    ) -> dict[str, ast.expr | object]:
        if len(call.args) > len(positional):
            raise FrontendError("too many positional arguments", self._location(call))
        result: dict[str, ast.expr | object] = dict(defaults)
        for name, value in zip(positional, call.args):
            result[name] = value
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg not in set(positional) | set(defaults):
                raise FrontendError("unknown or expanded keyword argument", self._location(keyword))
            if keyword.arg in positional[: len(call.args)]:
                raise FrontendError("argument supplied twice", self._location(keyword))
            result[keyword.arg] = keyword.value
        missing = [name for name in positional if name not in result]
        if missing:
            raise FrontendError(f"missing argument {missing[0]!r}", self._location(call))
        return result

    def _value_argument(self, value: ast.expr | object, node: ast.AST) -> Value:
        if isinstance(value, ast.expr):
            return self._expect_value(self._compile_expr(value), value)
        if isinstance(value, bool):
            return self._constant(value, i1, node)
        if isinstance(value, int):
            return self._constant(value, index, node)
        if isinstance(value, float):
            return self._constant(value, f32, node)
        if isinstance(value, InvalidValue) or value is None:
            return self._invalid(node)
        raise FrontendError("argument must be a Weft value", self._location(node))

    def _eval_static(self, node: ast.expr) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._eval_static(element) for element in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._eval_static(node.operand)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return -value
        if isinstance(node, ast.Name) and node.id in self.env:
            return self.env[node.id]
        return self._resolve_static(node)

    def _shape_spec(self, node: ast.expr) -> _ShapeSpec:
        elements = node.elts if isinstance(node, (ast.Tuple, ast.List)) else [node]
        dimensions: list[int] = []
        extents: list[Value] = []
        for element in elements:
            if isinstance(element, ast.Constant) and isinstance(element.value, int) and not isinstance(element.value, bool):
                if element.value <= 0:
                    raise FrontendError("shape extents must be positive", self._location(element))
                dimensions.append(element.value)
                extents.append(self._constant(element.value, index, element))
            else:
                extent = self._expect_value(self._compile_expr(element), element)
                if not _is_index(extent.type) or not _is_scalar(extent.type):
                    raise FrontendError("shape extent must be scalar index", self._location(element))
                dimensions.append(-1)
                extents.append(extent)
        return _ShapeSpec(tuple(dimensions), tuple(extents))

    def _block_shape_spec(self, node: ast.expr) -> _BlockShapeSpec:
        elements = node.elts if isinstance(node, (ast.Tuple, ast.List)) else [node]
        dimensions: list[int] = []
        axis_ids: list[int] = []
        coordinates: list[Value] = []
        for element in elements:
            coordinate = self._expect_value(self._compile_expr(element), element)
            axis = self.block_axes.get(coordinate)
            coordinate_type = bare_type(coordinate.type)
            if (
                axis is None
                or not isinstance(coordinate_type, BlockType)
                or len(coordinate_type.shape) != 1
                or coordinate_type.axes != (axis[0],)
                or coordinate_type.element_type != ScalarType(index)
            ):
                raise FrontendError(
                    "block shape entries must be direct W.block values",
                    self._location(element),
                )
            if axis[0] in axis_ids:
                raise FrontendError(
                    "one logical block axis cannot appear twice in a block shape",
                    self._location(element),
                )
            dimensions.append(coordinate_type.shape[0])
            axis_ids.append(axis[0])
            coordinates.append(coordinate)
        if not coordinates:
            raise FrontendError("block shape must contain at least one W.block", self._location(node))
        return _BlockShapeSpec(
            tuple(dimensions), tuple(axis_ids), tuple(coordinates)
        )

    def _intrinsic_select(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("predicate", "a", "b"), {})
        predicate = self._value_argument(args["predicate"], call)
        true_value = self._value_argument(args["a"], call)
        false_value = self._value_argument(args["b"], call)
        if not _is_predicate(predicate.type):
            raise FrontendError("W.select predicate must contain i1", self._location(call))
        if element_type(true_value.type) != element_type(false_value.type):
            raise FrontendError("W.select value types must match", self._location(call))
        kind, shape, axes = _join_shapes(
            true_value.type, false_value.type, self._location(call)
        )
        result_type: ValueType = shaped_type(
            kind, shape, axes, element_type(true_value.type)
        )
        if is_masked(true_value.type) or is_masked(false_value.type):
            result_type = MaskedType(result_type)
        if shape_kind(predicate.type) != "scalar":
            predicate_kind, predicate_shape, predicate_axes = _join_shapes(
                predicate.type, result_type, self._location(call)
            )
            if (
                predicate_kind != shape_kind(result_type)
                or predicate_shape != shape_of(result_type)
                or predicate_axes != axes_of(result_type)
            ):
                raise FrontendError(
                    "W.select predicate must broadcast to the result domain",
                    self._location(call),
                )
        return self._emit(
            "weft_kernel.select",
            call,
            operands=(predicate, true_value, false_value),
            result_types=(result_type,),
        )[0]

    def _intrinsic_load(self, call: ast.Call) -> Value:
        self._require_effect("read", call)
        args = self._positional_and_keywords(
            call,
            ("ptr",),
            {"where": True, "other": InvalidValue(), "alignment": None},
        )
        pointer = self._value_argument(args["ptr"], call)
        if not _is_pointer(pointer.type):
            raise FrontendError("W.load expects a typed pointer", self._location(call))
        where = self._value_argument(args["where"], call)
        if not _is_predicate(where.type):
            raise FrontendError("W.load where must be a predicate", self._location(call))
        other = self._value_argument(args["other"], call)
        pointer_type = element_type(pointer.type)
        assert isinstance(pointer_type, PointerType)
        result_type = with_element_type(pointer.type, pointer_type.element_type)
        if isinstance(other.type, NoneType) and not self._is_true_value(where):
            result_type = MaskedType(result_type)
        elif not isinstance(other.type, NoneType) and (
            is_masked(other.type) or element_type(other.type) != pointer_type.element_type
        ):
            raise FrontendError("W.load other must match pointer element type", self._location(call))
        self._require_footprint(where, pointer, "W.load where", call)
        if not isinstance(other.type, NoneType):
            self._require_footprint(other, pointer, "W.load other", call)
        alignment = args["alignment"]
        if isinstance(alignment, ast.expr):
            alignment = self._eval_static(alignment)
        if alignment is None:
            alignment = 0
        if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment < 0:
            raise FrontendError("W.load alignment must be a non-negative integer", self._location(call))
        return self._emit(
            "weft_kernel.load",
            call,
            operands=(pointer, where, other),
            result_types=(result_type,),
            attributes={"alignment": str(alignment)},
        )[0]

    def _intrinsic_load_f16_le(self, call: ast.Call) -> Value:
        self._require_effect("read", call)
        args = self._positional_and_keywords(call, ("base",), {})
        base = self._value_argument(args["base"], call)
        if (
            not isinstance(base.type, PointerType)
            or base.type.element_type != ScalarType(u8)
            or base.type.access == "write"
        ):
            raise FrontendError(
                "W.load_f16_le base must be a readable scalar u8 pointer",
                self._location(call),
            )
        return self._emit(
            "weft_ext.load_f16_le",
            call,
            operands=(base,),
            result_types=(ScalarType(f32),),
        )[0]

    def _intrinsic_store(self, call: ast.Call) -> None:
        self._require_effect("write", call)
        args = self._positional_and_keywords(
            call, ("ptr", "value"), {"where": True, "alignment": None}
        )
        pointer = self._value_argument(args["ptr"], call)
        value = self._value_argument(args["value"], call)
        where = self._value_argument(args["where"], call)
        if not _is_pointer(pointer.type) or not _is_predicate(where.type):
            raise FrontendError("W.store expects pointer and predicate", self._location(call))
        pointer_type = element_type(pointer.type)
        assert isinstance(pointer_type, PointerType)
        if element_type(value.type) != pointer_type.element_type:
            raise FrontendError("W.store value must match pointer element type", self._location(call))
        if is_masked(value.type):
            raise FrontendError(
                "W.store requires an explicit filled value, not validity-carrying data",
                self._location(call),
            )
        self._require_footprint(where, pointer, "W.store where", call)
        self._require_footprint(value, pointer, "W.store value", call)
        alignment = args["alignment"]
        if isinstance(alignment, ast.expr):
            alignment = self._eval_static(alignment)
        alignment = 0 if alignment is None else alignment
        if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment < 0:
            raise FrontendError("W.store alignment must be non-negative", self._location(call))
        self._emit(
            "weft_kernel.store",
            call,
            operands=(pointer, value, where),
            attributes={"alignment": str(alignment)},
        )
        return None

    def _intrinsic_sort_indices(self, call: ast.Call) -> None:
        self._require_effect("read", call)
        self._require_effect("write", call)
        args = self._positional_and_keywords(
            call,
            ("input", "output", "scratch", "extent"),
            {
                "order": "ascending",
                "nan": "last",
                "tie": "index_ascending",
            },
        )
        input_pointer = self._value_argument(args["input"], call)
        output_pointer = self._value_argument(args["output"], call)
        scratch_pointer = self._value_argument(args["scratch"], call)
        extent = self._value_argument(args["extent"], call)
        if not isinstance(input_pointer.type, PointerType) or not isinstance(
            output_pointer.type, PointerType
        ) or not isinstance(scratch_pointer.type, PointerType):
            raise FrontendError(
                "W.sort_indices expects scalar typed pointers", self._location(call)
            )
        if input_pointer.type.element_type != ScalarType(f32):
            raise FrontendError(
                "W.sort_indices input must point to f32", self._location(call)
            )
        if output_pointer.type.element_type != ScalarType(u32):
            raise FrontendError(
                "W.sort_indices output must point to u32", self._location(call)
            )
        if (
            scratch_pointer.type.element_type != ScalarType(u32)
            or scratch_pointer.type.storage_class != "workspace"
            or not scratch_pointer.type.noalias
        ):
            raise FrontendError(
                "W.sort_indices scratch must be a noalias u32 workspace pointer",
                self._location(call),
            )
        if not _is_index(extent.type) or not _is_scalar(extent.type):
            raise FrontendError(
                "W.sort_indices extent must be a scalar index", self._location(call)
            )
        scratch_shape = self.storage_shapes.get(scratch_pointer)
        if (
            scratch_shape is None
            or len(scratch_shape.extents) != 1
            or not self._same_index_extent(scratch_shape.extents[0], extent)
        ):
            raise FrontendError(
                "W.sort_indices scratch storage must be rank-one with the same extent",
                self._location(call),
            )
        order = args["order"]
        nan = args["nan"]
        tie = args["tie"]
        if isinstance(order, ast.expr):
            order = self._eval_static(order)
        if isinstance(nan, ast.expr):
            nan = self._eval_static(nan)
        if isinstance(tie, ast.expr):
            tie = self._eval_static(tie)
        if order not in {"ascending", "descending"}:
            raise FrontendError(
                "W.sort_indices order must be ascending or descending",
                self._location(call),
            )
        if nan != "last":
            raise FrontendError(
                "W.sort_indices nan must be last", self._location(call)
            )
        if tie != "index_ascending":
            raise FrontendError(
                "W.sort_indices tie must be index_ascending", self._location(call)
            )
        self._emit(
            "weft_kernel.sort_indices",
            call,
            operands=(input_pointer, output_pointer, scratch_pointer, extent),
            attributes={
                "order": _string(order),
                "nan": _string(nan),
                "tie": _string(tie),
            },
        )
        return None

    def _intrinsic_storage(self, call: ast.Call) -> None:
        if self.block is not self.entry_block:
            raise FrontendError(
                "W.storage must be declared directly in the kernel entry body",
                self._location(call),
            )
        args = self._positional_and_keywords(call, ("ptr", "shape"), {})
        pointer = self._value_argument(args["ptr"], call)
        if not isinstance(pointer.type, PointerType):
            raise FrontendError(
                "W.storage requires one entry pointer", self._location(call)
            )
        if self.entry_block is None or pointer not in self.entry_block.arguments:
            raise FrontendError(
                "W.storage must bind a kernel entry pointer directly",
                self._location(call),
            )
        if pointer.type.storage_class not in {"persistent", "workspace"}:
            raise FrontendError(
                "W.storage is required only for persistent or workspace pointers",
                self._location(call),
            )
        shape_node = args["shape"]
        if not isinstance(shape_node, ast.expr):
            raise FrontendError("storage shape must be explicit DSL syntax", self._location(call))
        shape = self._shape_spec(shape_node)
        if pointer in self.storage_shapes:
            raise FrontendError(
                "one entry pointer cannot have multiple W.storage declarations",
                self._location(call),
            )
        self._emit(
            "weft_kernel.storage",
            call,
            operands=(pointer,) + shape.extents,
            attributes={"shape": _dense_i64(shape.dimensions)},
        )
        self.storage_shapes[pointer] = shape
        return None

    def _intrinsic_block(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("extent",), {"offset": 0})
        extent = self._value_argument(args["extent"], call)
        offset = self._value_argument(args["offset"], call)
        if (
            not _is_index(extent.type)
            or not _is_scalar(extent.type)
            or not _is_index(offset.type)
            or not _is_scalar(offset.type)
        ):
            raise FrontendError(
                "W.block extent/offset must be scalar index", self._location(call)
            )
        dimension = -1
        extent_node = args["extent"]
        if (
            isinstance(extent_node, ast.Constant)
            and isinstance(extent_node.value, int)
            and not isinstance(extent_node.value, bool)
        ):
            if extent_node.value <= 0:
                raise FrontendError(
                    "W.block extent must be positive", self._location(extent_node)
                )
            dimension = extent_node.value
        axis_id = self.next_axis_id
        self.next_axis_id += 1
        result_type = BlockType((dimension,), (axis_id,), ScalarType(index))
        result = self._emit(
            "weft_kernel.block_index",
            call,
            operands=(extent, offset),
            result_types=(result_type,),
            attributes={"axis": str(axis_id)},
        )[0]
        self.block_axes[result] = (axis_id, extent)
        return result

    def _intrinsic_full(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call, ("shape", "value"), {"dtype": None}
        )
        if not isinstance(args["shape"], ast.expr):
            raise FrontendError("shape must be explicit DSL syntax", self._location(call))
        shape = self._block_shape_spec(args["shape"])
        value = self._value_argument(args["value"], call)
        dtype = args["dtype"]
        if isinstance(dtype, ast.expr):
            dtype = self._eval_static(dtype)
        if dtype is not None:
            if not isinstance(dtype, DType):
                raise FrontendError("dtype must be a Weft scalar type", self._location(call))
            if value.type != ScalarType(dtype):
                value = self._cast_value(value, dtype, call)
        if not isinstance(value.type, ScalarType):
            raise FrontendError("W.full fill value must be scalar", self._location(call))
        if value.type != ScalarType(f32) or len(shape.dimensions) not in {1, 2}:
            raise FrontendError(
                "W.full currently creates rank-one or rank-two f32 blocks",
                self._location(call),
            )
        result_type = BlockType(shape.dimensions, shape.axis_ids, value.type)
        return self._emit(
            "weft_kernel.full",
            call,
            operands=(value,) + shape.coordinates,
            result_types=(result_type,),
        )[0]

    def _intrinsic_zeros(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("shape", "dtype"), {})
        dtype = args["dtype"]
        if not isinstance(dtype, ast.expr):
            raise FrontendError("dtype must be explicit DSL syntax", self._location(call))
        resolved = self._eval_static(dtype)
        if not isinstance(resolved, DType):
            raise FrontendError("W.zeros dtype must be a Weft scalar type", self._location(call))
        zero_node = ast.Constant(0.0 if resolved.category in {DTypeCategory.FLOAT, DTypeCategory.BFLOAT} else 0)
        ast.copy_location(zero_node, call)
        synthetic = ast.Call(
            func=call.func,
            args=[args["shape"], zero_node],
            keywords=[ast.keyword(arg="dtype", value=dtype)],
        )
        ast.copy_location(synthetic, call)
        return self._intrinsic_full(synthetic)

    def _expand_dims(self, value: Value, axis: int, node: ast.AST) -> Value:
        shape = shape_of(value.type)
        axes = axes_of(value.type)
        if shape is None or axes is None or axis < 0 or axis > len(shape):
            raise FrontendError("expand_dims axis is outside logical rank", self._location(node))
        if shape_kind(value.type) == "region" and axis == 0:
            raise FrontendError(
                "the active VLA axis must remain the first region axis",
                self._location(node),
            )
        result_shape = shape[:axis] + (1,) + shape[axis:]
        result_axes = axes[:axis] + (0,) + axes[axis:]
        result_type = shaped_type(
            shape_kind(value.type), result_shape, result_axes, element_type(value.type)
        )
        if is_masked(value.type):
            result_type = MaskedType(result_type)
        return self._emit(
            "weft_kernel.expand_dims",
            node,
            operands=(value,),
            result_types=(result_type,),
            attributes={"axis": str(axis)},
        )[0]

    def _intrinsic_cast(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("value", "dtype"), {})
        value = self._value_argument(args["value"], call)
        dtype = args["dtype"]
        if isinstance(dtype, ast.expr):
            dtype = self._eval_static(dtype)
        if not isinstance(dtype, DType):
            raise FrontendError("W.cast dtype must be a Weft type", self._location(call))
        return self._cast_value(value, dtype, call)

    def _intrinsic_bitcast(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("value", "dtype"), {})
        value = self._value_argument(args["value"], call)
        dtype = args["dtype"]
        if isinstance(dtype, ast.expr):
            dtype = self._eval_static(dtype)
        if not isinstance(dtype, DType) or dtype.category is DTypeCategory.INDEX:
            raise FrontendError(
                "W.bitcast dtype must be a fixed-width Weft scalar type",
                self._location(call),
            )
        source = element_type(value.type)
        if (
            not isinstance(source, ScalarType)
            or source.dtype.category is DTypeCategory.INDEX
            or source.dtype.bits != dtype.bits
        ):
            raise FrontendError(
                "W.bitcast requires equal-width scalar element types",
                self._location(call),
            )
        kind = shape_kind(value.type)
        source_dtype = source.dtype
        scalar_pairs = {
            (u8, i8),
            (i8, u8),
            (u16, i16),
            (i16, u16),
            (u16, f16),
            (f16, u16),
            (f32, u32),
            (u32, f32),
        }
        byte_pairs = {(u8, i8), (i8, u8)}
        block_pairs = byte_pairs | {(u16, i16), (i16, u16), (u16, f16)}
        allowed = (
            scalar_pairs
            if kind == "scalar"
            else block_pairs
            if kind == "block"
            else byte_pairs
            if kind == "region"
            else set()
        )
        if (source_dtype, dtype) not in allowed:
            raise FrontendError(
                f"W.bitcast has no {kind} realization for {source_dtype} to {dtype}",
                self._location(call),
            )
        result_type = with_element_type(value.type, ScalarType(dtype))
        return self._emit(
            "weft_kernel.bitcast",
            call,
            operands=(value,),
            result_types=(result_type,),
        )[0]

    def _cast_value(self, value: Value, dtype: DType, node: ast.AST) -> Value:
        source = element_type(value.type)
        if not isinstance(source, ScalarType):
            raise FrontendError("W.cast requires scalar elements", self._location(node))
        kind = shape_kind(value.type)
        if kind == "region":
            allowed = {
                (f16, f32),
                (f32, f16),
                (u32, index),
                (index, f32),
            }
        elif kind == "block":
            allowed = {
                (index, u8),
                (u8, u16),
                (u8, i32),
                (u8, f32),
                (i8, i32),
                (i8, f32),
                (u16, i32),
                (u16, f32),
                (i16, i32),
                (f16, f32),
            }
        else:
            allowed = None
        if allowed is not None and (source.dtype, dtype) not in allowed:
            raise FrontendError(
                f"W.cast has no {kind} realization for {source.dtype} to {dtype}",
                self._location(node),
            )
        result_type = with_element_type(value.type, ScalarType(dtype))
        return self._emit(
            "weft_kernel.cast", node, operands=(value,), result_types=(result_type,)
        )[0]

    def _math_unary(self, call: ast.Call, kind: str) -> Value:
        args = self._positional_and_keywords(
            call, ("value",), {"math": "native"}
        )
        value = self._value_argument(args["value"], call)
        math = args["math"]
        if isinstance(math, ast.expr):
            math = self._eval_static(math)
        if math not in {"strict", "native", "fast"}:
            raise FrontendError(
                "unary math must be strict, native, or fast",
                self._location(call),
            )
        return self._emit(
            "weft_kernel.unary",
            call,
            operands=(value,),
            result_types=(value.type,),
            attributes={
                "kind": _string(kind),
                "math": _string(str(math)),
            },
        )[0]

    def _intrinsic_exp(self, call: ast.Call) -> Value:
        return self._math_unary(call, "exp")

    def _intrinsic_tanh(self, call: ast.Call) -> Value:
        return self._math_unary(call, "tanh")

    def _intrinsic_log(self, call: ast.Call) -> Value:
        return self._math_unary(call, "log")

    def _intrinsic_sin(self, call: ast.Call) -> Value:
        return self._math_unary(call, "sin")

    def _intrinsic_cos(self, call: ast.Call) -> Value:
        return self._math_unary(call, "cos")

    def _intrinsic_floor(self, call: ast.Call) -> Value:
        return self._math_unary(call, "floor")

    def _intrinsic_sqrt(self, call: ast.Call) -> Value:
        return self._math_unary(call, "sqrt")

    def _intrinsic_rsqrt(self, call: ast.Call) -> Value:
        return self._math_unary(call, "rsqrt")

    def _pointwise_pair(self, call: ast.Call, kind: str) -> Value:
        args = self._positional_and_keywords(call, ("a", "b"), {})
        lhs = self._value_argument(args["a"], call)
        rhs = self._value_argument(args["b"], call)
        return self._emit_binary_values(kind, lhs, rhs, call)

    def _intrinsic_maximum(self, call: ast.Call) -> Value:
        return self._pointwise_pair(call, "max")

    def _intrinsic_minimum(self, call: ast.Call) -> Value:
        return self._pointwise_pair(call, "min")

    def _intrinsic_neg_inf(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("dtype",), {})
        dtype = args["dtype"]
        if isinstance(dtype, ast.expr):
            dtype = self._eval_static(dtype)
        if not isinstance(dtype, DType) or dtype.category not in {
            DTypeCategory.FLOAT,
            DTypeCategory.BFLOAT,
        }:
            raise FrontendError("W.neg_inf expects a floating dtype", self._location(call))
        return self._emit(
            "weft_kernel.special_value",
            call,
            result_types=(ScalarType(dtype),),
            attributes={"kind": _string("neg_inf")},
        )[0]

    def _emit_tuple(self, values: tuple[Value, ...], node: ast.AST) -> Value:
        if not values:
            raise FrontendError("W.tuple requires at least one field", self._location(node))
        if any(not _is_scalar(value.type) or is_masked(value.type) for value in values):
            raise FrontendError(
                "W.tuple is a closed scalar state tuple; block state uses ordinary loop carry",
                self._location(node),
            )
        result_type = TupleType(tuple(value.type for value in values))
        return self._emit(
            "weft_kernel.tuple",
            node,
            operands=values,
            result_types=(result_type,),
        )[0]

    def _intrinsic_tuple(self, call: ast.Call) -> Value:
        if call.keywords:
            raise FrontendError("W.tuple accepts positional fields only", self._location(call))
        values = tuple(
            self._expect_value(self._compile_expr(argument), argument)
            for argument in call.args
        )
        return self._emit_tuple(values, call)

    def _intrinsic_reduce(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call,
            ("value",),
            {
                "op": "add",
                "identity": None,
                "where": True,
                "axis": None,
                "order": "relaxed",
                "acc_dtype": None,
            },
        )
        value = self._value_argument(args["value"], call)
        if args["identity"] is None:
            raise FrontendError("W.reduce requires an explicit identity", self._location(call))
        identity = self._value_argument(args["identity"], call)
        where = self._value_argument(args["where"], call)
        op = args["op"]
        axis = args["axis"]
        order = args["order"]
        acc_dtype = args["acc_dtype"]
        for name, item in (("op", op), ("axis", axis), ("order", order), ("acc_dtype", acc_dtype)):
            if isinstance(item, ast.expr):
                item = self._eval_static(item)
            if name == "op":
                op = item
            elif name == "axis":
                axis = item
            elif name == "order":
                order = item
            else:
                acc_dtype = item
        if acc_dtype is None:
            acc_type = element_type(identity.type)
        elif isinstance(acc_dtype, DType):
            acc_type = ScalarType(acc_dtype)
        else:
            raise FrontendError("acc_dtype must be a Weft dtype", self._location(call))
        if identity.type != acc_type:
            raise FrontendError("identity type must equal acc_dtype", self._location(call))
        bare = bare_type(value.type)
        if isinstance(bare, RegionType) and axis is None:
            remaining_shape = bare.shape[1:]
            remaining_axes = bare.axes[1:]
            result_type: ValueType = (
                BlockType(remaining_shape, remaining_axes, acc_type)
                if remaining_shape
                else acc_type
            )
            axis_value = -1
        elif isinstance(bare, BlockType) and isinstance(axis, int) and not isinstance(axis, bool):
            if axis < 0 or axis >= len(bare.shape):
                raise FrontendError("reduce axis is outside logical rank", self._location(call))
            result_shape = bare.shape[:axis] + bare.shape[axis + 1 :]
            result_axes = bare.axes[:axis] + bare.axes[axis + 1 :]
            result_type = (
                BlockType(result_shape, result_axes, acc_type)
                if result_shape
                else acc_type
            )
            axis_value = axis
        else:
            raise FrontendError(
                "axis=None reduces the active VLA; block reduce needs an axis",
                self._location(call),
            )
        return self._emit(
            "weft_kernel.reduce",
            call,
            operands=(value, identity, where),
            result_types=(result_type,),
            attributes={
                "kind": _string(str(op)),
                "axis": str(axis_value),
                "order": _string(str(order)),
                "acc_dtype": emit_type(acc_type),
            },
        )[0]

    def _intrinsic_scan(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call,
            ("value",),
            {
                "op": "add",
                "identity": None,
                "inclusive": True,
                "where": True,
                "segment_start": None,
                "order": "ordered",
                "acc_dtype": None,
            },
        )
        value = self._value_argument(args["value"], call)
        if is_masked(value.type):
            raise FrontendError(
                "W.scan does not define a logical-validity realization; fill the value before scanning",
                self._location(call),
            )
        if args["identity"] is None:
            raise FrontendError("W.scan requires an explicit identity", self._location(call))
        identity = self._value_argument(args["identity"], call)
        where = self._value_argument(args["where"], call)
        segment = self._value_argument(args["segment_start"], call)
        static: dict[str, object] = {}
        for name in ("op", "inclusive", "order", "acc_dtype"):
            item = args[name]
            static[name] = self._eval_static(item) if isinstance(item, ast.expr) else item
        acc_dtype = static["acc_dtype"]
        acc_type = (
            ScalarType(acc_dtype)
            if isinstance(acc_dtype, DType)
            else element_type(identity.type)
        )
        result_type = with_element_type(bare_type(value.type), acc_type)
        return self._emit(
            "weft_kernel.scan",
            call,
            operands=(value, identity, where, segment),
            result_types=(result_type,),
            attributes={
                "kind": _string(str(static["op"])),
                "inclusive": _bool(bool(static["inclusive"])),
                "order": _string(str(static["order"])),
                "acc_dtype": emit_type(acc_type),
            },
        )[0]

    def _intrinsic_argmax(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call,
            ("value", "coordinate"),
            {"tie": "lowest_coordinate", "order": "relaxed"},
        )
        value = self._value_argument(args["value"], call)
        coordinate = self._value_argument(args["coordinate"], call)
        if element_type(value.type) != ScalarType(f32):
            raise FrontendError("W.argmax value must contain f32", self._location(call))
        if element_type(coordinate.type) != ScalarType(index):
            raise FrontendError(
                "W.argmax coordinate must contain index", self._location(call)
            )
        if shape_kind(coordinate.type) != shape_kind(value.type) or shape_of(
            coordinate.type
        ) != shape_of(value.type) or axes_of(coordinate.type) != axes_of(value.type):
            raise FrontendError(
                "W.argmax coordinate must share the value logical domain",
                self._location(call),
            )
        tie = args["tie"]
        order = args["order"]
        if isinstance(tie, ast.expr):
            tie = self._eval_static(tie)
        if isinstance(order, ast.expr):
            order = self._eval_static(order)
        if tie != "lowest_coordinate" or order != "relaxed":
            raise FrontendError(
                "W.argmax requires tie='lowest_coordinate' and order='relaxed'",
                self._location(call),
            )
        result_type = TupleType((ScalarType(f32), ScalarType(index)))
        return self._emit(
            "weft_kernel.argmax",
            call,
            operands=(value, coordinate),
            result_types=(result_type,),
            attributes={"tie": _string(tie), "order": _string(order)},
        )[0]

    def _intrinsic_online_softmax_summary(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call,
            ("value",),
            {"math": "native", "order": "preserve"},
        )
        value = self._value_argument(args["value"], call)
        if element_type(value.type) != ScalarType(f32) or shape_kind(value.type) == "scalar":
            raise FrontendError(
                "W.online_softmax_summary value must be a logical f32 domain",
                self._location(call),
            )
        math = args["math"]
        order = args["order"]
        if isinstance(math, ast.expr):
            math = self._eval_static(math)
        if isinstance(order, ast.expr):
            order = self._eval_static(order)
        if math != "native" or order != "preserve":
            raise FrontendError(
                "W.online_softmax_summary requires math='native' and order='preserve'",
                self._location(call),
            )
        result_type = TupleType((ScalarType(f32), ScalarType(f32)))
        return self._emit(
            "weft_kernel.online_softmax_summary",
            call,
            operands=(value,),
            result_types=(result_type,),
            attributes={"math": _string(math), "order": _string(order)},
        )[0]

    def _inline_helper(self, definition: HelperDefinition, call: ast.Call) -> Value:
        source = HelperSource.from_definition(definition)
        positional = [
            self._expect_value(self._compile_expr(argument), argument)
            for argument in call.args
        ]
        parameters = source.function.args.args
        if len(positional) + len(call.keywords) != len(parameters):
            raise FrontendError(
                f"helper {definition.__name__} argument count mismatch",
                self._location(call),
            )
        values: dict[str, Value] = {
            parameter.arg: value for parameter, value in zip(parameters, positional)
        }
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in values:
                raise FrontendError("invalid helper keyword", self._location(keyword))
            values[keyword.arg] = self._expect_value(
                self._compile_expr(keyword.value), keyword.value
            )
        ordered: list[Value] = []
        for parameter in parameters:
            if parameter.arg not in values:
                raise FrontendError(
                    f"missing helper argument {parameter.arg!r}", self._location(call)
                )
            ordered.append(values[parameter.arg])
        assert self.block is not None
        return self._compile_helper_body(
            definition, source, self.block, tuple(ordered), call
        )

    def _compile_helper_body(
        self,
        definition: HelperDefinition,
        source: HelperSource,
        block: Region,
        arguments: tuple[Value, ...],
        call: ast.Call,
    ) -> Value:
        parameters = source.function.args.args
        previous = (
            self.source,
            self.block,
            self.env,
            self.helper_effects,
            self.vla_outer_names,
        )
        self.source = source
        self.block = block
        self.env = {
            parameter.arg: value for parameter, value in zip(parameters, arguments)
        }
        self.helper_effects = definition.effects
        self.vla_outer_names = None
        try:
            if not source.function.body or not isinstance(source.function.body[-1], ast.Return):
                raise FrontendError(
                    "Weft helper must end in one value return",
                    source.location(source.function),
                )
            self._compile_statements(source.function.body[:-1])
            return_node = source.function.body[-1]
            assert isinstance(return_node, ast.Return)
            if return_node.value is None:
                raise FrontendError("helper return requires a value", source.location(return_node))
            return self._expect_value(self._compile_expr(return_node.value), return_node.value)
        finally:
            (
                self.source,
                self.block,
                self.env,
                self.helper_effects,
                self.vla_outer_names,
            ) = previous

    def _product_arguments(
        self, call: ast.Call
    ) -> tuple[dict[str, ast.expr | object], Value, Value]:
        args = self._positional_and_keywords(
            call,
            ("lhs", "rhs"),
            {
                "init": None,
                "acc_dtype": None,
                "order": "relaxed",
                "math": "native",
            },
        )
        lhs = self._value_argument(args["lhs"], call)
        rhs = self._value_argument(args["rhs"], call)
        return args, lhs, rhs

    def _intrinsic_dot(self, call: ast.Call) -> Value:
        args, lhs, rhs = self._product_arguments(call)
        return self._emit_product(call, "dot", args, lhs, rhs)

    def _intrinsic_matmul(self, call: ast.Call) -> Value:
        args, lhs, rhs = self._product_arguments(call)
        return self._emit_product(call, "matmul", args, lhs, rhs)

    def _emit_product(
        self,
        call: ast.Call,
        kind: str,
        args: dict[str, ast.expr | object],
        lhs: Value,
        rhs: Value,
    ) -> Value:
        lhs_bare = bare_type(lhs.type)
        rhs_bare = bare_type(rhs.type)
        if not isinstance(lhs_bare, (BlockType, RegionType)) or not isinstance(
            rhs_bare, (BlockType, RegionType)
        ):
            raise FrontendError(
                f"{kind} operands must be logical block or VLA region values",
                self._location(call),
            )
        static: dict[str, object] = {}
        for name in ("acc_dtype", "order", "math"):
            item = args[name]
            static[name] = self._eval_static(item) if isinstance(item, ast.expr) else item
        if static["order"] not in {"ordered", "preserve", "relaxed"}:
            raise FrontendError(
                f"W.{kind} order must be ordered, preserve, or relaxed",
                self._location(call),
            )
        if static["math"] not in {"strict", "native", "fast"}:
            raise FrontendError(
                f"W.{kind} math must be strict, native, or fast",
                self._location(call),
            )
        lhs_region = isinstance(lhs_bare, RegionType)
        rhs_region = isinstance(rhs_bare, RegionType)
        if kind == "dot":
            if element_type(lhs.type) != ScalarType(f32) or element_type(
                rhs.type
            ) != ScalarType(f32):
                raise FrontendError(
                    "W.dot requires f32 multiplicands", self._location(call)
                )
            rhs_is_vector = isinstance(rhs_bare, BlockType) and len(rhs_bare.shape) == 1
            rhs_is_vla_rows = rhs_region and len(rhs_bare.shape) == 2
            if (
                len(lhs_bare.shape) != 2
                or (not rhs_is_vector and not rhs_is_vla_rows)
                or (lhs_region and not rhs_is_vector)
                or (rhs_region and not isinstance(lhs_bare, BlockType))
            ):
                raise FrontendError(
                    "W.dot supports [R,K] x [K], [VLA,K] x [K], or [R,K] x [VLA,K]",
                    self._location(call),
                )
            lhs_reduction_axis = lhs_bare.axes[-1]
            rhs_reduction_axis = rhs_bare.axes[-1]
            if lhs_reduction_axis != rhs_reduction_axis:
                raise FrontendError(
                    "W.dot operands must share one explicit reduction block",
                    self._location(call),
                )
            output_shape = ([-1] if lhs_region or rhs_region else []) + (
                [] if lhs_region else [lhs_bare.shape[0]]
            )
            output_axes = ([-1] if lhs_region or rhs_region else []) + (
                [] if lhs_region else [lhs_bare.axes[0]]
            )
        elif kind == "matmul":
            if element_type(lhs.type) != ScalarType(f16) or element_type(
                rhs.type
            ) != ScalarType(f16):
                raise FrontendError(
                    "W.matmul requires f16 multiplicands", self._location(call)
                )
            if (
                not isinstance(lhs_bare, BlockType)
                or not isinstance(rhs_bare, BlockType)
                or len(lhs_bare.shape) != 2
                or len(rhs_bare.shape) != 2
            ):
                raise FrontendError(
                    "W.matmul requires local [M,K] x [K,N] block operands",
                    self._location(call),
                )
            if lhs_bare.axes[1] != rhs_bare.axes[0]:
                raise FrontendError(
                    "W.matmul operands must share one explicit K block",
                    self._location(call),
                )
            output_shape = [lhs_bare.shape[0], rhs_bare.shape[1]]
            output_axes = [lhs_bare.axes[0], rhs_bare.axes[1]]
        else:
            raise AssertionError(f"unknown structured product {kind}")
        has_region = lhs_region or rhs_region
        acc_dtype = static["acc_dtype"]
        if isinstance(acc_dtype, DType):
            acc_type = ScalarType(acc_dtype)
        else:
            raise FrontendError(
                f"W.{kind} requires an explicit acc_dtype", self._location(call)
            )
        if acc_type != ScalarType(f32):
            raise FrontendError(
                f"W.{kind} requires f32 accumulation", self._location(call)
            )
        result_type: ValueType = (
            RegionType(tuple(output_shape), tuple(output_axes), acc_type)
            if has_region
            else BlockType(tuple(output_shape), tuple(output_axes), acc_type)
            if output_shape
            else acc_type
        )
        init_arg = args["init"]
        if init_arg is None:
            raise FrontendError(
                f"W.{kind} requires an explicit accumulator init",
                self._location(call),
            )
        if isinstance(init_arg, Value):
            init = init_arg
        else:
            init = self._value_argument(init_arg, call)
        if shape_kind(init.type) != "scalar" and bare_type(init.type) != result_type:
            raise FrontendError(
                f"{kind} init must be scalar or exactly output-shaped",
                self._location(call),
            )
        if is_masked(init.type):
            raise FrontendError(
                f"W.{kind} accumulator init cannot carry validity",
                self._location(call),
            )
        return self._emit(
            f"weft_kernel.{kind}",
            call,
            operands=(lhs, rhs, init),
            result_types=(result_type,),
            attributes={
                "acc_dtype": emit_type(acc_type),
                "order": _string(str(static["order"])),
                "math": _string(str(static["math"])),
            },
        )[0]

    def _intrinsic_lookup(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call, ("table", "indices"), {"where": True}
        )
        table = self._value_argument(args["table"], call)
        indices = self._value_argument(args["indices"], call)
        where = self._value_argument(args["where"], call)
        table_type = bare_type(table.type)
        if (
            not isinstance(table_type, BlockType)
            or table_type.shape != (16,)
            or element_type(table.type) != ScalarType(f32)
            or not isinstance(bare_type(indices.type), RegionType)
            or element_type(indices.type) != ScalarType(u8)
            or is_masked(table.type)
            or is_masked(indices.type)
            or not _is_predicate(where.type)
        ):
            raise FrontendError(
                "W.lookup requires block<16xf32>, an unmasked VLA u8 index, and a predicate",
                self._location(call),
            )
        if not self._is_true_value(where):
            raise FrontendError(
                "W.lookup currently requires where=True", self._location(call)
            )
        result_type = with_element_type(indices.type, element_type(table.type))
        return self._emit(
            "weft_kernel.lookup",
            call,
            operands=(table, indices, where),
            result_types=(result_type,),
        )[0]

    def _intrinsic_decode(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call, ("codes", "table"), {"where": True, "out_dtype": None}
        )
        codes = self._value_argument(args["codes"], call)
        table = self._value_argument(args["table"], call)
        where = self._value_argument(args["where"], call)
        codes_type = bare_type(codes.type)
        table_type = bare_type(table.type)
        if (
            not isinstance(codes_type, BlockType)
            or codes_type.shape != (16,)
            or element_type(codes.type) != ScalarType(u8)
            or not isinstance(table_type, BlockType)
            or table_type.shape != (16,)
            or element_type(table.type) != ScalarType(i8)
            or is_masked(codes.type)
            or is_masked(table.type)
            or not _is_predicate(where.type)
        ):
            raise FrontendError(
                "W.decode requires block<16xu8> codes, block<16xi8> table, and a predicate",
                self._location(call),
            )
        if not self._is_true_value(where):
            raise FrontendError(
                "W.decode currently requires where=True", self._location(call)
            )
        dtype = args["out_dtype"]
        if isinstance(dtype, ast.expr):
            dtype = self._eval_static(dtype)
        if dtype != i8:
            raise FrontendError("W.decode requires out_dtype=W.i8", self._location(call))
        result_type = with_element_type(codes.type, ScalarType(dtype))
        return self._emit(
            "weft_kernel.decode",
            call,
            operands=(codes, table, where),
            result_types=(result_type,),
            attributes={"out_dtype": emit_type(ScalarType(dtype))},
        )[0]

    def _intrinsic_narrow(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call,
            ("value", "dtype"),
            {"rounding": "rne", "saturation": False},
        )
        value = self._value_argument(args["value"], call)
        dtype = args["dtype"]
        if isinstance(dtype, ast.expr):
            dtype = self._eval_static(dtype)
        if not isinstance(dtype, DType):
            raise FrontendError("conversion dtype must be a Weft type", self._location(call))
        rounding = args["rounding"]
        saturation = args["saturation"]
        if isinstance(rounding, ast.expr):
            rounding = self._eval_static(rounding)
        if isinstance(saturation, ast.expr):
            saturation = self._eval_static(saturation)
        if (
            not isinstance(bare_type(value.type), RegionType)
            or element_type(value.type) != ScalarType(f32)
            or dtype != i8
            or rounding != "rne"
            or saturation is not True
        ):
            raise FrontendError(
                "W.narrow currently requires VLA f32 to i8 with rounding='rne' and saturation=True",
                self._location(call),
            )
        attributes = {
            "rounding": _string(str(rounding)),
            "saturation": _bool(bool(saturation)),
        }
        result_type = with_element_type(value.type, ScalarType(dtype))
        return self._emit(
            "weft_kernel.narrow",
            call,
            operands=(value,),
            result_types=(result_type,),
            attributes=attributes,
        )[0]

    def _require_extension_block(
        self, value: Value, name: str, extent: int, dtype: DType, call: ast.Call
    ) -> None:
        value_type = bare_type(value.type)
        if (
            is_masked(value.type)
            or not isinstance(value_type, BlockType)
            or value_type.shape != (extent,)
            or value_type.element_type != ScalarType(dtype)
        ):
            raise FrontendError(
                f"{name} must be an unmasked {dtype} block<{extent}>",
                self._location(call),
            )

    def _require_extension_scalar(
        self, value: Value, name: str, dtype: DType, call: ast.Call
    ) -> None:
        if is_masked(value.type) or value.type != ScalarType(dtype):
            raise FrontendError(
                f"{name} must be scalar {dtype}", self._location(call)
            )

    def _require_persistent_u8_pointer(
        self, value: Value, name: str, storage_format: str, call: ast.Call
    ) -> None:
        pointer = value.type if isinstance(value.type, PointerType) else None
        if (
            pointer is None
            or pointer.element_type != ScalarType(u8)
            or pointer.access == "write"
            or pointer.storage_class != "persistent"
            or pointer.storage_format != storage_format
        ):
            raise FrontendError(
                f"{name} must be a readable persistent u8 pointer with format "
                f"{storage_format!r}",
                self._location(call),
            )

    def _intrinsic_affine_i4_i8_dot(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call,
            ("activation", "packed_base"),
            {
                "activation_scale": None,
                "init": None,
            },
        )
        required = (
            "activation",
            "packed_base",
            "activation_scale",
            "init",
        )
        if any(args[name] is None for name in required):
            raise FrontendError(
                "affine_i4_i8_dot requires activation, packed_base, "
                "activation_scale, and init",
                self._location(call),
            )
        operands = tuple(self._value_argument(args[name], call) for name in required)
        self._require_extension_block(operands[0], "activation", 32, i8, call)
        self._require_persistent_u8_pointer(
            operands[1], "packed_base", "q4_k_n16_k32_304b", call
        )
        self._require_extension_scalar(
            operands[2], "activation_scale", f32, call
        )
        self._require_extension_block(operands[3], "init", 16, f32, call)
        return self._emit(
            "weft_ext.affine_i4_i8_dot",
            call,
            operands=operands,
            result_types=(operands[-1].type,),
        )[0]

    def _intrinsic_symmetric_i4_i8_dot(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call,
            ("activation", "packed_base"),
            {
                "activation_scale": None,
                "init": None,
            },
        )
        required = (
            "activation",
            "packed_base",
            "activation_scale",
            "init",
        )
        if any(args[name] is None for name in required):
            raise FrontendError(
                "symmetric_i4_i8_dot requires activation, packed_base, "
                "activation_scale, and init",
                self._location(call),
            )
        operands = tuple(self._value_argument(args[name], call) for name in required)
        self._require_extension_block(operands[0], "activation", 32, i8, call)
        self._require_persistent_u8_pointer(
            operands[1], "packed_base", "q4_0_n16_k32_288b", call
        )
        self._require_extension_scalar(
            operands[2], "activation_scale", f32, call
        )
        self._require_extension_block(operands[3], "init", 16, f32, call)
        return self._emit(
            "weft_ext.symmetric_i4_i8_dot",
            call,
            operands=operands,
            result_types=(operands[-1].type,),
        )[0]

    def _intrinsic_grouped_affine_i4_i8_dot(self, call: ast.Call) -> Value:
        names = (
            "packed_weight",
            "scale_min",
            "activation",
            "activation_sum_bytes",
            "dot_scale",
            "minimum_scale",
            "init",
        )
        args = self._positional_and_keywords(call, names, {})
        operands = tuple(self._value_argument(args[name], call) for name in names)
        for value, name, extent, dtype in (
            (operands[0], "packed_weight", 128, u8),
            (operands[1], "scale_min", 12, u8),
            (operands[2], "activation", 256, i8),
            (operands[3], "activation_sum_bytes", 32, u8),
        ):
            self._require_extension_block(value, name, extent, dtype, call)
        for value, name in (
            (operands[4], "dot_scale"),
            (operands[5], "minimum_scale"),
            (operands[6], "init"),
        ):
            self._require_extension_scalar(value, name, f32, call)
        return self._emit(
            "weft_ext.grouped_affine_i4_i8_dot",
            call,
            operands=operands,
            result_types=(operands[-1].type,),
        )[0]

    def _intrinsic_sign_bit_i8_dot(self, call: ast.Call) -> Value:
        names = (
            "sign_bits",
            "activation",
            "activation_scale",
            "sign_scale",
            "init",
        )
        args = self._positional_and_keywords(call, names, {})
        operands = tuple(self._value_argument(args[name], call) for name in names)
        self._require_extension_block(operands[0], "sign_bits", 4, u8, call)
        self._require_extension_block(operands[1], "activation", 32, i8, call)
        self._require_extension_scalar(
            operands[2], "activation_scale", f32, call
        )
        self._require_extension_scalar(operands[3], "sign_scale", f32, call)
        self._require_extension_scalar(operands[4], "init", f32, call)
        return self._emit(
            "weft_ext.sign_bit_i8_dot",
            call,
            operands=operands,
            result_types=(operands[-1].type,),
        )[0]

    def _intrinsic_e2m1_e8m0_i8_dot(self, call: ast.Call) -> Value:
        names = (
            "packed_codes",
            "exponent",
            "activation",
            "activation_scale",
            "init",
        )
        args = self._positional_and_keywords(call, names, {})
        operands = tuple(self._value_argument(args[name], call) for name in names)
        self._require_extension_block(operands[0], "packed_codes", 16, u8, call)
        self._require_extension_scalar(operands[1], "exponent", u8, call)
        self._require_extension_block(operands[2], "activation", 32, i8, call)
        self._require_extension_scalar(
            operands[3], "activation_scale", f32, call
        )
        self._require_extension_scalar(operands[4], "init", f32, call)
        return self._emit(
            "weft_ext.e2m1_e8m0_i8_dot",
            call,
            operands=operands,
            result_types=(operands[-1].type,),
        )[0]

    def _extension_scalar_dot(
        self,
        call: ast.Call,
        operation: str,
        names: tuple[str, ...],
        block_specs: tuple[tuple[str, int, DType], ...],
        scalar_specs: tuple[tuple[str, DType], ...],
    ) -> Value:
        args = self._positional_and_keywords(call, names, {})
        operands_by_name = {
            name: self._value_argument(args[name], call) for name in names
        }
        for name, extent, dtype in block_specs:
            self._require_extension_block(
                operands_by_name[name], name, extent, dtype, call
            )
        for name, dtype in scalar_specs:
            self._require_extension_scalar(operands_by_name[name], name, dtype, call)
        operands = tuple(operands_by_name[name] for name in names)
        return self._emit(
            operation,
            call,
            operands=operands,
            result_types=(operands[-1].type,),
        )[0]

    def _intrinsic_iq2_s_i8_dot(self, call: ast.Call) -> Value:
        return self._extension_scalar_dot(
            call,
            "weft_ext.iq2_s_i8_dot",
            (
                "codes",
                "high_bits",
                "sign_bits",
                "scales",
                "activation",
                "weight_scale",
                "activation_scale",
                "init",
            ),
            (
                ("codes", 32, u8),
                ("high_bits", 8, u8),
                ("sign_bits", 32, u8),
                ("scales", 8, u8),
                ("activation", 256, i8),
            ),
            (
                ("weight_scale", f32),
                ("activation_scale", f32),
                ("init", f32),
            ),
        )

    def _intrinsic_iq3_s_i8_dot(self, call: ast.Call) -> Value:
        return self._extension_scalar_dot(
            call,
            "weft_ext.iq3_s_i8_dot",
            (
                "codes",
                "high_bits",
                "sign_bits",
                "scales",
                "activation",
                "weight_scale",
                "activation_scale",
                "init",
            ),
            (
                ("codes", 64, u8),
                ("high_bits", 8, u8),
                ("sign_bits", 32, u8),
                ("scales", 4, u8),
                ("activation", 256, i8),
            ),
            (
                ("weight_scale", f32),
                ("activation_scale", f32),
                ("init", f32),
            ),
        )

    def _intrinsic_iq1_m_i8_dot(self, call: ast.Call) -> Value:
        return self._extension_scalar_dot(
            call,
            "weft_ext.iq1_m_i8_dot",
            (
                "codes",
                "high_delta_bits",
                "scales",
                "activation",
                "activation_scale",
                "init",
            ),
            (
                ("codes", 32, u8),
                ("high_delta_bits", 16, u8),
                ("scales", 8, u8),
                ("activation", 256, i8),
            ),
            (("activation_scale", f32), ("init", f32)),
        )

    def _intrinsic_q6_k_i8_dot(self, call: ast.Call) -> Value:
        return self._extension_scalar_dot(
            call,
            "weft_ext.q6_k_i8_dot",
            (
                "low_bits",
                "high_bits",
                "group_scales",
                "activation",
                "weight_scale",
                "activation_scale",
                "init",
            ),
            (
                ("low_bits", 128, u8),
                ("high_bits", 64, u8),
                ("group_scales", 16, i8),
                ("activation", 256, i8),
            ),
            (
                ("weight_scale", f32),
                ("activation_scale", f32),
                ("init", f32),
            ),
        )

    def _intrinsic_range(self, call: ast.Call) -> Value:
        raise FrontendError("W.range is valid only in a for statement", self._location(call))

    def _intrinsic_vla(self, call: ast.Call) -> Value:
        raise FrontendError("W.vla is valid only in a with statement", self._location(call))

    def _compile_for(self, statement: ast.For) -> None:
        if statement.orelse:
            raise FrontendError("for-else is not part of the Weft DSL", self._location(statement))
        if not isinstance(statement.target, ast.Name) or not isinstance(statement.iter, ast.Call):
            raise FrontendError("for must use `for name in W.range(...)`", self._location(statement))
        callee = self._resolve_static(statement.iter.func)
        if not isinstance(callee, Intrinsic) or callee.name != "range":
            raise FrontendError("for iteration must be W.range", self._location(statement))
        args = self._positional_and_keywords(
            statement.iter, ("begin", "end", "step"), {"step": 1}
        )
        lower = self._value_argument(args["begin"], statement.iter)
        upper = self._value_argument(args["end"], statement.iter)
        step = self._value_argument(args["step"], statement.iter)
        if not all(_is_index(value.type) and _is_scalar(value.type) for value in (lower, upper, step)):
            raise FrontendError("W.range bounds and step must be scalar index", self._location(statement))
        carried_names = sorted(_assigned_names(statement.body) & self.env.keys())
        init_values = tuple(self.env[name] for name in carried_names)
        body = self.builder.region(
            (ScalarType(index),) + tuple(value.type for value in init_values),
            (statement.target.id,) + tuple(carried_names),
        )
        previous_block, previous_env = self.block, self.env
        self.block = body
        self.env = dict(previous_env)
        self.env[statement.target.id] = body.arguments[0]
        for name, argument in zip(carried_names, body.arguments[1:]):
            self.env[name] = argument
        self._compile_statements(statement.body)
        yielded = tuple(self.env[name] for name in carried_names)
        self._emit("weft_kernel.yield", statement, operands=yielded)
        self.block, self.env = previous_block, previous_env
        results = self._emit(
            "weft_kernel.for",
            statement,
            operands=(lower, upper, step) + init_values,
            result_types=tuple(value.type for value in init_values),
            regions=(body,),
            result_names=tuple(carried_names),
        )
        for name, result in zip(carried_names, results):
            self.env[name] = result

    def _compile_with(self, statement: ast.With, live_after: set[str]) -> None:
        if len(statement.items) != 1:
            raise FrontendError("with statement must contain one W.vla", self._location(statement))
        item = statement.items[0]
        if not isinstance(item.context_expr, ast.Call) or not isinstance(item.optional_vars, ast.Name):
            raise FrontendError("use `with W.vla(begin, end) as i`", self._location(statement))
        callee = self._resolve_static(item.context_expr.func)
        if not isinstance(callee, Intrinsic) or callee.name != "vla":
            raise FrontendError("only W.vla is a Weft context manager", self._location(statement))
        if self.active_vla:
            raise FrontendError("VLA regions cannot be nested", self._location(statement))
        args = self._positional_and_keywords(item.context_expr, ("begin", "end"), {})
        begin = self._value_argument(args["begin"], item.context_expr)
        end = self._value_argument(args["end"], item.context_expr)
        if not all(_is_index(value.type) and _is_scalar(value.type) for value in (begin, end)):
            raise FrontendError("VLA bounds must be scalar index", self._location(statement))
        assigned = _assigned_names(statement.body)
        outer_names = set(self.env)
        mutated = assigned & outer_names
        if mutated:
            name = sorted(mutated)[0]
            raise FrontendError(
                f"VLA body cannot mutate outer state {name!r}; use reduce/scan or an explicit summary primitive",
                self._location(statement),
            )
        exported_names = sorted((assigned & live_after) - {item.optional_vars.id})
        body = self.builder.region(
            (RegionType((-1,), (-1,), ScalarType(index)),),
            (item.optional_vars.id,),
        )
        previous = (self.block, self.env, self.active_vla, self.vla_outer_names)
        self.block = body
        self.env = dict(previous[1])
        self.env[item.optional_vars.id] = body.arguments[0]
        self.active_vla = True
        self.vla_outer_names = outer_names
        self._compile_statements(statement.body)
        exported_values = tuple(self.env[name] for name in exported_names)
        for name, value in zip(exported_names, exported_values):
            if shape_kind(value.type) == "region" or (
                isinstance(value.type, MaskedType)
                and shape_kind(value.type.value_type) == "region"
            ):
                raise FrontendError(
                    f"VLA value {name!r} cannot escape its lexical region",
                    self._location(statement),
                )
        self._emit("weft_kernel.yield", statement, operands=exported_values)
        self.block, self.env, self.active_vla, self.vla_outer_names = previous
        results = self._emit(
            "weft_kernel.vla",
            statement,
            operands=(begin, end),
            result_types=tuple(value.type for value in exported_values),
            regions=(body,),
            result_names=tuple(exported_names),
        )
        for name, result in zip(exported_names, results):
            self.env[name] = result

    def _compile_if(self, statement: ast.If, live_after: set[str]) -> None:
        condition = self._expect_value(self._compile_expr(statement.test), statement.test)
        if condition.type != ScalarType(i1):
            raise FrontendError(
                "Python if requires scalar i1; use W.select for logical predicates",
                self._location(statement.test),
            )
        assigned = _assigned_names(statement.body) | _assigned_names(statement.orelse)
        merged_names = sorted(assigned & (set(self.env) | live_after))
        outer_env = self.env

        def compile_branch(statements: Sequence[ast.stmt]) -> tuple[Region, tuple[Value, ...]]:
            region = self.builder.region(())
            previous_block, previous_env = self.block, self.env
            self.block, self.env = region, dict(outer_env)
            self._compile_statements(statements)
            values: list[Value] = []
            for name in merged_names:
                if name not in self.env:
                    raise FrontendError(
                        f"if branch does not define merged value {name!r}",
                        self._location(statement),
                    )
                values.append(self.env[name])
            self._emit("weft_kernel.yield", statement, operands=tuple(values))
            self.block, self.env = previous_block, previous_env
            return region, tuple(values)

        then_region, then_values = compile_branch(statement.body)
        else_region, else_values = compile_branch(statement.orelse)
        for left, right in zip(then_values, else_values):
            if left.type != right.type:
                raise FrontendError("if branch result types must match", self._location(statement))
        results = self._emit(
            "weft_kernel.if",
            statement,
            operands=(condition,),
            result_types=tuple(value.type for value in then_values),
            regions=(then_region, else_region),
            result_names=tuple(merged_names),
        )
        for name, result in zip(merged_names, results):
            self.env[name] = result

    def _compile_while(self, statement: ast.While) -> None:
        if statement.orelse:
            raise FrontendError("while-else is not part of the Weft DSL", self._location(statement))
        carried_names = sorted(_assigned_names(statement.body) & self.env.keys())
        init_values = tuple(self.env[name] for name in carried_names)
        condition_region = self.builder.region(
            tuple(value.type for value in init_values), tuple(carried_names)
        )
        body_region = self.builder.region(
            tuple(value.type for value in init_values), tuple(carried_names)
        )
        outer_block, outer_env = self.block, self.env
        self.block, self.env = condition_region, dict(outer_env)
        for name, argument in zip(carried_names, condition_region.arguments):
            self.env[name] = argument
        condition = self._expect_value(self._compile_expr(statement.test), statement.test)
        if condition.type != ScalarType(i1):
            raise FrontendError("while condition must be scalar i1", self._location(statement.test))
        self._emit(
            "weft_kernel.condition",
            statement,
            operands=(condition,) + condition_region.arguments,
        )
        self.block, self.env = body_region, dict(outer_env)
        for name, argument in zip(carried_names, body_region.arguments):
            self.env[name] = argument
        self._compile_statements(statement.body)
        self._emit(
            "weft_kernel.yield",
            statement,
            operands=tuple(self.env[name] for name in carried_names),
        )
        self.block, self.env = outer_block, outer_env
        results = self._emit(
            "weft_kernel.while",
            statement,
            operands=init_values,
            result_types=tuple(value.type for value in init_values),
            regions=(condition_region, body_region),
            result_names=tuple(carried_names),
        )
        for name, result in zip(carried_names, results):
            self.env[name] = result


def lower_to_mlir(definition: KernelDefinition[object, object]) -> str:
    if not isinstance(definition, KernelDefinition):
        raise TypeError("lower_to_mlir expects an @weft.kernel definition")
    return FrontendCompiler(definition).compile()
