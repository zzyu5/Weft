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
from weft.language import f32
from weft.language import i1
from weft.language import index
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
                annotation.address_space,
                annotation.access,
                annotation.noalias,
                annotation.alignment,
                annotation.restrict_like,
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


def _join_shapes(lhs: ValueType, rhs: ValueType, location: SourceLocation) -> tuple[str, tuple[int, ...]]:
    lhs_kind = shape_kind(lhs)
    rhs_kind = shape_kind(rhs)
    lhs_shape = shape_of(lhs)
    rhs_shape = shape_of(rhs)
    if lhs_kind == "scalar":
        return rhs_kind, rhs_shape or ()
    if rhs_kind == "scalar":
        return lhs_kind, lhs_shape or ()
    if lhs_shape is None or rhs_shape is None or len(lhs_shape) != len(rhs_shape):
        raise FrontendError("logical ranks are not broadcast-compatible", location)
    result: list[int] = []
    for left, right in zip(lhs_shape, rhs_shape):
        if left == right:
            result.append(left)
        elif left == 1:
            result.append(right)
        elif right == 1:
            result.append(left)
        else:
            raise FrontendError("logical shapes are not broadcast-compatible", location)
    kind = "region" if "region" in {lhs_kind, rhs_kind} else "block"
    return kind, tuple(result)


class FrontendCompiler:
    def __init__(self, definition: KernelDefinition[object, object]) -> None:
        self.unit = SourceUnit.from_definition(definition)
        self.source: SourceUnit | HelperSource = self.unit
        self.builder = IRBuilder()
        self.block: Region | None = None
        self.env: dict[str, Value] = {}
        self.active_vla = False
        self.vla_outer_names: set[str] | None = None
        self.helper_effects: tuple[str, ...] | None = None

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
                    "chained assignment is not part of Weft source",
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
                        "exceptional": _string("preserve"),
                    },
                )[0]
            if isinstance(expression.op, ast.Invert):
                one = self._constant(True, i1, expression)
                return self._emit_binary_values("xor", operand, one, expression)
            raise FrontendError("unsupported unary operator", self._location(expression))
        if isinstance(expression, ast.Compare):
            if len(expression.ops) != 1 or len(expression.comparators) != 1:
                raise FrontendError(
                    "chained comparisons are not part of Weft source",
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
            kind, shape = _join_shapes(lhs.type, rhs.type, self._location(expression))
            result_type: ValueType = shaped_type(kind, shape, ScalarType(i1))
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
        return self._emit(
            "weft_kernel.constant",
            node,
            result_types=(value_type,),
            attributes={"value": spelling},
        )[0]

    def _invalid(self, node: ast.AST) -> Value:
        return self._emit(
            "weft_kernel.invalid", node, result_types=(NONE_TYPE,)
        )[0]

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
            kind, shape = _join_shapes(lhs.type, rhs.type, self._location(expression))
            result_type = shaped_type(kind, shape, element_type(lhs.type))
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
        result_kind, result_shape = _join_shapes(lhs.type, rhs.type, self._location(node))
        result_type: ValueType = shaped_type(
            result_kind, result_shape, element_type(lhs.type)
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

    def _intrinsic_select(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("predicate", "a", "b"), {})
        predicate = self._value_argument(args["predicate"], call)
        true_value = self._value_argument(args["a"], call)
        false_value = self._value_argument(args["b"], call)
        if not _is_predicate(predicate.type):
            raise FrontendError("W.select predicate must contain i1", self._location(call))
        if element_type(true_value.type) != element_type(false_value.type):
            raise FrontendError("W.select value types must match", self._location(call))
        kind, shape = _join_shapes(true_value.type, false_value.type, self._location(call))
        result_type: ValueType = shaped_type(kind, shape, element_type(true_value.type))
        if is_masked(true_value.type) or is_masked(false_value.type):
            result_type = MaskedType(result_type)
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
        if isinstance(other.type, NoneType):
            result_type = MaskedType(result_type)
        elif element_type(other.type) != pointer_type.element_type:
            raise FrontendError("W.load other must match pointer element type", self._location(call))
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

    def _intrinsic_prefetch(self, call: ast.Call) -> None:
        self._require_effect("read", call)
        args = self._positional_and_keywords(
            call, ("ptr",), {"where": True, "locality": "default"}
        )
        pointer = self._value_argument(args["ptr"], call)
        where = self._value_argument(args["where"], call)
        locality = args["locality"]
        if isinstance(locality, ast.expr):
            locality = self._eval_static(locality)
        self._emit(
            "weft_kernel.prefetch",
            call,
            operands=(pointer, where),
            attributes={"locality": _string(str(locality))},
        )
        return None

    def _intrinsic_atomic_add(self, call: ast.Call) -> Value:
        self._require_effect("atomic", call)
        args = self._positional_and_keywords(
            call, ("ptr", "value"), {"where": True, "order": "relaxed"}
        )
        pointer = self._value_argument(args["ptr"], call)
        value = self._value_argument(args["value"], call)
        where = self._value_argument(args["where"], call)
        order = args["order"]
        if isinstance(order, ast.expr):
            order = self._eval_static(order)
        result_type = with_element_type(pointer.type, element_type(value.type))
        return self._emit(
            "weft_kernel.atomic_add",
            call,
            operands=(pointer, value, where),
            result_types=(result_type,),
            attributes={"order": _string(str(order))},
        )[0]

    def _intrinsic_fence(self, call: ast.Call) -> None:
        self._require_effect("fence", call)
        args = self._positional_and_keywords(call, (), {"order": "acq_rel"})
        order = args["order"]
        if isinstance(order, ast.expr):
            order = self._eval_static(order)
        self._emit(
            "weft_kernel.fence", call, attributes={"order": _string(str(order))}
        )
        return None

    def _intrinsic_valid(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("value",), {})
        value = self._value_argument(args["value"], call)
        if not isinstance(value.type, MaskedType):
            raise FrontendError("W.valid expects a masked value", self._location(call))
        result_type = with_element_type(value.type.value_type, ScalarType(i1))
        return self._emit(
            "weft_kernel.valid",
            call,
            operands=(value,),
            result_types=(result_type,),
        )[0]

    def _intrinsic_fill(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("value", "fill_value"), {})
        value = self._value_argument(args["value"], call)
        fill_value = self._value_argument(args["fill_value"], call)
        if not isinstance(value.type, MaskedType):
            raise FrontendError("W.fill expects a masked value", self._location(call))
        return self._emit(
            "weft_kernel.fill",
            call,
            operands=(value, fill_value),
            result_types=(value.type.value_type,),
        )[0]

    def _intrinsic_block_axis(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("extent",), {"offset": 0})
        extent = self._value_argument(args["extent"], call)
        offset = self._value_argument(args["offset"], call)
        if not _is_index(extent.type) or not _is_index(offset.type):
            raise FrontendError("block axis extent/offset must be index", self._location(call))
        dimension = -1
        extent_node = args["extent"]
        if isinstance(extent_node, ast.Constant) and isinstance(extent_node.value, int):
            dimension = extent_node.value
        result_type = BlockType((dimension,), ScalarType(index))
        return self._emit(
            "weft_kernel.block_axis",
            call,
            operands=(extent, offset),
            result_types=(result_type,),
        )[0]

    def _intrinsic_full(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call, ("shape", "value"), {"dtype": None}
        )
        if not isinstance(args["shape"], ast.expr):
            raise FrontendError("shape must be source syntax", self._location(call))
        shape = self._shape_spec(args["shape"])
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
        result_type = BlockType(shape.dimensions, value.type)
        return self._emit(
            "weft_kernel.full",
            call,
            operands=(value,) + shape.extents,
            result_types=(result_type,),
            attributes={"shape": _dense_i64(shape.dimensions)},
        )[0]

    def _intrinsic_zeros(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("shape", "dtype"), {})
        dtype = args["dtype"]
        if not isinstance(dtype, ast.expr):
            raise FrontendError("dtype must be source syntax", self._location(call))
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
        if shape is None or axis < 0 or axis > len(shape):
            raise FrontendError("expand_dims axis is outside logical rank", self._location(node))
        result_shape = shape[:axis] + (1,) + shape[axis:]
        result_type = shaped_type(shape_kind(value.type), result_shape, element_type(value.type))
        if is_masked(value.type):
            result_type = MaskedType(result_type)
        return self._emit(
            "weft_kernel.expand_dims",
            node,
            operands=(value,),
            result_types=(result_type,),
            attributes={"axis": str(axis)},
        )[0]

    def _intrinsic_expand_dims(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(call, ("value", "axis"), {})
        value = self._value_argument(args["value"], call)
        axis = args["axis"]
        if isinstance(axis, ast.expr):
            axis = self._eval_static(axis)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise FrontendError("axis must be a constant integer", self._location(call))
        return self._expand_dims(value, axis, call)

    def _shape_transform(self, call: ast.Call, operation: str) -> Value:
        args = self._positional_and_keywords(call, ("value", "shape"), {})
        value = self._value_argument(args["value"], call)
        if not isinstance(args["shape"], ast.expr):
            raise FrontendError("shape must be source syntax", self._location(call))
        shape = self._shape_spec(args["shape"])
        result_type = shaped_type(shape_kind(value.type), shape.dimensions, element_type(value.type))
        if is_masked(value.type):
            result_type = MaskedType(result_type)
        return self._emit(
            operation,
            call,
            operands=(value,) + shape.extents,
            result_types=(result_type,),
            attributes={"shape": _dense_i64(shape.dimensions)},
        )[0]

    def _intrinsic_broadcast_to(self, call: ast.Call) -> Value:
        return self._shape_transform(call, "weft_kernel.broadcast_to")

    def _intrinsic_reshape(self, call: ast.Call) -> Value:
        return self._shape_transform(call, "weft_kernel.reshape")

    def _permutation(self, call: ast.Call, operation: str) -> Value:
        args = self._positional_and_keywords(call, ("value", "permutation"), {})
        value = self._value_argument(args["value"], call)
        permutation = args["permutation"]
        if not isinstance(permutation, ast.expr):
            raise FrontendError("permutation must be source syntax", self._location(call))
        static = self._eval_static(permutation)
        if not isinstance(static, tuple) or any(
            isinstance(axis, bool) or not isinstance(axis, int) for axis in static
        ):
            raise FrontendError("permutation must be a tuple of integers", self._location(call))
        shape = shape_of(value.type)
        if shape is None or sorted(static) != list(range(len(shape))):
            raise FrontendError("permutation must contain every logical axis", self._location(call))
        result_shape = tuple(shape[axis] for axis in static)
        result_type = shaped_type(shape_kind(value.type), result_shape, element_type(value.type))
        if is_masked(value.type):
            result_type = MaskedType(result_type)
        return self._emit(
            operation,
            call,
            operands=(value,),
            result_types=(result_type,),
            attributes={"permutation": _dense_i64(static)},
        )[0]

    def _intrinsic_transpose(self, call: ast.Call) -> Value:
        return self._permutation(call, "weft_kernel.transpose")

    def _intrinsic_permute(self, call: ast.Call) -> Value:
        return self._permutation(call, "weft_kernel.permute")

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
        result_type = with_element_type(value.type, ScalarType(dtype))
        return self._emit(
            "weft_kernel.bitcast",
            call,
            operands=(value,),
            result_types=(result_type,),
        )[0]

    def _cast_value(self, value: Value, dtype: DType, node: ast.AST) -> Value:
        result_type = with_element_type(value.type, ScalarType(dtype))
        return self._emit(
            "weft_kernel.cast", node, operands=(value,), result_types=(result_type,)
        )[0]

    def _math_unary(self, call: ast.Call, kind: str) -> Value:
        args = self._positional_and_keywords(
            call, ("value",), {"math": "native", "exceptional": "preserve"}
        )
        value = self._value_argument(args["value"], call)
        math = args["math"]
        exceptional = args["exceptional"]
        if isinstance(math, ast.expr):
            math = self._eval_static(math)
        if isinstance(exceptional, ast.expr):
            exceptional = self._eval_static(exceptional)
        return self._emit(
            "weft_kernel.unary",
            call,
            operands=(value,),
            result_types=(value.type,),
            attributes={
                "kind": _string(kind),
                "math": _string(str(math)),
                "exceptional": _string(str(exceptional)),
            },
        )[0]

    def _intrinsic_exp(self, call: ast.Call) -> Value:
        return self._math_unary(call, "exp")

    def _intrinsic_exp2(self, call: ast.Call) -> Value:
        return self._math_unary(call, "exp2")

    def _intrinsic_log(self, call: ast.Call) -> Value:
        return self._math_unary(call, "log")

    def _intrinsic_sin(self, call: ast.Call) -> Value:
        return self._math_unary(call, "sin")

    def _intrinsic_cos(self, call: ast.Call) -> Value:
        return self._math_unary(call, "cos")

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
        shape = shape_of(bare)
        if isinstance(bare, RegionType) and axis is None:
            remaining_shape = bare.shape[1:]
            result_type: ValueType = (
                BlockType(remaining_shape, acc_type) if remaining_shape else acc_type
            )
            axis_value = -1
        elif isinstance(bare, BlockType) and isinstance(axis, int) and not isinstance(axis, bool):
            if axis < 0 or axis >= len(bare.shape):
                raise FrontendError("reduce axis is outside logical rank", self._location(call))
            result_shape = bare.shape[:axis] + bare.shape[axis + 1 :]
            result_type = BlockType(result_shape, acc_type) if result_shape else acc_type
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

    def _intrinsic_summary_fold(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call,
            ("value",),
            {
                "identity": None,
                "lift": None,
                "merge": None,
                "finalize": None,
                "where": True,
                "coordinate": None,
                "order": "preserve",
            },
        )
        value = self._value_argument(args["value"], call)
        if args["identity"] is None:
            raise FrontendError("summary_fold requires identity", self._location(call))
        identity = self._value_argument(args["identity"], call)
        where = self._value_argument(args["where"], call)
        coordinate = self._value_argument(args["coordinate"], call)
        lift = self._resolve_helper_argument(args["lift"], call, "lift")
        merge = self._resolve_helper_argument(args["merge"], call, "merge")
        finalize_static = args["finalize"]
        if isinstance(finalize_static, ast.expr):
            finalize_static = self._eval_static(finalize_static)
        if finalize_static is not None and not isinstance(finalize_static, HelperDefinition):
            raise FrontendError("finalize must be a pure helper or None", self._location(call))
        order = args["order"]
        if isinstance(order, ast.expr):
            order = self._eval_static(order)

        input_element = element_type(bare_type(value.type))
        lift_arguments = (input_element,)
        if not isinstance(coordinate.type, NoneType):
            if shape_kind(coordinate.type) != shape_kind(value.type) or shape_of(
                coordinate.type
            ) != shape_of(value.type):
                raise FrontendError(
                    "summary coordinate must share the input logical domain",
                    self._location(call),
                )
            lift_arguments += (element_type(bare_type(coordinate.type)),)
        lift_region, lift_result = self._compile_helper_region(
            lift, lift_arguments, call
        )
        if lift_result != identity.type:
            raise FrontendError("lift result must match identity state", self._location(call))
        merge_region, merge_result = self._compile_helper_region(
            merge, (identity.type, identity.type), call
        )
        if merge_result != identity.type:
            raise FrontendError("merge must be closed over state type", self._location(call))
        if finalize_static is None:
            finalize_region = self.builder.region((identity.type,), ("state",))
            previous_block = self.block
            self.block = finalize_region
            self._emit(
                "weft_kernel.yield",
                call,
                operands=(finalize_region.arguments[0],),
            )
            self.block = previous_block
            result_type = identity.type
        else:
            if not finalize_static.pure:
                raise FrontendError("summary finalize must be pure", self._location(call))
            finalize_region, result_type = self._compile_helper_region(
                finalize_static, (identity.type,), call
            )
        return self._emit(
            "weft_kernel.summary_fold",
            call,
            operands=(value, identity, where, coordinate),
            result_types=(result_type,),
            attributes={"order": _string(str(order))},
            regions=(lift_region, merge_region, finalize_region),
        )[0]

    def _resolve_helper_argument(
        self, argument: ast.expr | object, call: ast.Call, name: str
    ) -> HelperDefinition:
        value = self._eval_static(argument) if isinstance(argument, ast.expr) else argument
        if not isinstance(value, HelperDefinition) or not value.pure:
            raise FrontendError(f"{name} must be a @W.pure helper", self._location(call))
        return value

    def _compile_helper_region(
        self,
        definition: HelperDefinition,
        argument_types: tuple[ValueType, ...],
        call: ast.Call,
    ) -> tuple[Region, ValueType]:
        source = HelperSource.from_definition(definition)
        parameters = source.function.args.args
        if len(parameters) != len(argument_types):
            raise FrontendError(
                f"helper {definition.__name__} argument count mismatch",
                self._location(call),
            )
        region = self.builder.region(
            argument_types, tuple(parameter.arg for parameter in parameters)
        )
        result = self._compile_helper_body(
            definition, source, region, region.arguments, call
        )
        previous_block = self.block
        self.block = region
        self._emit("weft_kernel.yield", call, operands=(result,))
        self.block = previous_block
        return region, result.type

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

    def _contract_arguments(self, call: ast.Call) -> tuple[dict[str, ast.expr | object], Value, Value]:
        args = self._positional_and_keywords(
            call,
            ("lhs", "rhs"),
            {
                "init": None,
                "lhs_axes": None,
                "rhs_axes": None,
                "output_order": None,
                "acc_dtype": None,
                "out_dtype": None,
                "where_lhs": True,
                "where_rhs": True,
                "order": "relaxed",
                "math": "native",
            },
        )
        lhs = self._value_argument(args["lhs"], call)
        rhs = self._value_argument(args["rhs"], call)
        return args, lhs, rhs

    def _intrinsic_contract(self, call: ast.Call) -> Value:
        args, lhs, rhs = self._contract_arguments(call)
        return self._emit_contract(call, args, lhs, rhs)

    def _intrinsic_dot(self, call: ast.Call) -> Value:
        args, lhs, rhs = self._contract_arguments(call)
        lhs_shape = shape_of(lhs.type)
        rhs_shape = shape_of(rhs.type)
        if lhs_shape is None or rhs_shape is None or len(lhs_shape) not in {1, 2} or len(rhs_shape) not in {1, 2}:
            raise FrontendError("W.dot supports rank-one or rank-two blocks", self._location(call))
        args["lhs_axes"] = (len(lhs_shape) - 1,)
        args["rhs_axes"] = (0,)
        return self._emit_contract(call, args, lhs, rhs)

    def _emit_contract(
        self,
        call: ast.Call,
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
                "contract operands must be logical block or VLA region values",
                self._location(call),
            )
        static: dict[str, object] = {}
        for name in ("lhs_axes", "rhs_axes", "output_order", "acc_dtype", "out_dtype", "order", "math"):
            item = args[name]
            static[name] = self._eval_static(item) if isinstance(item, ast.expr) else item
        lhs_axes = static["lhs_axes"]
        rhs_axes = static["rhs_axes"]
        if not isinstance(lhs_axes, tuple) or not isinstance(rhs_axes, tuple) or not lhs_axes:
            raise FrontendError("contract requires non-empty axis tuples", self._location(call))
        if len(lhs_axes) != len(rhs_axes) or any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in lhs_axes + rhs_axes
        ):
            raise FrontendError("contract axes must be equally sized integer tuples", self._location(call))
        if isinstance(lhs_bare, RegionType) and 0 in lhs_axes:
            raise FrontendError(
                "the active VLA axis cannot be a contraction axis",
                self._location(call),
            )
        if isinstance(rhs_bare, RegionType) and 0 in rhs_axes:
            raise FrontendError(
                "the active VLA axis cannot be a contraction axis",
                self._location(call),
            )
        lhs_free = [
            axis
            for axis in range(len(lhs_bare.shape))
            if axis not in lhs_axes
            and not (isinstance(lhs_bare, RegionType) and axis == 0)
        ]
        rhs_free = [
            axis
            for axis in range(len(rhs_bare.shape))
            if axis not in rhs_axes
            and not (isinstance(rhs_bare, RegionType) and axis == 0)
        ]
        has_region = isinstance(lhs_bare, RegionType) or isinstance(
            rhs_bare, RegionType
        )
        output_shape = ([-1] if has_region else []) + [
            lhs_bare.shape[axis] for axis in lhs_free
        ] + [
            rhs_bare.shape[axis] for axis in rhs_free
        ]
        output_order = static["output_order"]
        if output_order is not None:
            if not isinstance(output_order, tuple) or sorted(output_order) != list(range(len(output_shape))):
                raise FrontendError("output_order must permute output axes", self._location(call))
            if has_region and output_order[0] != 0:
                raise FrontendError(
                    "output_order must keep the active VLA axis first",
                    self._location(call),
                )
            output_shape = [output_shape[axis] for axis in output_order]
        acc_dtype = static["acc_dtype"]
        if acc_dtype is None:
            if args["init"] is None:
                acc_type = element_type(lhs.type)
            else:
                provisional_init = self._value_argument(args["init"], call)
                acc_type = element_type(provisional_init.type)
                args["init"] = provisional_init
        elif isinstance(acc_dtype, DType):
            acc_type = ScalarType(acc_dtype)
        else:
            raise FrontendError("acc_dtype must be a Weft dtype", self._location(call))
        out_dtype = static["out_dtype"]
        out_type = ScalarType(out_dtype) if isinstance(out_dtype, DType) else acc_type
        result_type: ValueType = (
            RegionType(tuple(output_shape), out_type)
            if has_region
            else BlockType(tuple(output_shape), out_type)
            if output_shape
            else out_type
        )
        init_arg = args["init"]
        if isinstance(init_arg, Value):
            init = init_arg
        elif init_arg is None:
            zero = 0.0 if isinstance(acc_type, ScalarType) and acc_type.dtype.category in {DTypeCategory.FLOAT, DTypeCategory.BFLOAT} else 0
            assert isinstance(acc_type, ScalarType)
            init = self._constant(zero, acc_type.dtype, call)
        else:
            init = self._value_argument(init_arg, call)
        if shape_kind(init.type) != "scalar" and bare_type(init.type) != result_type:
            raise FrontendError(
                "contract init must be scalar or exactly output-shaped",
                self._location(call),
            )
        where_lhs = self._value_argument(args["where_lhs"], call)
        where_rhs = self._value_argument(args["where_rhs"], call)
        return self._emit(
            "weft_kernel.contract",
            call,
            operands=(lhs, rhs, init, where_lhs, where_rhs),
            result_types=(result_type,),
            attributes={
                "lhs_axes": _dense_i64(lhs_axes),
                "rhs_axes": _dense_i64(rhs_axes),
                "output_order": _dense_i64(output_order or ()),
                "acc_dtype": emit_type(acc_type),
                "out_dtype": emit_type(out_type),
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
        if not isinstance(bare_type(table.type), BlockType):
            raise FrontendError(
                "W.lookup table must be a logical block", self._location(call)
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
        if not isinstance(bare_type(table.type), BlockType):
            raise FrontendError(
                "W.decode table must be a logical block", self._location(call)
            )
        dtype = args["out_dtype"]
        if isinstance(dtype, ast.expr):
            dtype = self._eval_static(dtype)
        if not isinstance(dtype, DType):
            raise FrontendError("W.decode requires out_dtype", self._location(call))
        result_type = with_element_type(codes.type, ScalarType(dtype))
        return self._emit(
            "weft_kernel.decode",
            call,
            operands=(codes, table, where),
            result_types=(result_type,),
            attributes={"out_dtype": emit_type(ScalarType(dtype))},
        )[0]

    def _convert(self, call: ast.Call, operation: str, narrow: bool) -> Value:
        defaults = {"rounding": "rne", "saturation": False} if narrow else {}
        args = self._positional_and_keywords(call, ("value", "dtype"), defaults)
        value = self._value_argument(args["value"], call)
        dtype = args["dtype"]
        if isinstance(dtype, ast.expr):
            dtype = self._eval_static(dtype)
        if not isinstance(dtype, DType):
            raise FrontendError("conversion dtype must be a Weft type", self._location(call))
        attributes: dict[str, str] = {}
        if narrow:
            rounding = args["rounding"]
            saturation = args["saturation"]
            if isinstance(rounding, ast.expr):
                rounding = self._eval_static(rounding)
            if isinstance(saturation, ast.expr):
                saturation = self._eval_static(saturation)
            attributes = {
                "rounding": _string(str(rounding)),
                "saturation": _bool(bool(saturation)),
            }
        result_type = with_element_type(value.type, ScalarType(dtype))
        return self._emit(
            operation,
            call,
            operands=(value,),
            result_types=(result_type,),
            attributes=attributes,
        )[0]

    def _intrinsic_widen(self, call: ast.Call) -> Value:
        return self._convert(call, "weft_kernel.widen", False)

    def _intrinsic_narrow(self, call: ast.Call) -> Value:
        return self._convert(call, "weft_kernel.narrow", True)

    def _intrinsic_affine_i4_i8_contract(self, call: ast.Call) -> Value:
        args = self._positional_and_keywords(
            call,
            ("activation", "packed_weight"),
            {
                "activation_scale": None,
                "weight_scale": None,
                "weight_zero_point": None,
                "init": None,
            },
        )
        required = (
            "activation",
            "packed_weight",
            "activation_scale",
            "weight_scale",
            "weight_zero_point",
            "init",
        )
        if any(args[name] is None for name in required):
            raise FrontendError(
                "affine_i4_i8_contract requires packed operands, scales, "
                "zero point, and init",
                self._location(call),
            )
        operands = tuple(self._value_argument(args[name], call) for name in required)
        return self._emit(
            "weft_ext.affine_i4_i8_contract",
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
        return self._emit(
            "weft_ext.grouped_affine_i4_i8_dot",
            call,
            operands=operands,
            result_types=(operands[-1].type,),
        )[0]

    def _intrinsic_range(self, call: ast.Call) -> Value:
        raise FrontendError("W.range is valid only in a for statement", self._location(call))

    def _intrinsic_vla(self, call: ast.Call) -> Value:
        raise FrontendError("W.vla is valid only in a with statement", self._location(call))

    def _compile_for(self, statement: ast.For) -> None:
        if statement.orelse:
            raise FrontendError("for-else is not part of Weft source", self._location(statement))
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
                f"VLA body cannot mutate outer state {name!r}; use reduce/scan/summary_fold",
                self._location(statement),
            )
        exported_names = sorted((assigned & live_after) - {item.optional_vars.id})
        body = self.builder.region(
            (RegionType((-1,), ScalarType(index)),), (item.optional_vars.id,)
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
            raise FrontendError("while-else is not part of Weft source", self._location(statement))
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
