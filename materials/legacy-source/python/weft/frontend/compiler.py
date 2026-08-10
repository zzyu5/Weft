from __future__ import annotations

import ast
import inspect
import json
import math
from dataclasses import dataclass

from weft.api import KernelDefinition
from weft.diagnostics import FrontendError
from weft.diagnostics import SourceLocation
from weft.language import ConstexprSpec
from weft.language import DType
from weft.language import DTypeCategory
from weft.language import Intrinsic
from weft.language import PtrSpec
from weft.language import bool as bool_dtype
from weft.language import f32
from weft.language import index

from .ir import IRBuilder
from .ir import Operation
from .ir import Region
from .ir import Value
from .source import SourceUnit
from .types import BlockType
from .types import ConstexprType
from .types import PointerType
from .types import ScalarType
from .types import ValueType
from .types import block_shape
from .types import element_type
from .types import emit_type
from .types import with_element_type


@dataclass(frozen=True, slots=True)
class Parameter:
    name: str
    type: ValueType
    kind: str


@dataclass(frozen=True, slots=True)
class Signature:
    parameters: tuple[Parameter, ...]


def lower_signature(
    definition: KernelDefinition[object, object],
    source: SourceUnit,
) -> Signature:
    signature = definition.signature
    if source.function.args.vararg is not None or source.function.args.kwarg is not None:
        raise FrontendError(
            "kernel entry cannot use *args or **kwargs",
            source.location(source.function),
        )
    parameters: list[Parameter] = []
    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Signature.empty:
            raise FrontendError(
                f"kernel parameter {name!r} cannot have a Python default",
                source.location(source.function),
            )
        annotation = parameter.annotation
        if isinstance(annotation, PtrSpec):
            parameters.append(
                Parameter(
                    name,
                    PointerType(ScalarType(annotation.dtype), annotation.address_space),
                    "pointer",
                )
            )
        elif isinstance(annotation, DType):
            parameters.append(Parameter(name, ScalarType(annotation), "scalar"))
        elif isinstance(annotation, ConstexprSpec):
            parameters.append(
                Parameter(name, ConstexprType(ScalarType(annotation.dtype)), "constexpr")
            )
        else:
            raise FrontendError(
                f"kernel parameter {name!r} requires W.ptr, W.dtype, or W.constexpr",
                source.location(source.function),
            )
    if signature.return_annotation not in (inspect.Signature.empty, None, type(None)):
        raise FrontendError(
            "Weft kernel entries cannot return a Python/SSA value",
            source.location(source.function),
        )
    return Signature(tuple(parameters))


class FrontendCompiler:
    def __init__(self, definition: KernelDefinition[object, object]) -> None:
        self.definition = definition
        self.source = SourceUnit.from_definition(definition)
        self.signature = lower_signature(definition, self.source)
        self.builder = IRBuilder()

    def lower(self) -> str:
        parameter_types = tuple(parameter.type for parameter in self.signature.parameters)
        parameter_names = tuple(parameter.name for parameter in self.signature.parameters)
        body = self.builder.region(parameter_types, parameter_names)
        environment = {
            parameter.name: value
            for parameter, value in zip(self.signature.parameters, body.arguments)
        }

        location = self.source.location(self.source.function)
        for parameter, argument in zip(self.signature.parameters, body.arguments):
            if parameter.kind != "constexpr":
                continue
            runtime_type = argument.type.value_type
            environment[parameter.name] = self.builder.emit(
                body,
                "weft_kernel.meta_value",
                location,
                operands=(argument,),
                result_types=(runtime_type,),
                result_names=(parameter.name,),
            )[0]

        lowerer = FunctionLowerer(
            self,
            body,
            environment,
            inside_loop=False,
        )
        lowerer.lower_statements(self.source.function.body)
        if not body.terminated:
            self.builder.emit(body, "weft_kernel.return", location)

        attributes = {
            "sym_name": json.dumps(self.definition.__name__),
            "grid_rank": f"{self.definition.grid_rank} : i64",
            "arg_names": _string_array(parameter.name for parameter in self.signature.parameters),
            "arg_kinds": _string_array(parameter.kind for parameter in self.signature.parameters),
            "source": json.dumps(location.format()),
        }
        kernel = Operation(
            name="weft_kernel.kernel",
            operands=(),
            results=(),
            attributes=tuple(attributes.items()),
            regions=(body,),
            location=location,
        )
        return self.builder.module(self.definition.__name__, kernel)


class FunctionLowerer:
    def __init__(
        self,
        compiler: FrontendCompiler,
        block: Region,
        environment: dict[str, Value],
        *,
        inside_loop: bool,
    ) -> None:
        self.compiler = compiler
        self.source = compiler.source
        self.builder = compiler.builder
        self.block = block
        self.environment = environment
        self.inside_loop = inside_loop

    def lower_statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            if self.block.terminated:
                raise FrontendError(
                    "statement appears after a block terminator",
                    self.source.location(statement),
                )
            self.lower_statement(statement)

    def lower_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise FrontendError(
                    "first Weft slice supports assignment to one local name",
                    self.source.location(statement),
                )
            value = self.lower_expression(statement.value)
            if value is None:
                raise FrontendError(
                    "effect-only intrinsic cannot be assigned",
                    self.source.location(statement),
                )
            self.environment[statement.targets[0].id] = value
            return
        if isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name):
                raise FrontendError(
                    "augmented assignment requires a local name",
                    self.source.location(statement),
                )
            if statement.target.id not in self.environment:
                raise FrontendError(
                    f"unknown local {statement.target.id!r}",
                    self.source.location(statement.target),
                )
            rhs = self.lower_expression(statement.value)
            if rhs is None:
                raise FrontendError("missing augmented-assignment value", self.source.location(statement))
            self.environment[statement.target.id] = self._lower_binary_values(
                self.environment[statement.target.id],
                rhs,
                statement.op,
                self.source.location(statement),
            )
            return
        if isinstance(statement, ast.Expr):
            result = self.lower_expression(statement.value)
            if result is not None:
                raise FrontendError(
                    "pure Weft value is unused",
                    self.source.location(statement),
                )
            return
        if isinstance(statement, ast.For):
            self._lower_for(statement)
            return
        if isinstance(statement, ast.Return):
            if self.inside_loop:
                raise FrontendError(
                    "return inside W.range is unsupported",
                    self.source.location(statement),
                )
            if statement.value is not None:
                raise FrontendError(
                    "Weft kernel cannot return a value",
                    self.source.location(statement),
                )
            self.builder.emit(
                self.block,
                "weft_kernel.return",
                self.source.location(statement),
            )
            return
        if isinstance(statement, ast.Pass):
            return
        raise FrontendError(
            f"unsupported Weft statement {type(statement).__name__}",
            self.source.location(statement),
        )

    def lower_expression(self, expression: ast.expr) -> Value | None:
        if isinstance(expression, ast.Name):
            value = self.environment.get(expression.id)
            if value is None:
                raise FrontendError(
                    f"unknown Weft local {expression.id!r}",
                    self.source.location(expression),
                )
            return value
        if isinstance(expression, ast.Constant):
            return self._constant(expression.value, self.source.location(expression))
        if isinstance(expression, ast.BinOp):
            lhs = self.lower_expression(expression.left)
            rhs = self.lower_expression(expression.right)
            if lhs is None or rhs is None:
                raise FrontendError("binary operands must be values", self.source.location(expression))
            return self._lower_binary_values(
                lhs,
                rhs,
                expression.op,
                self.source.location(expression),
            )
        if isinstance(expression, ast.UnaryOp):
            value = self.lower_expression(expression.operand)
            if value is None:
                raise FrontendError("unary operand must be a value", self.source.location(expression))
            if not isinstance(expression.op, ast.USub):
                raise FrontendError(
                    "first Weft slice supports only unary minus",
                    self.source.location(expression),
                )
            return self._unary(value, "neg", self.source.location(expression))
        if isinstance(expression, ast.Compare):
            if len(expression.ops) != 1 or len(expression.comparators) != 1:
                raise FrontendError(
                    "chained comparisons are unsupported",
                    self.source.location(expression),
                )
            lhs = self.lower_expression(expression.left)
            rhs = self.lower_expression(expression.comparators[0])
            if lhs is None or rhs is None:
                raise FrontendError("comparison operands must be values", self.source.location(expression))
            return self._compare(
                lhs,
                rhs,
                expression.ops[0],
                self.source.location(expression),
            )
        if isinstance(expression, ast.Call):
            return self._lower_call(expression)
        raise FrontendError(
            f"unsupported Weft expression {type(expression).__name__}",
            self.source.location(expression),
        )

    def _lower_call(self, call: ast.Call) -> Value | None:
        callee = self.source.resolve(call.func)
        if not isinstance(callee, Intrinsic):
            raise FrontendError(
                "Weft kernel calls must target a W.* intrinsic",
                self.source.location(call.func),
            )
        location = self.source.location(call)
        if callee.name == "task_id":
            _reject_keywords(call)
            if len(call.args) != 1:
                raise FrontendError("W.task_id expects one axis", location)
            axis = _literal_int(call.args[0], self.source)
            if axis < 0 or axis >= self.compiler.definition.grid_rank:
                raise FrontendError(
                    f"task axis {axis} is outside grid_rank "
                    f"{self.compiler.definition.grid_rank}",
                    location,
                )
            return self.builder.emit(
                self.block,
                "weft_kernel.task_id",
                location,
                result_types=(ScalarType(index),),
                attributes={"axis": f"{axis} : i64"},
            )[0]
        if callee.name == "arange":
            _reject_keywords(call)
            if len(call.args) != 2:
                raise FrontendError("W.arange expects start and extent", location)
            start = self._require_value(call.args[0])
            extent = self._require_value(call.args[1])
            _require_index(start, location)
            _require_index(extent, location)
            return self.builder.emit(
                self.block,
                "weft_kernel.arange",
                location,
                operands=(start, extent),
                result_types=(BlockType((-1,), ScalarType(index)),),
            )[0]
        if callee.name == "expand_dims":
            _reject_keywords(call)
            if len(call.args) != 2:
                raise FrontendError("W.expand_dims expects a block and axis", location)
            value = self._require_value(call.args[0])
            if not isinstance(value.type, BlockType):
                raise FrontendError("W.expand_dims input must be a block", location)
            axis = _literal_int(call.args[1], self.source)
            if axis < 0 or axis > len(value.type.shape):
                raise FrontendError(
                    f"expand axis {axis} is outside [0, {len(value.type.shape)}]",
                    location,
                )
            shape = value.type.shape[:axis] + (1,) + value.type.shape[axis:]
            return self.builder.emit(
                self.block,
                "weft_kernel.expand_dims",
                location,
                operands=(value,),
                result_types=(BlockType(shape, value.type.element_type),),
                attributes={"axis": f"{axis} : i64"},
            )[0]
        if callee.name in {"full_like", "zeros_like"}:
            _reject_keywords(call)
            if len(call.args) != 2:
                raise FrontendError(
                    f"W.{callee.name} expects a shape-like block and "
                    + ("scalar value" if callee.name == "full_like" else "W.dtype"),
                    location,
                )
            shape_like = self._require_value(call.args[0])
            if not isinstance(shape_like.type, BlockType):
                raise FrontendError(
                    f"W.{callee.name} shape source must be a block", location
                )
            if callee.name == "full_like":
                value = self._require_value(call.args[1])
                if not isinstance(value.type, ScalarType):
                    raise FrontendError("W.full_like value must be scalar", location)
            else:
                dtype = self.source.resolve(call.args[1])
                if not isinstance(dtype, DType):
                    raise FrontendError("W.zeros_like dtype must be a Weft dtype", location)
                value = self._zero(ScalarType(dtype), location)
            return self.builder.emit(
                self.block,
                "weft_kernel.splat",
                location,
                operands=(value, shape_like),
                result_types=(BlockType(shape_like.type.shape, value.type),),
            )[0]
        if callee.name == "load":
            pointer_node, keywords = _call_arguments(call, 1, {"mask", "other"}, self.source)
            pointer = self._require_value(pointer_node[0])
            mask = self._require_value(keywords["mask"])
            other = self._require_value(keywords["other"])
            result_type = _loaded_type(pointer.type, location)
            _require_mask(mask.type, pointer.type, location)
            _require_memory_value(other.type, result_type, location, "load other")
            return self.builder.emit(
                self.block,
                "weft_kernel.load",
                location,
                operands=(pointer, mask, other),
                result_types=(result_type,),
            )[0]
        if callee.name == "store":
            positional, keywords = _call_arguments(call, 2, {"mask"}, self.source)
            pointer = self._require_value(positional[0])
            value = self._require_value(positional[1])
            mask = self._require_value(keywords["mask"])
            expected = _loaded_type(pointer.type, location)
            _require_mask(mask.type, pointer.type, location)
            _require_memory_value(value.type, expected, location, "store value")
            self.builder.emit(
                self.block,
                "weft_kernel.store",
                location,
                operands=(pointer, value, mask),
            )
            return None
        if callee.name in {"contract", "dot"}:
            if callee.name == "contract":
                positional, keywords = _call_arguments(
                    call,
                    3,
                    {"lhs_axes", "rhs_axes", "ordered"},
                    self.source,
                    optional={"ordered"},
                )
                lhs = self._require_value(positional[0])
                rhs = self._require_value(positional[1])
                init = self._require_value(positional[2])
                if not isinstance(lhs.type, BlockType) or not isinstance(
                    rhs.type, BlockType
                ):
                    raise FrontendError("W.contract lhs and rhs must be blocks", location)
                lhs_axes = _literal_axes(
                    keywords["lhs_axes"], len(lhs.type.shape), "lhs", self.source
                )
                rhs_axes = _literal_axes(
                    keywords["rhs_axes"], len(rhs.type.shape), "rhs", self.source
                )
            else:
                if len(call.args) not in {2, 3}:
                    raise FrontendError(
                        "W.dot expects lhs, rhs[, accumulator]", location
                    )
                positional, keywords = _call_arguments(
                    call,
                    len(call.args),
                    {"ordered"},
                    self.source,
                    optional={"ordered"},
                )
                lhs = self._require_value(positional[0])
                rhs = self._require_value(positional[1])
                if not isinstance(lhs.type, BlockType) or not isinstance(
                    rhs.type, BlockType
                ):
                    raise FrontendError("W.dot lhs and rhs must be blocks", location)
                if len(lhs.type.shape) not in {1, 2} or len(rhs.type.shape) not in {
                    1,
                    2,
                }:
                    raise FrontendError(
                        "W.dot supports rank-one or rank-two blocks; use "
                        "W.contract for general contractions",
                        location,
                    )
                lhs_axes = (len(lhs.type.shape) - 1,)
                rhs_axes = (0,)
                lhs_element = element_type(lhs.type)
                if not isinstance(lhs_element, ScalarType):
                    raise FrontendError("W.dot requires scalar block elements", location)
                init = (
                    self._require_value(positional[2])
                    if len(positional) == 3
                    else self._zero(lhs_element, location)
                )
            ordered = (
                _literal_bool(keywords["ordered"], self.source)
                if "ordered" in keywords
                else False
            )
            return self._contract(
                lhs,
                rhs,
                init,
                lhs_axes,
                rhs_axes,
                ordered,
                location,
            )
        if callee.name in {"sum", "max", "min", "reduce"}:
            if callee.name == "sum":
                positional, keywords = _call_arguments(
                    call,
                    1,
                    {"axis", "ordered"},
                    self.source,
                    optional={"ordered"},
                )
                value = self._require_value(positional[0])
                value_element = element_type(value.type)
                if not isinstance(value_element, ScalarType):
                    raise FrontendError("W.sum requires scalar block elements", location)
                init = self._zero(value_element, location)
                kind = "sum"
            elif callee.name in {"max", "min"}:
                positional, keywords = _call_arguments(
                    call,
                    2,
                    {"axis", "ordered"},
                    self.source,
                    optional={"ordered"},
                )
                value = self._require_value(positional[0])
                init = self._require_value(positional[1])
                kind = callee.name
            else:
                positional, keywords = _call_arguments(
                    call,
                    2,
                    {"axis", "kind", "ordered"},
                    self.source,
                    optional={"ordered"},
                )
                value = self._require_value(positional[0])
                init = self._require_value(positional[1])
                kind = _literal_string(keywords["kind"], self.source)
            axis = _literal_int(keywords["axis"], self.source)
            ordered = (
                _literal_bool(keywords["ordered"], self.source)
                if "ordered" in keywords
                else False
            )
            return self._reduce(value, init, axis, kind, ordered, location)
        if callee.name == "cast":
            _reject_keywords(call)
            if len(call.args) != 2:
                raise FrontendError("W.cast expects a value and W.dtype", location)
            value = self._require_value(call.args[0])
            if not _is_numeric_value(value.type, allow_bool=True):
                raise FrontendError(
                    "W.cast input must be scalar or numeric block data",
                    location,
                )
            dtype = self.source.resolve(call.args[1])
            if not isinstance(dtype, DType):
                raise FrontendError("W.cast target must be a Weft dtype", location)
            result_type = with_element_type(value.type, ScalarType(dtype))
            return self.builder.emit(
                self.block,
                "weft_kernel.cast",
                location,
                operands=(value,),
                result_types=(result_type,),
            )[0]
        if callee.name in {"exp", "exp2", "log", "rsqrt"}:
            _reject_keywords(call)
            if len(call.args) != 1:
                raise FrontendError(f"W.{callee.name} expects one value", location)
            return self._unary(self._require_value(call.args[0]), callee.name, location)
        if callee.name in {"maximum", "minimum"}:
            _reject_keywords(call)
            if len(call.args) != 2:
                raise FrontendError(f"W.{callee.name} expects two values", location)
            kind = "max" if callee.name == "maximum" else "min"
            return self._binary(
                self._require_value(call.args[0]),
                self._require_value(call.args[1]),
                kind,
                location,
            )
        if callee.name == "range":
            raise FrontendError("W.range is valid only in a for statement", location)
        raise FrontendError(f"unsupported intrinsic W.{callee.name}", location)

    def _lower_for(self, statement: ast.For) -> None:
        if statement.orelse:
            raise FrontendError("W.range loops cannot have else", self.source.location(statement))
        if not isinstance(statement.target, ast.Name):
            raise FrontendError("W.range target must be one local name", self.source.location(statement))
        if not isinstance(statement.iter, ast.Call):
            raise FrontendError("for loops must iterate over W.range", self.source.location(statement))
        callee = self.source.resolve(statement.iter.func)
        if not isinstance(callee, Intrinsic) or callee.name != "range":
            raise FrontendError("for loops must iterate over W.range", self.source.location(statement.iter))
        _reject_keywords(statement.iter)
        bounds = tuple(self._require_value(argument) for argument in statement.iter.args)
        location = self.source.location(statement)
        if len(bounds) == 1:
            lower, upper, step = self._index_constant(0, location), bounds[0], self._index_constant(1, location)
        elif len(bounds) == 2:
            lower, upper = bounds
            step = self._index_constant(1, location)
        elif len(bounds) == 3:
            lower, upper, step = bounds
        else:
            raise FrontendError("W.range expects stop or start, stop[, step]", location)
        for bound in (lower, upper, step):
            _require_index(bound, location)

        assigned = _assigned_names(statement.body)
        carried_names = tuple(
            name
            for name in assigned
            if name in self.environment and name != statement.target.id
        )
        init_values = tuple(self.environment[name] for name in carried_names)
        argument_types = (ScalarType(index),) + tuple(value.type for value in init_values)
        argument_names = (statement.target.id,) + carried_names
        region = self.builder.region(argument_types, argument_names)

        nested_environment = dict(self.environment)
        nested_environment[statement.target.id] = region.arguments[0]
        for name, argument in zip(carried_names, region.arguments[1:]):
            nested_environment[name] = argument
        nested = FunctionLowerer(
            self.compiler,
            region,
            nested_environment,
            inside_loop=True,
        )
        nested.lower_statements(statement.body)
        if region.terminated:
            raise FrontendError(
                "W.range body may terminate only through its implicit yield",
                location,
            )
        yielded = tuple(nested.environment[name] for name in carried_names)
        self.builder.emit(
            region,
            "weft_kernel.yield",
            location,
            operands=yielded,
        )
        results = self.builder.emit(
            self.block,
            "weft_kernel.for",
            location,
            operands=(lower, upper, step, *init_values),
            result_types=tuple(value.type for value in init_values),
            regions=(region,),
            result_names=carried_names,
        )
        for name, result in zip(carried_names, results):
            self.environment[name] = result

    def _require_value(self, expression: ast.expr) -> Value:
        value = self.lower_expression(expression)
        if value is None:
            raise FrontendError("expected a Weft value", self.source.location(expression))
        return value

    def _lower_binary_values(
        self,
        lhs: Value,
        rhs: Value,
        operator: ast.operator,
        location: SourceLocation,
    ) -> Value:
        if isinstance(operator, ast.Add) and (_is_pointer(lhs.type) or _is_pointer(rhs.type)):
            base, offset = (lhs, rhs) if _is_pointer(lhs.type) else (rhs, lhs)
            return self._ptr_add(base, offset, location)
        kinds: tuple[tuple[type[ast.operator], str], ...] = (
            (ast.Add, "add"),
            (ast.Sub, "sub"),
            (ast.Mult, "mul"),
            (ast.Div, "div"),
            (ast.Mod, "mod"),
            (ast.BitAnd, "and"),
            (ast.BitOr, "or"),
            (ast.BitXor, "xor"),
        )
        for operator_type, kind in kinds:
            if isinstance(operator, operator_type):
                return self._binary(lhs, rhs, kind, location)
        if isinstance(operator, ast.FloorDiv):
            raise FrontendError(
                "Python // has no canonical first-slice Weft semantics",
                location,
            )
        raise FrontendError(
            f"unsupported binary operator {type(operator).__name__}",
            location,
        )

    def _contract(
        self,
        lhs: Value,
        rhs: Value,
        init: Value,
        lhs_axes: tuple[int, ...],
        rhs_axes: tuple[int, ...],
        ordered: bool,
        location: SourceLocation,
    ) -> Value:
        if not isinstance(lhs.type, BlockType) or not isinstance(rhs.type, BlockType):
            raise FrontendError("contraction lhs and rhs must be blocks", location)
        if not lhs_axes or len(lhs_axes) != len(rhs_axes):
            raise FrontendError(
                "contraction axes must be non-empty and paired", location
            )
        if len(set(lhs_axes)) != len(lhs_axes) or len(set(rhs_axes)) != len(rhs_axes):
            raise FrontendError("contraction axes must be unique", location)
        for lhs_axis, rhs_axis in zip(lhs_axes, rhs_axes):
            if lhs.type.shape[lhs_axis] != rhs.type.shape[rhs_axis]:
                raise FrontendError(
                    "paired contraction dimensions must have identical extents",
                    location,
                )

        lhs_element = element_type(lhs.type)
        rhs_element = element_type(rhs.type)
        if (
            lhs_element != rhs_element
            or not isinstance(lhs_element, ScalarType)
            or lhs_element.dtype.category is DTypeCategory.BOOL
        ):
            raise FrontendError(
                "contraction inputs require equal non-boolean numeric element types",
                location,
            )
        result_shape = tuple(
            dimension
            for axis, dimension in enumerate(lhs.type.shape)
            if axis not in lhs_axes
        ) + tuple(
            dimension
            for axis, dimension in enumerate(rhs.type.shape)
            if axis not in rhs_axes
        )
        init_element = element_type(init.type)
        signed_i8_to_i32 = (
            isinstance(lhs_element, ScalarType)
            and isinstance(rhs_element, ScalarType)
            and isinstance(init_element, ScalarType)
            and lhs_element.dtype.category is DTypeCategory.INTEGER
            and rhs_element.dtype.category is DTypeCategory.INTEGER
            and init_element.dtype.category is DTypeCategory.INTEGER
            and lhs_element.dtype.bits == 8
            and rhs_element.dtype.bits == 8
            and init_element.dtype.bits == 32
            and lhs_element.dtype.signedness == "signed"
            and rhs_element.dtype.signedness == "signed"
            and init_element.dtype.signedness == "signed"
        )
        if init_element != lhs_element and not signed_i8_to_i32:
            raise FrontendError(
                "contraction accumulator must match its inputs or use the "
                "validated signed i8 x signed i8 to signed i32 widening form",
                location,
            )
        if isinstance(init.type, BlockType):
            if not result_shape or init.type.shape != result_shape:
                raise FrontendError(
                    "blocked contraction accumulator must match result shape",
                    location,
                )
        elif not isinstance(init.type, ScalarType):
            raise FrontendError(
                "contraction accumulator must be scalar or result-shaped block",
                location,
            )
        result_type: ValueType = (
            BlockType(result_shape, init_element) if result_shape else init_element
        )
        return self.builder.emit(
            self.block,
            "weft_kernel.contract",
            location,
            operands=(lhs, rhs, init),
            result_types=(result_type,),
            attributes={
                "lhs_axes": _i64_array(lhs_axes),
                "rhs_axes": _i64_array(rhs_axes),
                "ordered": "true" if ordered else "false",
            },
        )[0]

    def _reduce(
        self,
        value: Value,
        init: Value,
        axis: int,
        kind: str,
        ordered: bool,
        location: SourceLocation,
    ) -> Value:
        if not isinstance(value.type, BlockType):
            raise FrontendError("reduction input must be a block", location)
        if not _is_numeric_value(value.type, allow_bool=False):
            raise FrontendError("reduction requires numeric block elements", location)
        if kind not in {"sum", "max", "min"}:
            raise FrontendError("reduction kind must be 'sum', 'max', or 'min'", location)
        if axis < 0 or axis >= len(value.type.shape):
            raise FrontendError(
                f"reduction axis {axis} is outside block rank {len(value.type.shape)}",
                location,
            )
        result_element = element_type(value.type)
        if not isinstance(result_element, ScalarType):
            raise FrontendError("reduction requires scalar block elements", location)
        if init.type != result_element:
            raise FrontendError(
                "reduction init must be a scalar matching the input element type",
                location,
            )
        projected_shape = value.type.shape[:axis] + value.type.shape[axis + 1 :]
        result_type: ValueType = (
            BlockType(projected_shape, result_element)
            if projected_shape
            else result_element
        )
        return self.builder.emit(
            self.block,
            "weft_kernel.reduce",
            location,
            operands=(value, init),
            result_types=(result_type,),
            attributes={
                "axis": f"{axis} : i64",
                "kind": json.dumps(kind),
                "ordered": "true" if ordered else "false",
            },
        )[0]

    def _binary(self, lhs: Value, rhs: Value, kind: str, location: SourceLocation) -> Value:
        result_type = _broadcast_type(lhs.type, rhs.type, location)
        result_element = element_type(result_type)
        if not isinstance(result_element, ScalarType):
            raise FrontendError("pointwise arithmetic requires scalar data elements", location)
        category = result_element.dtype.category
        if kind in {"and", "or", "xor"}:
            if category not in {
                DTypeCategory.BOOL,
                DTypeCategory.INTEGER,
                DTypeCategory.INDEX,
            }:
                raise FrontendError(
                    f"pointwise {kind} requires boolean/integer/index operands",
                    location,
                )
        elif category is DTypeCategory.BOOL:
            raise FrontendError(f"pointwise {kind} does not accept boolean operands", location)
        elif kind == "div" and category not in {
            DTypeCategory.FLOAT,
            DTypeCategory.BFLOAT,
        }:
            raise FrontendError("/ is defined only for floating Weft values", location)
        elif kind == "mod" and category not in {
            DTypeCategory.INTEGER,
            DTypeCategory.INDEX,
        }:
            raise FrontendError("% is defined only for integer/index Weft values", location)
        return self.builder.emit(
            self.block,
            "weft_kernel.binary",
            location,
            operands=(lhs, rhs),
            result_types=(result_type,),
            attributes={"kind": json.dumps(kind)},
        )[0]

    def _compare(
        self,
        lhs: Value,
        rhs: Value,
        operator: ast.cmpop,
        location: SourceLocation,
    ) -> Value:
        predicates: tuple[tuple[type[ast.cmpop], str], ...] = (
            (ast.Eq, "eq"),
            (ast.NotEq, "ne"),
            (ast.Lt, "lt"),
            (ast.LtE, "le"),
            (ast.Gt, "gt"),
            (ast.GtE, "ge"),
        )
        predicate = next(
            (name for kind, name in predicates if isinstance(operator, kind)),
            None,
        )
        if predicate is None:
            raise FrontendError("unsupported comparison", location)
        broadcast = _broadcast_type(lhs.type, rhs.type, location)
        if not _is_numeric_value(broadcast, allow_bool=True):
            raise FrontendError("comparison operands must contain scalar data", location)
        if (
            element_type(broadcast) == ScalarType(bool_dtype)
            and not isinstance(operator, (ast.Eq, ast.NotEq))
        ):
            raise FrontendError("boolean values support only == and !=", location)
        result_type = with_element_type(broadcast, ScalarType(bool_dtype))
        return self.builder.emit(
            self.block,
            "weft_kernel.compare",
            location,
            operands=(lhs, rhs),
            result_types=(result_type,),
            attributes={"predicate": json.dumps(predicate)},
        )[0]

    def _unary(self, value: Value, kind: str, location: SourceLocation) -> Value:
        value_element = element_type(value.type)
        if not isinstance(value_element, ScalarType):
            raise FrontendError("unary operation requires scalar data elements", location)
        category = value_element.dtype.category
        if kind in {"exp", "exp2", "log", "rsqrt"} and category not in {
            DTypeCategory.FLOAT,
            DTypeCategory.BFLOAT,
        }:
            raise FrontendError(f"W.{kind} requires floating values", location)
        if kind == "neg" and category is DTypeCategory.BOOL:
            raise FrontendError("unary minus does not accept boolean values", location)
        return self.builder.emit(
            self.block,
            "weft_kernel.unary",
            location,
            operands=(value,),
            result_types=(value.type,),
            attributes={"kind": json.dumps(kind)},
        )[0]

    def _ptr_add(self, base: Value, offset: Value, location: SourceLocation) -> Value:
        if not _is_pointer(base.type):
            raise FrontendError("pointer addition requires one pointer operand", location)
        if not _is_index_value(offset.type):
            raise FrontendError(
                "pointer offset must be index or a block of index values",
                location,
            )
        if (
            isinstance(base.type, BlockType)
            and isinstance(offset.type, BlockType)
            and _broadcast_shape(base.type.shape, offset.type.shape) is None
        ):
            raise FrontendError(
                "blocked pointer and offset shapes are not broadcast-compatible",
                location,
            )
        if isinstance(base.type, BlockType) and isinstance(offset.type, BlockType):
            shape = _broadcast_shape(base.type.shape, offset.type.shape)
            assert shape is not None
            result_type = BlockType(shape, base.type.element_type)
        elif isinstance(base.type, BlockType):
            result_type = base.type
        elif isinstance(offset.type, BlockType):
            result_type = BlockType(offset.type.shape, base.type)
        else:
            result_type = base.type
        return self.builder.emit(
            self.block,
            "weft_kernel.ptr_add",
            location,
            operands=(base, offset),
            result_types=(result_type,),
        )[0]

    def _constant(self, value: object, location: SourceLocation) -> Value:
        if isinstance(value, bool):
            value_type = ScalarType(bool_dtype)
            spelling = "true" if value else "false"
        elif isinstance(value, int):
            value_type = ScalarType(index)
            spelling = f"{value} : index"
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise FrontendError("non-finite float literals are unsupported", location)
            value_type = ScalarType(f32)
            spelling = f"{repr(value)} : f32"
        else:
            raise FrontendError(
                f"unsupported literal {type(value).__name__}",
                location,
            )
        return self.builder.emit(
            self.block,
            "weft_kernel.constant",
            location,
            result_types=(value_type,),
            attributes={"value": spelling},
        )[0]

    def _index_constant(self, value: int, location: SourceLocation) -> Value:
        return self._constant(value, location)

    def _zero(self, value_type: ScalarType, location: SourceLocation) -> Value:
        category = value_type.dtype.category
        value = 0.0 if category in {DTypeCategory.FLOAT, DTypeCategory.BFLOAT} else 0
        constant = self._constant(value, location)
        if constant.type == value_type:
            return constant
        return self.builder.emit(
            self.block,
            "weft_kernel.cast",
            location,
            operands=(constant,),
            result_types=(value_type,),
        )[0]


def lower_to_mlir(definition: KernelDefinition[object, object]) -> str:
    if not isinstance(definition, KernelDefinition):
        raise TypeError("lower_to_mlir expects an @weft.kernel definition")
    return FrontendCompiler(definition).lower()


def _string_array(values: object) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def _literal_int(node: ast.AST, source: SourceUnit) -> int:
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool) or not isinstance(node.value, int):
        raise FrontendError("expected an integer literal", source.location(node))
    return node.value


def _literal_bool(node: ast.AST, source: SourceUnit) -> bool:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, bool):
        raise FrontendError("expected a boolean literal", source.location(node))
    return node.value


def _literal_string(node: ast.AST, source: SourceUnit) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise FrontendError("expected a string literal", source.location(node))
    return node.value


def _literal_axes(
    node: ast.AST,
    rank: int,
    role: str,
    source: SourceUnit,
) -> tuple[int, ...]:
    elements = node.elts if isinstance(node, (ast.Tuple, ast.List)) else (node,)
    axes: list[int] = []
    for element in elements:
        if isinstance(element, ast.Constant) and not isinstance(element.value, bool) and isinstance(
            element.value, int
        ):
            axis = element.value
        elif (
            isinstance(element, ast.UnaryOp)
            and isinstance(element.op, ast.USub)
            and isinstance(element.operand, ast.Constant)
            and not isinstance(element.operand.value, bool)
            and isinstance(element.operand.value, int)
        ):
            axis = -element.operand.value
        else:
            raise FrontendError(
                f"{role}_axes must contain only integer literals",
                source.location(element),
            )
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise FrontendError(
                f"{role} contraction axis {axis} is outside rank {rank}",
                source.location(element),
            )
        axes.append(axis)
    return tuple(axes)


def _i64_array(values: tuple[int, ...]) -> str:
    return "array<i64: " + ", ".join(str(value) for value in values) + ">"


def _reject_keywords(call: ast.Call) -> None:
    if call.keywords:
        raise FrontendError("this intrinsic does not accept keyword arguments")


def _call_arguments(
    call: ast.Call,
    positional_count: int,
    keyword_names: set[str],
    source: SourceUnit,
    *,
    optional: set[str] | None = None,
) -> tuple[tuple[ast.expr, ...], dict[str, ast.expr]]:
    if len(call.args) != positional_count:
        raise FrontendError(
            f"intrinsic expects {positional_count} positional argument(s)",
            source.location(call),
        )
    values: dict[str, ast.expr] = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in keyword_names:
            raise FrontendError("unsupported intrinsic keyword", source.location(keyword))
        if keyword.arg in values:
            raise FrontendError("duplicate intrinsic keyword", source.location(keyword))
        values[keyword.arg] = keyword.value
    missing = keyword_names - set(values) - set(optional or ())
    if missing:
        raise FrontendError(
            f"missing intrinsic keyword {sorted(missing)[0]!r}",
            source.location(call),
        )
    return tuple(call.args), values


def _require_index(value: Value, location: SourceLocation) -> None:
    if value.type != ScalarType(index):
        raise FrontendError(
            f"expected index, got {emit_type(value.type)}",
            location,
        )


def _is_pointer(value_type: ValueType) -> bool:
    if isinstance(value_type, PointerType):
        return True
    return isinstance(value_type, BlockType) and isinstance(value_type.element_type, PointerType)


def _loaded_type(pointer_type: ValueType, location: SourceLocation) -> ValueType:
    if isinstance(pointer_type, PointerType):
        return pointer_type.element_type
    if isinstance(pointer_type, BlockType) and isinstance(pointer_type.element_type, PointerType):
        return BlockType(pointer_type.shape, pointer_type.element_type.element_type)
    raise FrontendError("W.load expects a pointer or block of pointers", location)


def _is_index_value(value_type: ValueType) -> bool:
    return element_type(value_type) == ScalarType(index)


def _is_numeric_value(value_type: ValueType, *, allow_bool: bool) -> bool:
    value_element = element_type(value_type)
    if not isinstance(value_element, ScalarType):
        return False
    if value_element.dtype.category is DTypeCategory.BOOL:
        return allow_bool
    return value_element.dtype.category in {
        DTypeCategory.INTEGER,
        DTypeCategory.INDEX,
        DTypeCategory.FLOAT,
        DTypeCategory.BFLOAT,
    }


def _broadcast_compatible(lhs: ValueType, rhs: ValueType) -> bool:
    lhs_shape = block_shape(lhs)
    rhs_shape = block_shape(rhs)
    return (
        lhs_shape is None
        or rhs_shape is None
        or _broadcast_shape(lhs_shape, rhs_shape) is not None
    )


def _broadcasts_into_memory_shape(value: ValueType, pointer: ValueType) -> bool:
    value_shape = block_shape(value)
    pointer_shape = block_shape(pointer)
    if pointer_shape is None:
        return value_shape is None
    if value_shape is None:
        return True
    return _broadcast_shape(value_shape, pointer_shape) == pointer_shape


def _require_mask(
    mask_type: ValueType,
    pointer_type: ValueType,
    location: SourceLocation,
) -> None:
    if element_type(mask_type) != ScalarType(bool_dtype):
        raise FrontendError("memory mask must be bool or a block of bool", location)
    if not _broadcasts_into_memory_shape(mask_type, pointer_type):
        raise FrontendError(
            "memory mask must broadcast exactly into the pointer footprint",
            location,
        )


def _require_memory_value(
    actual: ValueType,
    expected: ValueType,
    location: SourceLocation,
    role: str,
) -> None:
    if element_type(actual) != element_type(expected):
        raise FrontendError(
            f"{role} element type must be {emit_type(element_type(expected))}",
            location,
        )
    if not _broadcasts_into_memory_shape(actual, expected):
        raise FrontendError(
            f"{role} must broadcast exactly into the memory footprint",
            location,
        )


def _broadcast_type(lhs: ValueType, rhs: ValueType, location: SourceLocation) -> ValueType:
    lhs_shape = block_shape(lhs)
    rhs_shape = block_shape(rhs)
    shape: tuple[int, ...] | None
    if lhs_shape is not None and rhs_shape is not None:
        shape = _broadcast_shape(lhs_shape, rhs_shape)
        if shape is None:
            raise FrontendError(
                "pointwise block shapes are not axis-wise broadcast-compatible",
                location,
            )
    else:
        shape = lhs_shape if lhs_shape is not None else rhs_shape
    lhs_element = element_type(lhs)
    rhs_element = element_type(rhs)
    if lhs_element != rhs_element:
        raise FrontendError(
            f"pointwise element types differ: {emit_type(lhs_element)} and {emit_type(rhs_element)}",
            location,
        )
    return BlockType(shape, lhs_element) if shape is not None else lhs_element


def _broadcast_shape(
    lhs: tuple[int, ...], rhs: tuple[int, ...]
) -> tuple[int, ...] | None:
    if len(lhs) != len(rhs):
        return None
    result: list[int] = []
    for lhs_dimension, rhs_dimension in zip(lhs, rhs):
        if lhs_dimension == rhs_dimension:
            result.append(lhs_dimension)
        elif lhs_dimension == 1:
            result.append(rhs_dimension)
        elif rhs_dimension == 1:
            result.append(lhs_dimension)
        else:
            return None
    return tuple(result)


def _assigned_names(statements: list[ast.stmt]) -> tuple[str, ...]:
    names: list[str] = []

    class Collector(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store) and node.id not in names:
                names.append(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    collector = Collector()
    for statement in statements:
        collector.visit(statement)
    return tuple(names)
