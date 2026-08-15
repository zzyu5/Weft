from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass

from weft.api import KernelDefinition
from weft.diagnostics import FrontendError
from weft.diagnostics import SourceLocation
from weft.language import HelperDefinition


@dataclass(frozen=True, slots=True)
class SourceUnit:
    definition: KernelDefinition[object, object]
    function: ast.FunctionDef
    filename: str
    first_line: int
    bindings: dict[str, object]

    @classmethod
    def from_definition(
        cls, definition: KernelDefinition[object, object]
    ) -> SourceUnit:
        source_text = definition.source_text()
        tree = ast.parse(source_text, filename=definition.source.filename)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        location = SourceLocation(
            definition.source.filename, definition.source.first_line, 0
        )
        if len(functions) != 1:
            raise FrontendError(
                "a Weft DSL kernel must contain exactly one synchronous function", location
            )
        function = functions[0]
        if function.name != definition.__name__:
            raise FrontendError(
                "decorated function name does not match the captured DSL kernel", location
            )
        closure = inspect.getclosurevars(definition.python_function)
        bindings = dict(definition.python_function.__globals__)
        bindings.update(closure.globals)
        bindings.update(closure.nonlocals)
        bindings.update(closure.builtins)
        return cls(
            definition,
            function,
            definition.source.filename,
            definition.source.first_line,
            bindings,
        )

    def location(self, node: ast.AST) -> SourceLocation:
        return SourceLocation(
            self.filename,
            self.first_line + getattr(node, "lineno", 1) - 1,
            getattr(node, "col_offset", 0),
        )

    def resolve(self, node: ast.AST) -> object:
        if isinstance(node, ast.Name):
            if node.id not in self.bindings:
                raise FrontendError(
                    f"unknown Python binding {node.id!r}", self.location(node)
                )
            return self.bindings[node.id]
        if isinstance(node, ast.Attribute):
            owner = self.resolve(node.value)
            try:
                return getattr(owner, node.attr)
            except AttributeError as error:
                raise FrontendError(
                    f"Python binding has no attribute {node.attr!r}",
                    self.location(node),
                ) from error
        raise FrontendError("expected a static Python binding", self.location(node))


@dataclass(frozen=True, slots=True)
class HelperSource:
    definition: HelperDefinition
    function: ast.FunctionDef
    filename: str
    first_line: int
    bindings: dict[str, object]

    @classmethod
    def from_definition(cls, definition: HelperDefinition) -> HelperSource:
        function = definition.python_function
        try:
            lines, first_line = inspect.getsourcelines(function)
        except (OSError, TypeError) as error:
            raise FrontendError(f"cannot inspect helper {definition.__name__}") from error
        filename = inspect.getsourcefile(function) or function.__code__.co_filename
        tree = ast.parse(textwrap.dedent("".join(lines)), filename=filename)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            raise FrontendError("a DSL helper must contain exactly one function")
        closure = inspect.getclosurevars(function)
        bindings = dict(function.__globals__)
        bindings.update(closure.globals)
        bindings.update(closure.nonlocals)
        bindings.update(closure.builtins)
        return cls(definition, functions[0], filename, first_line, bindings)

    def location(self, node: ast.AST) -> SourceLocation:
        return SourceLocation(
            self.filename,
            self.first_line + getattr(node, "lineno", 1) - 1,
            getattr(node, "col_offset", 0),
        )
