"""Compile expression AST to callable."""

import ast
import numpy as np
from typing import Any, Callable
from .functions import CONSTANTS, get_function
from .errors import UnknownVariableError

Array = Any


def _is_torch(x) -> bool:
    return type(x).__module__.startswith('torch')


class ExprEvaluator(ast.NodeVisitor):
    """Evaluates an expression AST with given variable bindings."""

    def __init__(self, variables: dict[str, Array], use_torch: bool):
        self.variables = variables
        self.use_torch = use_torch
        self._torch = None

    def _get_torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        return node.value

    def visit_Name(self, node):
        name = node.id
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name not in self.variables:
            raise UnknownVariableError(f"Unknown variable: {name}")
        return self.variables[name]

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.Div):
            return left / right
        elif isinstance(op, ast.Pow):
            if self.use_torch:
                return self._get_torch().pow(left, right)
            return np.power(left, right)
        else:
            raise RuntimeError(f"Unhandled binary operator: {type(op).__name__}")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        else:
            raise RuntimeError(f"Unhandled unary operator: {type(node.op).__name__}")

    def visit_Call(self, node):
        func_name = node.func.id
        args = [self.visit(arg) for arg in node.args]
        func = get_function(func_name, self.use_torch)
        return func(*args)


def compile_expr(
    tree: ast.AST, variables: set[str]
) -> Callable[[dict[str, Array]], Array]:
    """Compile an expression AST to a callable.

    Args:
        tree: Validated AST from parser.parse()
        variables: Set of variable names the expression references

    Returns:
        Callable that takes dict of arrays and returns result array
    """

    def evaluate(bindings: dict[str, Array]) -> Array:
        missing = variables - set(bindings.keys())
        if missing:
            raise UnknownVariableError(f"Missing variables: {missing}")

        # Detect backend from first array
        first_array = next((v for v in bindings.values() if hasattr(v, 'shape')), None)
        use_torch = first_array is not None and _is_torch(first_array)

        evaluator = ExprEvaluator(bindings, use_torch)
        return evaluator.visit(tree)

    return evaluate
