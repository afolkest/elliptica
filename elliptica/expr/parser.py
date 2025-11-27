"""Parse and validate expression strings."""

import ast
from .errors import ParseError, ValidationError, LimitExceededError
from .functions import FUNCTIONS, CONSTANTS


ALLOWED_BINOPS = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow}
ALLOWED_UNARYOPS = {ast.USub}


class ExprValidator(ast.NodeVisitor):
    """Validates that an AST only contains allowed constructs."""

    def __init__(self, max_depth: int = 20, max_nodes: int = 100):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.depth = 0
        self.node_count = 0
        self.variables: set[str] = set()
        self.functions_used: set[str] = set()

    def visit(self, node):
        self.node_count += 1
        if self.node_count > self.max_nodes:
            raise LimitExceededError(f"Expression exceeds {self.max_nodes} nodes")

        self.depth += 1
        if self.depth > self.max_depth:
            raise LimitExceededError(f"Expression exceeds depth {self.max_depth}")

        result = super().visit(node)
        self.depth -= 1
        return result

    def visit_Expression(self, node):
        self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValidationError(
                f"Only numeric constants allowed, got {type(node.value).__name__}"
            )

    def visit_Name(self, node):
        if node.id not in CONSTANTS:
            self.variables.add(node.id)

    def visit_BinOp(self, node):
        if type(node.op) not in ALLOWED_BINOPS:
            raise ValidationError(f"Operator {type(node.op).__name__} not allowed")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        if type(node.op) not in ALLOWED_UNARYOPS:
            raise ValidationError(
                f"Unary operator {type(node.op).__name__} not allowed"
            )
        self.visit(node.operand)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValidationError("Only direct function calls allowed")

        if node.keywords:
            raise ValidationError("Keyword arguments not allowed")

        func_name = node.func.id
        if func_name not in FUNCTIONS:
            raise ValidationError(f"Unknown function: {func_name}")

        self.functions_used.add(func_name)

        expected = FUNCTIONS[func_name][2]
        got = len(node.args)
        if got != expected:
            raise ValidationError(f"{func_name} expects {expected} args, got {got}")

        for arg in node.args:
            self.visit(arg)

    def generic_visit(self, node):
        raise ValidationError(f"Construct not allowed: {type(node).__name__}")


def parse(
    expr: str, max_depth: int = 20, max_nodes: int = 100
) -> tuple[ast.AST, set[str]]:
    """Parse and validate an expression string.

    Returns:
        (ast, variables): The AST and set of variable names referenced
    """
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ParseError(f"Syntax error: {e}")

    validator = ExprValidator(max_depth, max_nodes)
    validator.visit(tree)

    return tree, validator.variables
