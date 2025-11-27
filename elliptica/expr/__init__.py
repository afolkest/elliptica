"""Expression parsing and compilation for field computations.

Parse Python-like math expressions and compile them to efficient callables
that work with numpy arrays or torch tensors.

Example:
    from flowcol.expr import compile_expression

    # Parse and compile
    fn = compile_expression("lic * 0.7 + sin(e_mag) * 0.3")

    # Evaluate with field arrays
    result = fn({'lic': lic_array, 'e_mag': e_mag_array})
"""

from .parser import parse
from .compiler import compile_expr
from .errors import (
    ExprError,
    ParseError,
    ValidationError,
    UnknownVariableError,
    LimitExceededError,
)
from .functions import FUNCTIONS, CONSTANTS


def compile_expression(
    expr: str,
    max_depth: int = 20,
    max_nodes: int = 100,
) -> callable:
    """Parse and compile an expression string.

    Args:
        expr: Expression string like "lic * 0.7 + e_mag * 0.3"
        max_depth: Maximum AST depth (default 20)
        max_nodes: Maximum AST nodes (default 100)

    Returns:
        Callable that takes dict[str, Array] and returns Array

    Raises:
        ParseError: If expression has syntax errors
        ValidationError: If expression uses disallowed constructs
        LimitExceededError: If expression exceeds safety limits

    Example:
        fn = compile_expression("normalize(lic) * 0.5 + 0.5")
        result = fn({'lic': lic_array})
    """
    tree, variables = parse(expr, max_depth, max_nodes)
    return compile_expr(tree, variables)


def get_variables(expr: str) -> set[str]:
    """Get the set of variable names referenced in an expression.

    Useful for knowing what fields need to be provided.

    Args:
        expr: Expression string

    Returns:
        Set of variable names (excluding built-in constants)
    """
    _, variables = parse(expr)
    return variables


def list_functions() -> dict[str, int]:
    """List available built-in functions and their argument counts."""
    return {name: info[2] for name, info in FUNCTIONS.items()}


def list_constants() -> dict[str, float]:
    """List available built-in constants."""
    return dict(CONSTANTS)


__all__ = [
    'compile_expression',
    'get_variables',
    'list_functions',
    'list_constants',
    # Errors
    'ExprError',
    'ParseError',
    'ValidationError',
    'UnknownVariableError',
    'LimitExceededError',
]
