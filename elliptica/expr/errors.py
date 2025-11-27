"""Expression parsing and evaluation errors."""


class ExprError(Exception):
    """Base class for expression errors."""
    pass


class ParseError(ExprError):
    """Failed to parse expression."""
    pass


class ValidationError(ExprError):
    """Expression contains disallowed constructs."""
    pass


class UnknownVariableError(ExprError):
    """Reference to unknown variable."""
    pass


class UnknownFunctionError(ExprError):
    """Call to unknown function."""
    pass


class LimitExceededError(ExprError):
    """Expression exceeds safety limits."""
    pass


class RuntimeExprError(ExprError):
    """Error during expression evaluation (invalid arguments, etc.)."""
    pass
