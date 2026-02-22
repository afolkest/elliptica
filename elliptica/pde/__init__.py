"""
PDE abstraction framework for multi-equation support.
"""

from .base import PDEDefinition, SolveContext
from .registry import PDERegistry

__all__ = [
    'PDEDefinition',
    'SolveContext',
    'PDERegistry',
]