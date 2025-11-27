"""
PDE abstraction framework for multi-equation support.
"""

from .base import PDEDefinition
from .registry import PDERegistry

__all__ = [
    'PDEDefinition',
    'PDERegistry',
]