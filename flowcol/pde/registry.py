"""
Global registry for available PDEs.
"""

from typing import Optional
from .base import PDEDefinition


class PDERegistry:
    """
    Global registry of available PDEs.

    PDEs are registered at startup and can be switched at runtime.
    """
    _pdefs: dict[str, PDEDefinition] = {}
    _active: str = "poisson"  # Default to Poisson for compatibility

    @classmethod
    def register(cls, pde: PDEDefinition) -> None:
        """Register a PDE definition."""
        if pde.name in cls._pdefs:
            raise ValueError(f"PDE '{pde.name}' is already registered")
        cls._pdefs[pde.name] = pde

    @classmethod
    def get(cls, name: str) -> PDEDefinition:
        """Get a specific PDE definition by name."""
        if name not in cls._pdefs:
            raise ValueError(f"Unknown PDE: {name}. Available: {list(cls._pdefs.keys())}")
        return cls._pdefs[name]

    @classmethod
    def get_active(cls) -> PDEDefinition:
        """Get the currently active PDE definition."""
        if not cls._pdefs:
            raise RuntimeError("No PDEs registered. Call register_all_pdes() first.")
        return cls._pdefs[cls._active]

    @classmethod
    def set_active(cls, name: str) -> None:
        """Set the active PDE."""
        if name not in cls._pdefs:
            raise ValueError(f"Unknown PDE: {name}. Available: {list(cls._pdefs.keys())}")
        cls._active = name

    @classmethod
    def get_active_name(cls) -> str:
        """Get the name of the currently active PDE."""
        return cls._active

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered PDE names."""
        return list(cls._pdefs.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered PDEs (mainly for testing)."""
        cls._pdefs.clear()
        cls._active = "poisson"