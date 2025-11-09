"""
PDE registration module.

Call register_all_pdes() at application startup to register all available PDEs.
"""

from .registry import PDERegistry
from .poisson_pde import POISSON_PDE


def register_all_pdes() -> None:
    """
    Register all available PDEs with the global registry.

    This should be called once at application startup.
    """
    # Clear any existing registrations
    PDERegistry.clear()

    # Register Poisson equation (electrostatics)
    PDERegistry.register(POISSON_PDE)

    # Set default active PDE
    PDERegistry.set_active("poisson")


def list_pde_names() -> list[str]:
    """Get list of available PDE names."""
    return PDERegistry.list_available()


def get_active_pde_name() -> str:
    """Get the name of the currently active PDE."""
    return PDERegistry.get_active_name()