"""
PDE registration module.

Call register_all_pdes() at application startup to register all available PDEs.
"""

from .registry import PDERegistry
from .poisson_pde import POISSON_PDE
from .biharmonic_pde import BIHARMONIC_PDE


def register_all_pdes() -> None:
    """
    Register all available PDEs with the global registry.

    This should be called once at application startup.
    """
    # Clear any existing registrations
    PDERegistry.clear()

    # Register Poisson equation (electrostatics)
    PDERegistry.register(POISSON_PDE)

    # Register Biharmonic equation (Stokes flow)
    PDERegistry.register(BIHARMONIC_PDE)

    # Set default active PDE
    PDERegistry.set_active("poisson")