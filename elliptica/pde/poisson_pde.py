"""
Poisson equation PDE implementation (electrostatics).
"""

import numpy as np
from ..poisson import solve_poisson_system, DIRICHLET
from ..mask_utils import place_mask_in_grid
from .base import PDEDefinition, BCField, SolveContext


# Constants for inner boundary BC types
INNER_DIRICHLET = 0  # Fixed potential (default boundary behavior)
INNER_NEUMANN = 1    # Specified normal flux (∂φ/∂n = g)


def solve_poisson(project: SolveContext) -> dict[str, np.ndarray]:
    """
    Solve the Poisson equation for electrostatic potential.

    Supports both Dirichlet (φ = V) and Neumann (∂φ/∂n = g) boundaries
    on interior objects. Neumann boundaries can have non-zero flux values.

    Args:
        project: Project object with boundary_objects and shape

    Returns:
        Dictionary with 'phi' key containing the potential field,
        and 'dirichlet_mask' for LIC blocking if any boundaries exist.
    """
    boundary_objects = project.boundary_objects
    grid_h, grid_w = project.shape

    if not boundary_objects:
        phi = np.zeros((grid_h, grid_w), dtype=np.float32)
        return {"phi": phi}

    # Separate masks for Dirichlet vs Neumann boundaries
    dirichlet_mask = np.zeros((grid_h, grid_w), dtype=bool)
    dirichlet_values = np.zeros((grid_h, grid_w), dtype=float)
    neumann_mask = np.zeros((grid_h, grid_w), dtype=bool)
    neumann_values = np.zeros((grid_h, grid_w), dtype=float)

    # Get domain dimensions and margin from project
    domain_w, domain_h = project.domain_size
    grid_scale_x = grid_w / domain_w if domain_w > 0 else 1.0
    grid_scale_y = grid_h / domain_h if domain_h > 0 else 1.0

    margin_x, margin_y = project.margin

    for obj in boundary_objects:
        result = place_mask_in_grid(
            obj.mask, obj.position, (grid_h, grid_w),
            margin=(margin_x, margin_y), scale=(grid_scale_x, grid_scale_y),
            edge_smooth_sigma=obj.edge_smooth_sigma,
        )
        if result is None:
            continue
        mask_slice, (y0, y1, x0, x1) = result
        mask_bool = mask_slice > 0.5

        # Check boundary type for this object
        bc_type = obj.params.get('bc_type', INNER_DIRICHLET)

        if bc_type == INNER_NEUMANN:
            # Neumann boundary - add to neumann_mask with flux value
            flux = obj.params.get('neumann_flux', 0.0)
            neumann_mask[y0:y1, x0:x1] |= mask_bool
            neumann_values[y0:y1, x0:x1] = np.where(
                mask_bool, flux, neumann_values[y0:y1, x0:x1]
            )
        else:
            # Dirichlet (fixed potential) boundary - add to dirichlet_mask
            value = obj.params.get('voltage', 0.0)
            dirichlet_mask[y0:y1, x0:x1] |= mask_bool
            dirichlet_values[y0:y1, x0:x1] = np.where(
                mask_bool, value, dirichlet_values[y0:y1, x0:x1]
            )

    # Get domain edge boundary conditions
    bc = project.boundary_conditions if hasattr(project, 'boundary_conditions') else {}

    # Check if we have any Neumann boundaries
    has_neumann = neumann_mask.any()

    # Solve the Poisson equation
    phi = solve_poisson_system(
        dirichlet_mask,
        dirichlet_values,
        boundary_top=bc.get('top', DIRICHLET),
        boundary_bottom=bc.get('bottom', DIRICHLET),
        boundary_left=bc.get('left', DIRICHLET),
        boundary_right=bc.get('right', DIRICHLET),
        neumann_mask=neumann_mask if has_neumann else None,
        neumann_values=neumann_values if has_neumann else None,
    )

    result = {"phi": phi}

    # Include combined mask for LIC blocking (both Dirichlet and Neumann regions)
    # Field is zero inside both types of boundaries
    combined_mask = dirichlet_mask | neumann_mask
    if combined_mask.any():
        result["dirichlet_mask"] = combined_mask

    return result


def extract_electric_field(solution: dict[str, np.ndarray], project: SolveContext) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract electric field from potential.

    The electric field is the negative gradient of the potential: E = -∇φ

    Args:
        solution: Dictionary containing 'phi' key with potential
        project: Project object (unused but required by interface)

    Returns:
        Tuple of (ex, ey) electric field components
    """
    phi = solution["phi"]

    # Compute gradient
    gy, gx = np.gradient(phi)

    # Electric field is negative gradient
    return -gx, -gy


# Create the Poisson PDE definition
POISSON_PDE = PDEDefinition(
    name="poisson",
    display_name="Electrostatics",
    description="Solve Laplace equation for electric potential and field",
    solve=solve_poisson,
    extract_lic_field=extract_electric_field,
    solution_variables=[("phi", "Electric potential")],
    boundary_params=[],  # Using boundary_fields instead for richer controls
    boundary_fields=[
        BCField(
            name="bc_type",
            display_name="Boundary Type",
            field_type="enum",
            default=INNER_DIRICHLET,
            choices=[("Dirichlet (fixed V)", INNER_DIRICHLET), ("Neumann (flux BC)", INNER_NEUMANN)],
            description="Dirichlet = fixed potential, Neumann = specified normal flux"
        ),
        BCField(
            name="voltage",
            display_name="Voltage",
            field_type="float",
            default=0.0,
            min_value=-1.0,
            max_value=1.0,
            description="Electric potential of the boundary",
            visible_when={"bc_type": INNER_DIRICHLET},
        ),
        BCField(
            name="neumann_flux",
            display_name="Normal Flux (∂φ/∂n)",
            field_type="float",
            default=0.0,
            min_value=-1.0,
            max_value=1.0,
            description="Normal derivative of potential at boundary (0 = insulating)",
            visible_when={"bc_type": INNER_NEUMANN},
        ),
    ],
    bc_fields=[
        BCField(
            name="type",
            display_name="Boundary Type",
            field_type="enum",
            default=DIRICHLET,
            choices=[("Dirichlet", DIRICHLET), ("Neumann", 1)],
            description="Dirichlet=fixed potential, Neumann=insulated edge"
        )
    ],
)
