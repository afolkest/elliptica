"""
Poisson equation PDE implementation (electrostatics).
"""

import numpy as np
from typing import Any
from scipy.ndimage import zoom
from ..poisson import solve_poisson_system, DIRICHLET
from ..mask_utils import blur_mask
from .base import PDEDefinition


def solve_poisson(project: Any) -> dict[str, np.ndarray]:
    """
    Solve the Poisson equation for electrostatic potential.

    Args:
        project: Project object with boundary_objects and shape

    Returns:
        Dictionary with 'phi' key containing the potential field
    """
    # Extract boundary objects (conductors)
    boundary_objects = project.boundary_objects
    grid_h, grid_w = project.shape

    if not boundary_objects:
        # No boundary objects, return zero field
        phi = np.zeros((grid_h, grid_w), dtype=np.float32)
        return {"phi": phi}

    # Build dirichlet mask and values from boundary objects
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    values = np.zeros((grid_h, grid_w), dtype=float)

    # Get domain dimensions and margin from project
    if hasattr(project, 'domain_size'):
        domain_w, domain_h = project.domain_size
        grid_scale_x = grid_w / domain_w if domain_w > 0 else 1.0
        grid_scale_y = grid_h / domain_h if domain_h > 0 else 1.0
    else:
        # Fallback for direct calls without SolveProject wrapper
        domain_w = grid_w
        domain_h = grid_h
        grid_scale_x = 1.0
        grid_scale_y = 1.0

    # Get margin for positioning
    margin_x, margin_y = project.margin if hasattr(project, 'margin') else (0, 0)

    for obj in boundary_objects:
        # Get position with margin adjustment
        if hasattr(obj, 'position'):
            x = (obj.position[0] + margin_x) * grid_scale_x
            y = (obj.position[1] + margin_y) * grid_scale_y
        else:
            x = margin_x * grid_scale_x
            y = margin_y * grid_scale_y

        # Scale mask if needed
        obj_mask = obj.mask
        if not np.isclose(grid_scale_x, 1.0) or not np.isclose(grid_scale_y, 1.0):
            obj_mask = zoom(obj_mask, (grid_scale_y, grid_scale_x), order=0)

        # Apply edge smoothing if specified
        if hasattr(obj, 'edge_smooth_sigma') and obj.edge_smooth_sigma > 0:
            scale_factor = (grid_scale_x + grid_scale_y) / 2.0
            scaled_sigma = obj.edge_smooth_sigma * scale_factor
            obj_mask = blur_mask(obj_mask, scaled_sigma)

        # Place mask in grid
        mask_h, mask_w = obj_mask.shape
        ix, iy = int(round(x)), int(round(y))
        x0, y0 = max(0, ix), max(0, iy)
        x1, y1 = min(ix + mask_w, grid_w), min(iy + mask_h, grid_h)

        mx0, my0 = max(0, -ix), max(0, -iy)
        mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

        mask_slice = obj_mask[my0:my1, mx0:mx1]
        mask_bool = mask_slice > 0.5

        # Get value (voltage or generic value)
        value = obj.voltage if hasattr(obj, 'voltage') else obj.value

        mask[y0:y1, x0:x1] |= mask_bool
        values[y0:y1, x0:x1] = np.where(mask_bool, value, values[y0:y1, x0:x1])

    # Get boundary conditions
    bc = project.boundary_conditions if hasattr(project, 'boundary_conditions') else {}

    # Solve the Poisson equation
    phi = solve_poisson_system(
        mask,
        values,
        boundary_top=bc.get('top', DIRICHLET),
        boundary_bottom=bc.get('bottom', DIRICHLET),
        boundary_left=bc.get('left', DIRICHLET),
        boundary_right=bc.get('right', DIRICHLET),
    )

    return {"phi": phi}


def extract_electric_field(solution: dict[str, np.ndarray], project: Any) -> tuple[np.ndarray, np.ndarray]:
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
    description="Solve Poisson equation ∇²φ = 0 for electric potential and field",
    solve=solve_poisson,
    extract_lic_field=extract_electric_field,
)