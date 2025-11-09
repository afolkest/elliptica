"""
Field computation using PDE abstraction.

This module provides the new compute_field_pde function that uses the
PDE registry system while maintaining compatibility with the existing API.
"""

import numpy as np
from scipy.ndimage import zoom
from flowcol.types import Project
from flowcol.pde import PDERegistry
from flowcol.poisson import DIRICHLET


def compute_field_pde(
    project: Project,
    multiplier: float = 1.0,
    supersample: float = 1.0,
    margin: tuple[float, float] = (0.0, 0.0),
    boundary_top: int = DIRICHLET,
    boundary_bottom: int = DIRICHLET,
    boundary_left: int = DIRICHLET,
    boundary_right: int = DIRICHLET,
    poisson_scale: float = 1.0,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Compute field using the active PDE definition.

    This is the new multi-PDE version that returns both the solution dict
    and the extracted LIC field.

    Args:
        project: Project object with boundary objects and PDE settings
        multiplier: Scale factor for canvas resolution
        supersample: Additional supersampling factor
        margin: Extra margin around canvas
        boundary_*: Boundary conditions (for compatibility)
        poisson_scale: Scale factor for solve resolution

    Returns:
        Tuple of (solution_dict, (ex, ey)) where:
        - solution_dict: Dictionary of solution arrays from PDE solver
        - (ex, ey): Extracted vector field for LIC visualization
    """
    # Get active PDE definition
    pde = PDERegistry.get_active()

    # Calculate grid dimensions
    canvas_w, canvas_h = project.canvas_resolution
    margin_x, margin_y = margin
    domain_w = canvas_w + 2.0 * margin_x
    domain_h = canvas_h + 2.0 * margin_y
    scale = multiplier * supersample
    field_w = max(1, int(round(domain_w * scale)))
    field_h = max(1, int(round(domain_h * scale)))

    # Create a temporary project-like object with the solve dimensions
    # This is a bit of a hack but maintains compatibility
    class SolveProject:
        def __init__(self, original_project, solve_shape, margin, domain_size):
            self.boundary_objects = original_project.boundary_objects
            self.shape = solve_shape
            self.margin = margin  # (margin_x, margin_y) tuple
            self.domain_size = domain_size  # (domain_w, domain_h) tuple
            self.boundary_conditions = {
                'top': boundary_top,
                'bottom': boundary_bottom,
                'left': boundary_left,
                'right': boundary_right,
            }
            self.poisson_scale = poisson_scale
            self.pde_params = original_project.pde_params
            # Pass through other attributes
            self.canvas_resolution = original_project.canvas_resolution
            self.conductors = original_project.conductors

    # Handle resolution scaling for preview mode
    if poisson_scale < 0.999:
        # Solve at lower resolution
        solve_w = max(1, int(round(field_w * poisson_scale)))
        solve_h = max(1, int(round(field_h * poisson_scale)))
        solve_project = SolveProject(project, (solve_h, solve_w), margin, (domain_w, domain_h))

        # Solve PDE at lower resolution
        solution_lowres = pde.solve(solve_project)

        # Upscale solution to full resolution
        solution = {}
        for key, array in solution_lowres.items():
            if array.ndim == 2:
                zoom_factors = (field_h / array.shape[0], field_w / array.shape[1])
                upscaled = zoom(array, zoom_factors, order=1)
                # Ensure exact shape match
                if upscaled.shape != (field_h, field_w):
                    upscaled = _match_shape(upscaled, (field_h, field_w))
                solution[key] = upscaled
            else:
                # For non-2D arrays, just copy
                solution[key] = array
    else:
        # Solve at full resolution
        solve_project = SolveProject(project, (field_h, field_w), margin, (domain_w, domain_h))
        solution = pde.solve(solve_project)

    # Extract LIC field from solution
    ex, ey = pde.extract_lic_field(solution, solve_project)

    return solution, (ex, ey)


def _match_shape(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Crop or pad array (edge mode) to match target shape."""
    target_h, target_w = target_shape
    current_h, current_w = array.shape

    if current_h > target_h:
        array = array[:target_h, :]
    elif current_h < target_h:
        pad_h = target_h - current_h
        array = np.pad(array, ((0, pad_h), (0, 0)), mode="edge")

    if current_w > target_w:
        array = array[:, :target_w]
    elif current_w < target_w:
        pad_w = target_w - current_w
        array = np.pad(array, ((0, 0), (0, pad_w)), mode="edge")

    return array


def compute_field_legacy(
    project: Project,
    multiplier: float = 1.0,
    supersample: float = 1.0,
    margin: tuple[float, float] = (0.0, 0.0),
    boundary_top: int = DIRICHLET,
    boundary_bottom: int = DIRICHLET,
    boundary_left: int = DIRICHLET,
    boundary_right: int = DIRICHLET,
    poisson_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legacy compute_field wrapper that only returns the field.

    This maintains backwards compatibility with existing code.
    """
    _, (ex, ey) = compute_field_pde(
        project, multiplier, supersample, margin,
        boundary_top, boundary_bottom, boundary_left, boundary_right,
        poisson_scale
    )
    return ex, ey