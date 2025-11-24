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
from flowcol import defaults
from flowcol.pde.relaxation import build_relaxation_mask, relax_potential_band
from flowcol.mask_utils import blur_mask


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

        # Apply relaxation if enabled (only for 'phi' field currently)
        if "phi" in solution:

            if (
                defaults.POISSON_PREVIEW_RELAX_ITERS > 0
                and defaults.POISSON_PREVIEW_RELAX_BAND > 0
            ):

                
                # We need to rebuild the Dirichlet mask at full resolution to know where to relax
                # This is a bit expensive but necessary for good previews
                # Create a temporary project for full-res mask generation
                full_res_project = SolveProject(project, (field_h, field_w), margin, (domain_w, domain_h))
                
                # TODO: This assumes Poisson PDE internals (dirichlet_mask). 
                # Ideally the PDE solver would expose a "get_boundary_mask" method.
                # For now, we'll use a helper to generate it.
                dirichlet_mask = _build_dirichlet_mask(full_res_project)
                
                relax_mask = build_relaxation_mask(
                    dirichlet_mask,
                    defaults.POISSON_PREVIEW_RELAX_BAND,
                )
                
                if np.any(relax_mask):
                    phi = solution["phi"]
                    # Ensure contiguous array for Numba
                    if not phi.flags['C_CONTIGUOUS']:
                        phi = np.ascontiguousarray(phi)
                        
                    relax_potential_band(
                        phi,
                        relax_mask,
                        defaults.POISSON_PREVIEW_RELAX_ITERS,
                        float(defaults.POISSON_PREVIEW_RELAX_OMEGA),
                    )
                    
                    # Re-enforce boundary conditions after relaxation
                    # (We need values for this, which is another expense. 
                    #  Optimization: maybe skip this if relaxation is gentle enough?)
                    dirichlet_values = _build_dirichlet_values(full_res_project)
                    phi[dirichlet_mask] = dirichlet_values[dirichlet_mask]
                    solution["phi"] = phi

    else:
        # Solve at full resolution
        solve_project = SolveProject(project, (field_h, field_w), margin, (domain_w, domain_h))
        solution = pde.solve(solve_project)

    # Extract LIC field from solution
    ex, ey = pde.extract_lic_field(solution, solve_project)

    return solution, (ex, ey)


    return ex, ey


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


def _build_dirichlet_mask(project) -> np.ndarray:
    """Helper to build Dirichlet mask for relaxation."""

    
    grid_h, grid_w = project.shape
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    
    domain_w, domain_h = project.domain_size
    grid_scale_x = grid_w / domain_w if domain_w > 0 else 1.0
    grid_scale_y = grid_h / domain_h if domain_h > 0 else 1.0
    margin_x, margin_y = project.margin

    for obj in project.boundary_objects:
        x = (obj.position[0] + margin_x) * grid_scale_x
        y = (obj.position[1] + margin_y) * grid_scale_y

        obj_mask = obj.mask
        if not np.isclose(grid_scale_x, 1.0) or not np.isclose(grid_scale_y, 1.0):
            obj_mask = zoom(obj_mask, (grid_scale_y, grid_scale_x), order=0)

        if hasattr(obj, 'edge_smooth_sigma') and obj.edge_smooth_sigma > 0:
            scale_factor = (grid_scale_x + grid_scale_y) / 2.0
            scaled_sigma = obj.edge_smooth_sigma * scale_factor
            obj_mask = blur_mask(obj_mask, scaled_sigma)

        mask_h, mask_w = obj_mask.shape
        ix, iy = int(round(x)), int(round(y))
        x0, y0 = max(0, ix), max(0, iy)
        x1, y1 = min(ix + mask_w, grid_w), min(iy + mask_h, grid_h)

        mx0, my0 = max(0, -ix), max(0, -iy)
        mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

        mask_slice = obj_mask[my0:my1, mx0:mx1]
        mask[y0:y1, x0:x1] |= (mask_slice > 0.5)
        
    return mask


def _build_dirichlet_values(project) -> np.ndarray:
    """Helper to build Dirichlet values for relaxation."""

    
    grid_h, grid_w = project.shape
    values = np.zeros((grid_h, grid_w), dtype=float)
    
    domain_w, domain_h = project.domain_size
    grid_scale_x = grid_w / domain_w if domain_w > 0 else 1.0
    grid_scale_y = grid_h / domain_h if domain_h > 0 else 1.0
    margin_x, margin_y = project.margin

    for obj in project.boundary_objects:
        x = (obj.position[0] + margin_x) * grid_scale_x
        y = (obj.position[1] + margin_y) * grid_scale_y

        obj_mask = obj.mask
        if not np.isclose(grid_scale_x, 1.0) or not np.isclose(grid_scale_y, 1.0):
            obj_mask = zoom(obj_mask, (grid_scale_y, grid_scale_x), order=0)

        if hasattr(obj, 'edge_smooth_sigma') and obj.edge_smooth_sigma > 0:
            scale_factor = (grid_scale_x + grid_scale_y) / 2.0
            scaled_sigma = obj.edge_smooth_sigma * scale_factor
            obj_mask = blur_mask(obj_mask, scaled_sigma)

        mask_h, mask_w = obj_mask.shape
        ix, iy = int(round(x)), int(round(y))
        x0, y0 = max(0, ix), max(0, iy)
        x1, y1 = min(ix + mask_w, grid_w), min(iy + mask_h, grid_h)

        mx0, my0 = max(0, -ix), max(0, -iy)
        mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

        mask_slice = obj_mask[my0:my1, mx0:mx1]
        mask_bool = mask_slice > 0.5
        
        value = obj.voltage if hasattr(obj, 'voltage') else obj.value
        values[y0:y1, x0:x1] = np.where(mask_bool, value, values[y0:y1, x0:x1])
        
    return values


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