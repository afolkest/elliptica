"""
Field computation using PDE abstraction.

This module provides the new compute_field_pde function that uses the
PDE registry system while maintaining compatibility with the existing API.
"""

import numpy as np
from scipy.ndimage import zoom
from elliptica.types import Project
from elliptica.pde import PDERegistry
from elliptica.poisson import DIRICHLET
from elliptica import defaults
from elliptica.pde.relaxation import build_relaxation_mask, relax_potential_band
from elliptica.pde.boundary_utils import resolve_bc_map, bc_map_to_legacy, build_dirichlet_from_objects


def compute_field_pde(
    project: Project,
    multiplier: float = 1.0,
    supersample: float = 1.0,
    margin: tuple[float, float] = (0.0, 0.0),
    boundary_top: int = DIRICHLET,
    boundary_bottom: int = DIRICHLET,
    boundary_left: int = DIRICHLET,
    boundary_right: int = DIRICHLET,
    solve_scale: float = 1.0,
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
        solve_scale: Scale factor for PDE solve resolution (0.1-1.0)

    Returns:
        Tuple of (solution_dict, (ex, ey)) where:
        - solution_dict: Dictionary of solution arrays from PDE solver
        - (ex, ey): Extracted vector field for LIC visualization
    """
    # Get active PDE definition
    pde = PDERegistry.get_active()
    if getattr(pde, "bc_fields", None):
        bc_map = resolve_bc_map(project, pde)
        legacy_bc = bc_map_to_legacy(bc_map, DIRICHLET)
        # Persist resolved BCs for this PDE on the project (for UI/state roundtrips)
        if hasattr(project, "pde_bc"):
            project.pde_bc[pde.name] = bc_map
    else:
        bc_map = {}
        legacy_bc = {
            'top': boundary_top,
            'bottom': boundary_bottom,
            'left': boundary_left,
            'right': boundary_right,
        }

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
        def __init__(self, original_project, solve_shape, margin, domain_size, bc_map, legacy_bc):
            self.boundary_objects = original_project.boundary_objects
            self.shape = solve_shape
            self.margin = margin  # (margin_x, margin_y) tuple
            self.domain_size = domain_size  # (domain_w, domain_h) tuple
            self.bc = bc_map  # Rich BC map
            self.boundary_conditions = {
                'top': legacy_bc.get('top', boundary_top),
                'bottom': legacy_bc.get('bottom', boundary_bottom),
                'left': legacy_bc.get('left', boundary_left),
                'right': legacy_bc.get('right', boundary_right),
            }
            self.solve_scale = solve_scale
            self.pde_params = original_project.pde_params
            # Pass through other attributes
            self.canvas_resolution = original_project.canvas_resolution
            self.conductors = original_project.conductors

    # Handle resolution scaling
    if solve_scale < 0.999:
        # Solve at lower resolution
        solve_w = max(1, int(round(field_w * solve_scale)))
        solve_h = max(1, int(round(field_h * solve_scale)))
        solve_project = SolveProject(project, (solve_h, solve_w), margin, (domain_w, domain_h), bc_map, legacy_bc)

        # Solve PDE at lower resolution
        solution_lowres = pde.solve(solve_project)

        # Upscale solution to full resolution
        solution = {}
        for key, array in solution_lowres.items():
            if array.ndim == 2:
                zoom_factors = (field_h / array.shape[0], field_w / array.shape[1])
                # Use nearest neighbor for boolean masks, bilinear for continuous fields
                if array.dtype == bool:
                    upscaled = zoom(array.astype(np.uint8), zoom_factors, order=0) > 0
                else:
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
                defaults.SOLVE_RELAX_ITERS > 0
                and defaults.SOLVE_RELAX_BAND > 0
            ):

                
                # We need to rebuild the Dirichlet mask at full resolution to know where to relax
                # This is a bit expensive but necessary for good previews
                # Create a temporary project for full-res mask generation
                full_res_project = SolveProject(project, (field_h, field_w), margin, (domain_w, domain_h), bc_map, legacy_bc)
                
                # TODO: This assumes Poisson PDE internals (dirichlet_mask). 
                # Ideally the PDE solver would expose a "get_boundary_mask" method.
                # For now, we'll use a helper to generate it.
                dirichlet_mask = _build_dirichlet_mask(full_res_project)
                
                relax_mask = build_relaxation_mask(
                    dirichlet_mask,
                    defaults.SOLVE_RELAX_BAND,
                )
                
                if np.any(relax_mask):
                    phi = solution["phi"]
                    # Ensure contiguous array for Numba
                    if not phi.flags['C_CONTIGUOUS']:
                        phi = np.ascontiguousarray(phi)
                        
                    relax_potential_band(
                        phi,
                        relax_mask,
                        defaults.SOLVE_RELAX_ITERS,
                        float(defaults.SOLVE_RELAX_OMEGA),
                    )
                    
                    # Re-enforce boundary conditions after relaxation
                    # (We need values for this, which is another expense. 
                    #  Optimization: maybe skip this if relaxation is gentle enough?)
                    dirichlet_values = _build_dirichlet_values(full_res_project)
                    phi[dirichlet_mask] = dirichlet_values[dirichlet_mask]
                    solution["phi"] = phi

    else:
        # Solve at full resolution
        solve_project = SolveProject(project, (field_h, field_w), margin, (domain_w, domain_h), bc_map, legacy_bc)
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


def _build_dirichlet_mask(project) -> np.ndarray:
    """Helper to build Dirichlet mask for relaxation."""
    mask, _ = build_dirichlet_from_objects(project)
    return mask


def _build_dirichlet_values(project) -> np.ndarray:
    """Helper to build Dirichlet values for relaxation."""
    _, values = build_dirichlet_from_objects(project)
    return values
