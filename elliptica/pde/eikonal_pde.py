"""
Eikonal equation PDE implementation (geometric optics).

Solves |∇φ|² = n(x,y)² for wavefront propagation using Fast Marching Method.
"""

import numpy as np
from typing import Any
from scipy.ndimage import zoom
import skfmm

from .base import PDEDefinition, BCField
from ..mask_utils import blur_mask

# Object type constants
SOURCE = 0
LENS = 1

# Edge BC types
EDGE_OPEN = 0      # Not a source, waves propagate through
EDGE_SOURCE = 1    # Plane wave source from this edge


def solve_eikonal(project: Any) -> dict[str, np.ndarray]:
    """
    Solve the eikonal equation for optical wavefront propagation.

    |∇φ| = n(x,y)  (or equivalently, |∇φ| = 1/speed)

    Uses Fast Marching Method via scikit-fmm.

    Objects can be:
    - Source: light origin (φ = 0 here)
    - Lens: region with refractive index n (high n ≈ blocker)

    Args:
        project: Project object with boundary_objects and shape

    Returns:
        Dictionary with 'phi' (travel time/phase) and 'n_field'
    """
    boundary_objects = project.boundary_objects
    grid_h, grid_w = project.shape

    # Initialize refractive index field (n=1 is free space)
    n_field = np.ones((grid_h, grid_w), dtype=np.float64)

    # phi_init for FMM: negative inside sources, positive outside
    # FMM propagates from the zero level set
    phi_init = np.ones((grid_h, grid_w), dtype=np.float64)

    # Track source regions for LIC masking (sources block, lenses don't)
    source_mask = np.zeros((grid_h, grid_w), dtype=bool)

    has_source = False

    # Get domain scaling
    if hasattr(project, 'domain_size'):
        domain_w, domain_h = project.domain_size
        grid_scale_x = grid_w / domain_w if domain_w > 0 else 1.0
        grid_scale_y = grid_h / domain_h if domain_h > 0 else 1.0
    else:
        grid_scale_x = 1.0
        grid_scale_y = 1.0

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

        if x1 <= x0 or y1 <= y0:
            continue

        mask_slice = obj_mask[my0:my1, mx0:mx1]
        mask_bool = mask_slice > 0.5

        # Get object type
        obj_type = obj.params.get('object_type', LENS)

        if obj_type == SOURCE:
            # Source: mark as negative in phi_init (wavefront origin)
            phi_init[y0:y1, x0:x1] = np.where(mask_bool, -1.0, phi_init[y0:y1, x0:x1])
            # Track for LIC masking
            source_mask[y0:y1, x0:x1] |= mask_bool
            has_source = True
        else:
            # Lens: set refractive index
            n_value = obj.params.get('refractive_index', 1.5)
            n_field[y0:y1, x0:x1] = np.where(mask_bool, n_value, n_field[y0:y1, x0:x1])

    # Check edge boundary conditions for sources
    bc = project.boundary_conditions if hasattr(project, 'boundary_conditions') else {}
    # Also check the richer bc dict if available (has 'type' subfield)
    bc_rich = getattr(project, 'bc', {}) or {}

    def edge_is_source(edge_name):
        # Check rich BC first
        if edge_name in bc_rich:
            entry = bc_rich[edge_name]
            if isinstance(entry, dict):
                return entry.get('type', EDGE_OPEN) == EDGE_SOURCE
        # Fall back to simple BC
        return bc.get(edge_name, EDGE_OPEN) == EDGE_SOURCE

    if edge_is_source('left'):
        phi_init[:, 0] = -1.0
        has_source = True
    if edge_is_source('right'):
        phi_init[:, -1] = -1.0
        has_source = True
    if edge_is_source('top'):
        phi_init[0, :] = -1.0
        has_source = True
    if edge_is_source('bottom'):
        phi_init[-1, :] = -1.0
        has_source = True

    # Fallback: if still no sources, use left edge
    if not has_source:
        phi_init[:, 0] = -1.0

    # Speed = 1/n (FMM uses speed, light slows in high-n media)
    speed = 1.0 / n_field

    # Solve eikonal with FMM
    phi = skfmm.travel_time(phi_init, speed)

    # Handle any masked/invalid values
    if hasattr(phi, 'mask'):
        phi = np.where(phi.mask, 0.0, phi.data)

    # Sources block LIC (solid emitters), lenses don't (transparent)
    return {
        "phi": phi.astype(np.float32),
        "n_field": n_field.astype(np.float32),
        "dirichlet_mask": source_mask,
    }


def extract_rays(solution: dict[str, np.ndarray], project: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ray direction field from travel time.

    Rays propagate in direction of ∇φ (away from sources, perpendicular to wavefronts).

    Args:
        solution: Dictionary containing 'phi' (travel time)
        project: Project object (unused)

    Returns:
        Tuple of (ex, ey) ray direction components (normalized to unit vectors)
    """
    phi = solution["phi"]

    # Compute gradient (gy, gx order because numpy uses row-major)
    gy, gx = np.gradient(phi)

    # Normalize to unit vectors for LIC
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.where(mag < 1e-10, 1.0, mag)  # Avoid division by zero

    return gx / mag, gy / mag


# PDE Definition
EIKONAL_PDE = PDEDefinition(
    name="eikonal",
    display_name="Geometric Optics",
    description="Solve eikonal equation for light propagation (ray optics with refraction)",
    solve=solve_eikonal,
    extract_lic_field=extract_rays,
    boundary_params=[],
    boundary_fields=[
        BCField(
            name="object_type",
            display_name="Type",
            field_type="enum",
            default=LENS,
            choices=[("Source", SOURCE), ("Lens", LENS)],
            description="Source = light origin, Lens = refractive region"
        ),
        BCField(
            name="refractive_index",
            display_name="Refractive Index (n)",
            field_type="float",
            default=1.5,
            min_value=0.1,
            max_value=20.0,
            description="n>1 bends rays inward (converging), n<1 bends outward, high n ≈ blocker",
            visible_when={"object_type": LENS},
        ),
    ],
    bc_fields=[
        BCField(
            name="type",
            display_name="Edge Type",
            field_type="enum",
            default=EDGE_OPEN,
            choices=[("Open", EDGE_OPEN), ("Source (plane wave)", EDGE_SOURCE)],
            description="Open = waves pass through, Source = plane wave enters from this edge"
        ),
    ],
)
