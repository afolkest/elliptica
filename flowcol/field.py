import numpy as np
from scipy.ndimage import zoom
from flowcol.types import Project
from flowcol.poisson import solve_poisson_system, DIRICHLET
from flowcol.mask_utils import blur_mask


def compute_field(
    project: Project,
    multiplier: float = 1.0,
    supersample: float = 1.0,
    margin: tuple[float, float] = (0.0, 0.0),
    boundary_top: int = DIRICHLET,
    boundary_bottom: int = DIRICHLET,
    boundary_left: int = DIRICHLET,
    boundary_right: int = DIRICHLET,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute electric field from conductors on a supersampled, padded grid."""
    canvas_w, canvas_h = project.canvas_resolution
    margin_x, margin_y = margin
    domain_w = canvas_w + 2.0 * margin_x
    domain_h = canvas_h + 2.0 * margin_y
    scale = multiplier * supersample
    field_w = max(1, int(round(domain_w * scale)))
    field_h = max(1, int(round(domain_h * scale)))
    scale_x = field_w / domain_w if domain_w > 0 else 1.0
    scale_y = field_h / domain_h if domain_h > 0 else 1.0

    dirichlet_mask = np.zeros((field_h, field_w), dtype=bool)
    dirichlet_values = np.zeros((field_h, field_w), dtype=float)

    for conductor in project.conductors:
        x = (conductor.position[0] + margin_x) * scale_x
        y = (conductor.position[1] + margin_y) * scale_y

        # Scale mask first to field resolution
        if not np.isclose(scale_x, 1.0) or not np.isclose(scale_y, 1.0):
            # Use order=0 (nearest neighbor) to preserve sharp edges and prevent thinning
            scaled_mask = zoom(conductor.mask, (scale_y, scale_x), order=0)
        else:
            scaled_mask = conductor.mask

        # Apply edge smoothing AFTER scaling at field resolution
        # Scale sigma proportionally to field resolution
        scale_factor = (scale_x + scale_y) / 2.0
        scaled_sigma = conductor.edge_smooth_sigma * scale_factor
        scaled_mask = blur_mask(scaled_mask, scaled_sigma)

        mask_h, mask_w = scaled_mask.shape

        ix, iy = int(round(x)), int(round(y))
        x0, y0 = max(0, ix), max(0, iy)
        x1, y1 = min(ix + mask_w, field_w), min(iy + mask_h, field_h)

        mx0, my0 = max(0, -ix), max(0, -iy)
        mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

        mask_slice = scaled_mask[my0:my1, mx0:mx1]
        mask_bool = mask_slice > 0.5

        dirichlet_mask[y0:y1, x0:x1] |= mask_bool
        dirichlet_values[y0:y1, x0:x1] = np.where(mask_bool, conductor.voltage, dirichlet_values[y0:y1, x0:x1])

    phi = solve_poisson_system(
        dirichlet_mask,
        dirichlet_values,
        boundary_top=boundary_top,
        boundary_bottom=boundary_bottom,
        boundary_left=boundary_left,
        boundary_right=boundary_right,
    )
    grad_y, grad_x = np.gradient(phi)
    return -grad_x, -grad_y
