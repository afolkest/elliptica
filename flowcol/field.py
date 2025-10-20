import numpy as np
from scipy.ndimage import zoom
from flowcol.types import Project
from flowcol.poisson import solve_poisson_system


def compute_field(project: Project, multiplier: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Compute electric field from conductors. Returns (Ex, Ey)."""
    canvas_w, canvas_h = project.canvas_resolution
    field_w, field_h = canvas_w * multiplier, canvas_h * multiplier

    dirichlet_mask = np.zeros((field_h, field_w), dtype=bool)
    dirichlet_values = np.zeros((field_h, field_w), dtype=float)

    for conductor in project.conductors:
        x = round(conductor.position[0] * multiplier)
        y = round(conductor.position[1] * multiplier)

        if multiplier > 1:
            scaled_mask = zoom(conductor.mask, multiplier, order=1)
        else:
            scaled_mask = conductor.mask

        mask_h, mask_w = scaled_mask.shape

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(x + mask_w, field_w), min(y + mask_h, field_h)

        mx0, my0 = max(0, -x), max(0, -y)
        mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

        mask_slice = scaled_mask[my0:my1, mx0:mx1]
        mask_bool = mask_slice > 0.5

        dirichlet_mask[y0:y1, x0:x1] |= mask_bool
        dirichlet_values[y0:y1, x0:x1] = np.where(mask_bool, conductor.voltage, dirichlet_values[y0:y1, x0:x1])

    phi = solve_poisson_system(dirichlet_mask, dirichlet_values)
    grad_y, grad_x = np.gradient(phi)
    return -grad_x, -grad_y
