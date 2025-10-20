import numpy as np
from flowcol.types import Project
from flowcol.poisson import solve_poisson_system


def compute_field(project: Project) -> tuple[np.ndarray, np.ndarray]:
    """Compute electric field from conductors. Returns (Ex, Ey)."""
    canvas_w, canvas_h = project.canvas_resolution

    dirichlet_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    dirichlet_values = np.zeros((canvas_h, canvas_w), dtype=float)

    for conductor in project.conductors:
        x, y = int(conductor.position[0]), int(conductor.position[1])
        mask_h, mask_w = conductor.mask.shape

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(x + mask_w, canvas_w), min(y + mask_h, canvas_h)

        mx0, my0 = max(0, -x), max(0, -y)
        mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

        mask_slice = conductor.mask[my0:my1, mx0:mx1]
        mask_bool = mask_slice > 0.5

        dirichlet_mask[y0:y1, x0:x1] |= mask_bool
        dirichlet_values[y0:y1, x0:x1] = np.where(mask_bool, conductor.voltage, dirichlet_values[y0:y1, x0:x1])

    phi = solve_poisson_system(dirichlet_mask, dirichlet_values)
    grad_y, grad_x = np.gradient(phi)
    return -grad_x, -grad_y
