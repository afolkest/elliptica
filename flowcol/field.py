import numba
import numpy as np
from scipy.ndimage import zoom, binary_dilation
from flowcol import defaults
from flowcol.types import Project
from flowcol.poisson import solve_poisson_system, DIRICHLET
from flowcol.mask_utils import blur_mask


@numba.njit(cache=True)
def _relax_potential_band(phi: np.ndarray, relax_mask: np.ndarray, iterations: int, omega: float) -> None:
    height, width = phi.shape
    for it in range(iterations):
        parity = it & 1
        for i in range(height):
            for j in range(width):
                if not relax_mask[i, j]:
                    continue
                if ((i + j) & 1) != parity:
                    continue

                sum_val = 0.0
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < height and 0 <= jj < width:
                        sum_val += phi[ii, jj]
                    else:
                        sum_val += phi[i, j]
                avg = 0.25 * sum_val
                phi[i, j] = (1.0 - omega) * phi[i, j] + omega * avg


def _build_relaxation_mask(dirichlet_mask: np.ndarray, band_width: int) -> np.ndarray:
    if band_width <= 0:
        return np.zeros_like(dirichlet_mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(dirichlet_mask, structure=structure, iterations=band_width)
    return np.logical_and(dilated, ~dirichlet_mask)


def compute_field(
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
    """Compute electric field from conductors on a supersampled, padded grid."""
    canvas_w, canvas_h = project.canvas_resolution
    margin_x, margin_y = margin
    domain_w = canvas_w + 2.0 * margin_x
    domain_h = canvas_h + 2.0 * margin_y
    scale = multiplier * supersample
    field_w = max(1, int(round(domain_w * scale)))
    field_h = max(1, int(round(domain_h * scale)))

    def build_dirichlet_arrays(grid_w: int, grid_h: int) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize Dirichlet mask/values onto a grid of the requested size."""
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        values = np.zeros((grid_h, grid_w), dtype=float)
        grid_scale_x = grid_w / domain_w if domain_w > 0 else 1.0
        grid_scale_y = grid_h / domain_h if domain_h > 0 else 1.0

        for conductor in project.conductors:
            x = (conductor.position[0] + margin_x) * grid_scale_x
            y = (conductor.position[1] + margin_y) * grid_scale_y

            if not np.isclose(grid_scale_x, 1.0) or not np.isclose(grid_scale_y, 1.0):
                scaled_mask = zoom(conductor.mask, (grid_scale_y, grid_scale_x), order=0)
            else:
                scaled_mask = conductor.mask

            scale_factor = (grid_scale_x + grid_scale_y) / 2.0
            scaled_sigma = conductor.edge_smooth_sigma * scale_factor
            scaled_mask = blur_mask(scaled_mask, scaled_sigma)

            mask_h, mask_w = scaled_mask.shape
            ix, iy = int(round(x)), int(round(y))
            x0, y0 = max(0, ix), max(0, iy)
            x1, y1 = min(ix + mask_w, grid_w), min(iy + mask_h, grid_h)

            mx0, my0 = max(0, -ix), max(0, -iy)
            mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

            mask_slice = scaled_mask[my0:my1, mx0:mx1]
            mask_bool = mask_slice > 0.5

            mask[y0:y1, x0:x1] |= mask_bool
            values[y0:y1, x0:x1] = np.where(mask_bool, conductor.voltage, values[y0:y1, x0:x1])

        return mask, values

    def match_shape(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
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

    poisson_scale = max(float(poisson_scale), 0.0)
    if poisson_scale <= 0.0:
        raise ValueError("poisson_scale must be positive")

    dirichlet_mask, dirichlet_values = build_dirichlet_arrays(field_w, field_h)

    use_preview = poisson_scale < 0.999
    if use_preview:
        solve_w = max(1, int(round(field_w * poisson_scale)))
        solve_h = max(1, int(round(field_h * poisson_scale)))
        preview_mask, preview_values = build_dirichlet_arrays(solve_w, solve_h)

        phi_preview = solve_poisson_system(
            preview_mask,
            preview_values,
            boundary_top=boundary_top,
            boundary_bottom=boundary_bottom,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
        )

        zoom_factors = (
            field_h / phi_preview.shape[0],
            field_w / phi_preview.shape[1],
        )
        phi = zoom(phi_preview, zoom_factors, order=1)
        if phi.shape != (field_h, field_w):
            phi = match_shape(phi, (field_h, field_w))
        phi = phi.astype(np.float64, copy=False)
        phi = np.ascontiguousarray(phi)
        phi[dirichlet_mask] = dirichlet_values[dirichlet_mask]
        if (
            defaults.POISSON_PREVIEW_RELAX_ITERS > 0
            and defaults.POISSON_PREVIEW_RELAX_BAND > 0
        ):
            relax_mask = _build_relaxation_mask(
                dirichlet_mask,
                defaults.POISSON_PREVIEW_RELAX_BAND,
            )
            if np.any(relax_mask):
                relax_mask = np.ascontiguousarray(relax_mask)
                _relax_potential_band(
                    phi,
                    relax_mask,
                    defaults.POISSON_PREVIEW_RELAX_ITERS,
                    float(defaults.POISSON_PREVIEW_RELAX_OMEGA),
                )
                phi[dirichlet_mask] = dirichlet_values[dirichlet_mask]
    else:
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
