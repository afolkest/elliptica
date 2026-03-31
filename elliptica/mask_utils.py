
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

def load_alpha(path: str, threshold: float = 0.0):
    """Load PNG alpha channel as mask (preserves full alpha values by default)."""
    img = Image.open(path).convert('RGBA')
    alpha = np.array(img)[..., 3] / 255.0
    if threshold > 0.0:
        return (alpha > threshold).astype(np.float32)
    return alpha.astype(np.float32)

def load_boundary_masks(shell_path: str):
    """Load shell mask and try to load matching interior mask."""
    shell = load_alpha(shell_path)
    interior_path = Path(shell_path).parent / Path(shell_path).stem.replace("_shell", "_interior")
    interior_path = interior_path.with_suffix(".png")
    interior = load_alpha(str(interior_path)) if interior_path.exists() else None
    return shell, interior

def blur_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to a mask.

    Args:
        mask: Input mask array
        sigma: Gaussian blur sigma in pixels (0 = no blur)

    Returns:
        Blurred mask, clipped to [0, 1]
    """
    if sigma <= 0:
        return mask
    blurred = gaussian_filter(mask.astype(np.float32), sigma=sigma, mode='reflect')
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def place_mask_in_grid(
    mask: np.ndarray,
    position: tuple[float, float],
    target_shape: tuple[int, int],
    margin: tuple[float, float],
    scale: tuple[float, float],
    edge_smooth_sigma: float = 0.0,
    offset: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Place a boundary mask into a grid with scaling, smoothing, and clipping.

    Args:
        mask: Source mask array (2D float, values in [0, 1]).
        position: (x, y) position on canvas (pre-margin, pre-scale).
        target_shape: (height, width) of the target grid.
        margin: (margin_x, margin_y) physical margin offsets.
        scale: (scale_x, scale_y) grid scale factors.
        edge_smooth_sigma: Edge smoothing sigma in canvas pixels (scaled internally).
        offset: (offset_x, offset_y) crop offsets to subtract from grid position.

    Returns:
        (mask_slice, (y0, y1, x0, x1)) for the valid grid region,
        or None if the mask falls entirely outside the grid.
    """
    target_h, target_w = target_shape
    pos_x, pos_y = position
    margin_x, margin_y = margin
    scale_x, scale_y = scale
    offset_x, offset_y = offset

    # Compute grid position
    grid_x = (pos_x + margin_x) * scale_x - offset_x
    grid_y = (pos_y + margin_y) * scale_y - offset_y

    # Scale mask if needed
    if not np.isclose(scale_x, 1.0) or not np.isclose(scale_y, 1.0):
        scaled_mask = zoom(mask, (scale_y, scale_x), order=0)
    else:
        scaled_mask = mask

    # Clip zoom output to [0, 1] (safety — only needed after interpolation)
    if scaled_mask is not mask:
        scaled_mask = np.clip(scaled_mask, 0.0, 1.0).astype(np.float32)

    # Apply edge smoothing if sigma > 0
    if edge_smooth_sigma > 0:
        scale_factor = (scale_x + scale_y) / 2.0
        scaled_sigma = edge_smooth_sigma * scale_factor
        scaled_mask = blur_mask(scaled_mask, scaled_sigma)

    # Compute overlap region
    mask_h, mask_w = scaled_mask.shape
    ix, iy = int(round(grid_x)), int(round(grid_y))

    x0, y0 = max(0, ix), max(0, iy)
    x1, y1 = min(ix + mask_w, target_w), min(iy + mask_h, target_h)

    if x0 >= x1 or y0 >= y1:
        return None

    mx0, my0 = max(0, -ix), max(0, -iy)
    mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

    mask_slice = scaled_mask[my0:my1, mx0:mx1]

    return mask_slice, (y0, y1, x0, x1)
