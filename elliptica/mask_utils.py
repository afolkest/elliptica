
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter

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
