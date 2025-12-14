
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_closing, generate_binary_structure

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

def create_masks(mask, thickness):
    """Partition mask into shell and interior"""
    dist = distance_transform_edt(mask)
    interior = (dist > thickness).astype(np.float32)
    shell = (mask * (dist <= thickness)).astype(np.float32)
    return shell, interior

def save_mask(mask: np.ndarray, path: str):
    """Save mask as PNG with alpha channel."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., 3] = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgba).save(path)

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

def smooth_mask_morphological(mask: np.ndarray, radius: int) -> np.ndarray:
    """Smooth binary mask boundary using morphological closing.

    Args:
        mask: Binary mask array
        radius: Structuring element radius in pixels (0 = no smoothing)

    Returns:
        Smoothed binary mask
    """
    if radius <= 0:
        return mask
    # Create circular structuring element
    struct = generate_binary_structure(2, radius)
    smoothed = binary_closing(mask > 0.5, structure=struct, iterations=1)
    return smoothed.astype(np.float32)
