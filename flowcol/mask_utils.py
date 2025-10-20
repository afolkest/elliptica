
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

def load_alpha(path: str, threshold: float = 0.8):
    """Load PNG alpha channel as binary mask."""
    img = Image.open(path).convert('RGBA')
    alpha = np.array(img)[..., 3] / 255.0
    return (alpha > threshold).astype(np.float32)

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
