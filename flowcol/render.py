import numpy as np
import pygame
from PIL import Image
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist
from flowcol.types import RenderInfo, Project
from flowcol.lic import convolve, get_cosine_kernel


def compute_lic(
    ex: np.ndarray,
    ey: np.ndarray,
    *,
    streamlength: int,
    num_passes: int = 1,
    texture: np.ndarray | None = None,
    seed: int | None = 0
) -> np.ndarray:
    """Compute LIC visualization. Returns array normalized to [-1, 1]."""
    field_h, field_w = ex.shape

    if texture is None:
        rng = np.random.default_rng(seed)
        texture = rng.random((field_h, field_w)).astype(np.float32)
    else:
        texture = texture.astype(np.float32, copy=False)

    mag = np.sqrt(ex**2 + ey**2)
    mag_max = max(mag.max(), 1e-10)
    vx = (ex / mag_max).astype(np.float32)
    vy = (ey / mag_max).astype(np.float32)

    kernel = get_cosine_kernel(streamlength).astype(np.float32)
    lic_result = convolve(texture, vx, vy, kernel, iterations=num_passes, boundaries="closed")

    max_abs = np.max(np.abs(lic_result))
    if max_abs > 1e-12:
        lic_result = lic_result / max_abs

    return lic_result


def array_to_surface(arr: np.ndarray) -> pygame.Surface:
    """Convert intensity array to pygame surface."""
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        norm = np.zeros_like(arr)
    img = (norm * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode='L').convert('RGB')
    return pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode)


def save_render(arr: np.ndarray, project: Project, multiplier: int) -> RenderInfo:
    """Save render to file and return RenderInfo."""
    renders_dir = Path("renders")
    renders_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lic_{multiplier}x_{timestamp}.png"
    filepath = renders_dir / filename

    img = (arr * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(filepath)

    render_info = RenderInfo(multiplier=multiplier, filepath=str(filepath), timestamp=timestamp)
    project.renders.append(render_info)

    return render_info



def apply_highpass_clahe(
    arr: np.ndarray,
    sigma: float,
    clip_limit: float,
    kernel_rows: int,
    kernel_cols: int,
    num_bins: int,
) -> np.ndarray:
    """Gaussian high-pass followed by CLAHE. Output retains original dynamic range."""
    sigma = max(sigma, 1e-3)
    clip_limit = max(clip_limit, 1e-4)
    kernel_rows = max(kernel_rows, 1)
    kernel_cols = max(kernel_cols, 1)
    num_bins = max(num_bins, 2)

    high = arr - gaussian_filter(arr, sigma)
    min_val = float(high.min())
    max_val = float(high.max())
    enhanced = equalize_adapthist(
        image=high,
        kernel_size=(kernel_rows, kernel_cols),
        clip_limit=clip_limit,
        nbins=num_bins,
    )
    if max_val > 1.0 or min_val < 0.0:
        enhanced = enhanced * (max_val - min_val) + min_val
    return enhanced
