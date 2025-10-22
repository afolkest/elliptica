import numpy as np
import pygame
from PIL import Image
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter, zoom
from skimage.exposure import equalize_adapthist
from flowcol.types import RenderInfo, Project
from flowcol.lic import convolve, get_cosine_kernel
from flowcol import defaults


def generate_noise(
    shape: tuple[int, int],
    seed: int | None,
    oversample: float = 1.0,
    lowpass_sigma: float = defaults.DEFAULT_NOISE_SIGMA,
) -> np.ndarray:
    height, width = shape
    rng = np.random.default_rng(seed)
    if oversample > 1.0:
        high_h = max(1, int(round(height * oversample)))
        high_w = max(1, int(round(width * oversample)))
        base = rng.random((high_h, high_w)).astype(np.float32)
    else:
        base = rng.random((height, width)).astype(np.float32)

    sigma = max(lowpass_sigma, 1e-3)
    base = gaussian_filter(base, sigma=sigma)

    if oversample > 1.0:
        scale_y = height / base.shape[0]
        scale_x = width / base.shape[1]
        base = zoom(base, (scale_y, scale_x), order=1)

    base_min = float(base.min())
    base_max = float(base.max())
    if base_max > base_min:
        base = (base - base_min) / (base_max - base_min)
    else:
        base = np.zeros_like(base)
    return base.astype(np.float32)


def compute_lic(
    ex: np.ndarray,
    ey: np.ndarray,
    streamlength: int,
    num_passes: int = 1,
    *,
    texture: np.ndarray | None = None,
    seed: int | None = 0,
    boundaries: str = "closed",
    noise_oversample: float = 1.5,
    noise_sigma: float = defaults.DEFAULT_NOISE_SIGMA,
) -> np.ndarray:
    """Compute LIC visualization. Returns array normalized to [-1, 1]."""
    field_h, field_w = ex.shape

    streamlength = max(int(streamlength), 1)

    if texture is None:
        texture = generate_noise(
            (field_h, field_w),
            seed,
            oversample=noise_oversample,
            lowpass_sigma=noise_sigma,
        )
    else:
        texture = texture.astype(np.float32, copy=False)

    mag = np.sqrt(ex**2 + ey**2)
    mag_max = max(mag.max(), 1e-10)
    vx = (ex / mag_max).astype(np.float32)
    vy = (ey / mag_max).astype(np.float32)

    kernel = get_cosine_kernel(streamlength).astype(np.float32)
    lic_result = convolve(texture, vx, vy, kernel, iterations=num_passes, boundaries=boundaries)

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

    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        norm = np.zeros_like(arr)
    img = (norm * 255).astype(np.uint8)
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
    strength: float = defaults.DEFAULT_CLAHE_STRENGTH,
) -> np.ndarray:
    """Gaussian high-pass followed by CLAHE blended with original array."""
    sigma = max(sigma, 1e-3)
    clip_limit = max(clip_limit, 1e-4)
    kernel_rows = max(kernel_rows, 1)
    kernel_cols = max(kernel_cols, 1)
    num_bins = max(num_bins, 2)
    strength = float(np.clip(strength, 0.0, 1.0))

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
    if strength >= 1.0:
        return enhanced
    if strength <= 0.0:
        return arr
    return (1.0 - strength) * arr + strength * enhanced


def downsample_lic(
    arr: np.ndarray,
    target_shape: tuple[int, int],
    supersample: float,
    sigma: float,
) -> np.ndarray:
    """Gaussian blur + bilinear resize from supersampled grid to target resolution."""
    if arr.shape == target_shape:
        return arr.copy()

    sigma = max(sigma, 0.0)
    filtered = gaussian_filter(arr, sigma=sigma) if sigma > 0 else arr
    scale_y = target_shape[0] / filtered.shape[0]
    scale_x = target_shape[1] / filtered.shape[1]
    return zoom(filtered, (scale_y, scale_x), order=1)
