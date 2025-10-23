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

COLOR_PALETTES: dict[str, np.ndarray] = {
    "Ink & Gold": np.array(
        [
            (0.02, 0.04, 0.10),
            (0.10, 0.18, 0.28),
            (0.28, 0.42, 0.40),
            (0.48, 0.56, 0.40),
            (0.75, 0.66, 0.30),
            (0.95, 0.80, 0.22),
            (1.00, 0.90, 0.60),
        ],
        dtype=np.float32,
    ),
    "Deep Ocean": np.array(
        [
            (0.05, 0.05, 0.20),
            (0.10, 0.15, 0.40),
            (0.20, 0.32, 0.62),
            (0.28, 0.50, 0.78),
            (0.45, 0.70, 0.88),
            (0.68, 0.82, 0.94),
            (0.90, 0.96, 1.00),
        ],
        dtype=np.float32,
    ),
    "Ember Ash": np.array(
        [
            (0.00, 0.00, 0.00),
            (0.10, 0.05, 0.02),
            (0.30, 0.12, 0.04),
            (0.60, 0.26, 0.08),
            (0.75, 0.55, 0.35),
            (0.88, 0.89, 0.90),
        ],
        dtype=np.float32,
    ),
}

DEFAULT_COLOR_PALETTE_NAME = "Ink & Gold"


def _build_palette_lut(colors: np.ndarray, size: int = 256) -> np.ndarray:
    """Linearly interpolate palette stops into a lookup table."""
    positions = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
    samples = np.linspace(0.0, 1.0, size, dtype=np.float32)
    lut = np.empty((size, 3), dtype=np.float32)
    for channel in range(3):
        lut[:, channel] = np.interp(samples, positions, colors[:, channel])
    return lut


PALETTE_LUTS: dict[str, np.ndarray] = {
    name: _build_palette_lut(colors) for name, colors in COLOR_PALETTES.items()
}


def list_color_palettes() -> tuple[str, ...]:
    return tuple(PALETTE_LUTS.keys())


def _get_palette_lut(name: str | None) -> np.ndarray:
    if name and name in PALETTE_LUTS:
        return PALETTE_LUTS[name]
    return PALETTE_LUTS[DEFAULT_COLOR_PALETTE_NAME]


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

    vx = ex.astype(np.float32, copy=False)
    vy = ey.astype(np.float32, copy=False)
    if not np.any(vx) and not np.any(vy):
        return np.zeros_like(ex, dtype=np.float32)

    kernel = get_cosine_kernel(streamlength).astype(np.float32)
    lic_result = convolve(texture, vx, vy, kernel, iterations=num_passes, boundaries=boundaries)

    max_abs = np.max(np.abs(lic_result))
    if max_abs > 1e-12:
        lic_result = lic_result / max_abs

    return lic_result


def _normalize_unit(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1]."""
    arr = arr.astype(np.float32, copy=False)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr, dtype=np.float32)


def colorize_array(
    arr: np.ndarray,
    palette: str | None = None,
    contrast: float = 1.0,
    gamma: float = 1.0,
    clip_percent: float = 0.5,
) -> np.ndarray:
    """Map scalar field to RGB using the provided LUT.

    Args:
        arr: Input array to colorize
        palette: Color palette name
        contrast: Contrast adjustment (1.0 = no change)
        gamma: Gamma correction (1.0 = no change)
        clip_percent: Percentile clipping (e.g., 0.5 clips bottom 0.5% and top 0.5%).
                     Use 0.0 for min/max normalization. Default 0.5 matches gauss_law_morph.
    """
    arr = arr.astype(np.float32, copy=False)

    # Use percentile normalization if clip_percent > 0, matching gauss_law_morph behavior
    if clip_percent > 0.0:
        vmin = float(np.percentile(arr, clip_percent))
        vmax = float(np.percentile(arr, 100.0 - clip_percent))
        if vmax > vmin:
            norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            norm = _normalize_unit(arr)
    else:
        norm = _normalize_unit(arr)
    if not np.isclose(contrast, 1.0):
        norm = np.clip((norm - 0.5) * contrast + 0.5, 0.0, 1.0)
    gamma = max(float(gamma), 1e-3)
    if not np.isclose(gamma, 1.0):
        norm = np.power(norm, gamma, dtype=np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    lut = _get_palette_lut(palette)
    idx = np.clip((norm * (lut.shape[0] - 1)).astype(np.int32), 0, lut.shape[0] - 1)
    rgb = lut[idx]
    return (rgb * 255.0).astype(np.uint8)


def array_to_surface(
    arr: np.ndarray,
    *,
    use_color: bool = False,
    palette: str | None = None,
    contrast: float = 1.0,
    gamma: float = 1.0,
    clip_percent: float = 0.5,
) -> pygame.Surface:
    """Convert scalar array to pygame surface, optionally colorized.

    Note: UI always passes clip_percent explicitly from state (defaults to 0.0 there).
          This default of 0.5 is for direct API usage to match gauss_law_morph behavior.
    """
    if use_color:
        rgb = colorize_array(arr, palette=palette, contrast=contrast, gamma=gamma, clip_percent=clip_percent)
        pil_img = Image.fromarray(rgb, mode='RGB')
    else:
        norm = _normalize_unit(arr)
        img = (norm * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img, mode='L').convert('RGB')
    return pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode)


def save_render(
    arr: np.ndarray,
    project: Project,
    multiplier: int,
    *,
    use_color: bool = False,
    palette: str | None = None,
    contrast: float = 1.0,
    gamma: float = 1.0,
    clip_percent: float = 0.5,
) -> RenderInfo:
    """Save render to file and return RenderInfo."""
    renders_dir = Path("renders")
    renders_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lic_{multiplier}x_{timestamp}.png"
    filepath = renders_dir / filename

    if use_color:
        rgb = colorize_array(arr, palette=palette, contrast=contrast, gamma=gamma, clip_percent=clip_percent)
        Image.fromarray(rgb, mode='RGB').save(filepath)
    else:
        norm = _normalize_unit(arr)
        img = (norm * 255.0).astype(np.uint8)
        Image.fromarray(img, mode='L').save(filepath)

    render_info = RenderInfo(multiplier=multiplier, filepath=str(filepath), timestamp=timestamp)
    project.renders.append(render_info)

    return render_info

def apply_gaussian_highpass(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Subtract Gaussian blur from array (returns highpass with unbounded range)."""
    arr = arr.astype(np.float32, copy=False)
    sigma = float(max(sigma, 0.0))
    if sigma <= 1e-6:
        return arr.copy()
    return arr - gaussian_filter(arr, sigma=sigma)


def apply_highpass_clahe(
    arr: np.ndarray,
    sigma: float,
    clip_limit: float,
    kernel_rows: int,
    kernel_cols: int,
    num_bins: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Gaussian high-pass followed by CLAHE, blended with original.

    This is the ORIGINAL working version from gauss_law_morph.
    """
    sigma = max(sigma, 1e-3)
    clip_limit = max(clip_limit, 1e-4)
    kernel_rows = max(kernel_rows, 1)
    kernel_cols = max(kernel_cols, 1)
    num_bins = max(num_bins, 2)
    strength = float(np.clip(strength, 0.0, 1.0))

    # Apply highpass
    high = arr - gaussian_filter(arr, sigma)
    min_val = float(high.min())
    max_val = float(high.max())

    # Apply CLAHE to highpass output
    enhanced = equalize_adapthist(
        image=high,
        kernel_size=(kernel_rows, kernel_cols),
        clip_limit=clip_limit,
        nbins=num_bins,
    )

    # Rescale CLAHE output to match highpass range
    if max_val > 1.0 or min_val < 0.0:
        enhanced = enhanced * (max_val - min_val) + min_val

    # Blend with original
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
