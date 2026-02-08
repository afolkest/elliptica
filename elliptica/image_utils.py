import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from elliptica.palettes import _get_palette_lut


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
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    clip_low_percent: float = 0.5,
    clip_high_percent: float = 0.5,
) -> np.ndarray:
    """Map scalar field to RGB using the provided LUT.

    Args:
        arr: Input array to colorize
        palette: Color palette name
        brightness: Brightness adjustment (0.0 = no change, additive)
        contrast: Contrast adjustment (1.0 = no change)
        gamma: Gamma correction (1.0 = no change)
        clip_low_percent: Percentile clipping from low end (e.g., 0.5 clips bottom 0.5%).
        clip_high_percent: Percentile clipping from high end (e.g., 0.5 clips top 0.5%).
                          Use 0.0/0.0 for min/max normalization.
    """
    arr = arr.astype(np.float32, copy=False)

    low = max(float(clip_low_percent), 0.0)
    high = max(float(clip_high_percent), 0.0)
    if low > 0.0 or high > 0.0:
        lower = max(0.0, min(low, 100.0))
        upper = max(0.0, min(100.0 - high, 100.0))
        if upper > lower:
            vmin = float(np.percentile(arr, lower))
            vmax = float(np.percentile(arr, upper))
            if vmax > vmin:
                norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                norm = _normalize_unit(arr)
        else:
            norm = _normalize_unit(arr)
    else:
        norm = _normalize_unit(arr)
    if not np.isclose(contrast, 1.0):
        norm = np.clip((norm - 0.5) * contrast + 0.5, 0.0, 1.0)
    if not np.isclose(brightness, 0.0):
        norm = np.clip(norm + brightness, 0.0, 1.0)
    gamma = max(float(gamma), 1e-3)
    if not np.isclose(gamma, 1.0):
        norm = np.power(norm, gamma).astype(np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    lut = _get_palette_lut(palette)
    idx = np.clip((norm * (lut.shape[0] - 1)).astype(np.int32), 0, lut.shape[0] - 1)
    rgb = lut[idx]
    return (rgb * 255.0).astype(np.uint8)


def _apply_display_transforms(
    arr: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    clip_low_percent: float = 0.5,
    clip_high_percent: float = 0.5,
) -> np.ndarray:
    """Apply clip/brightness/contrast/gamma transforms to normalize array to [0,1]."""
    arr = arr.astype(np.float32, copy=False)

    # Percentile-based normalization (clipping)
    low = max(float(clip_low_percent), 0.0)
    high = max(float(clip_high_percent), 0.0)
    if low > 0.0 or high > 0.0:
        lower = max(0.0, min(low, 100.0))
        upper = max(0.0, min(100.0 - high, 100.0))
        if upper > lower:
            vmin = float(np.percentile(arr, lower))
            vmax = float(np.percentile(arr, upper))
            if vmax > vmin:
                norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                norm = _normalize_unit(arr)
        else:
            norm = _normalize_unit(arr)
    else:
        norm = _normalize_unit(arr)

    # Contrast adjustment
    if not np.isclose(contrast, 1.0):
        norm = np.clip((norm - 0.5) * contrast + 0.5, 0.0, 1.0)

    # Brightness adjustment (after contrast, before gamma)
    if not np.isclose(brightness, 0.0):
        norm = np.clip(norm + brightness, 0.0, 1.0)

    # Gamma correction
    gamma = max(float(gamma), 1e-3)
    if not np.isclose(gamma, 1.0):
        norm = np.power(norm, gamma).astype(np.float32)

    return np.clip(norm, 0.0, 1.0)


def array_to_pil(
    arr: np.ndarray,
    *,
    use_color: bool = False,
    palette: str | None = None,
    contrast: float = 1.0,
    gamma: float = 1.0,
    clip_low_percent: float = 0.5,
    clip_high_percent: float = 0.5,
) -> Image.Image:
    """Convert scalar array to PIL Image, optionally colorized.

    Framework-agnostic version for Streamlit, headless rendering, etc.

    Note: Default clip (0.5/0.5) matches gauss_law_morph behavior.
    """
    if use_color:
        rgb = colorize_array(
            arr,
            palette=palette,
            contrast=contrast,
            gamma=gamma,
            clip_low_percent=clip_low_percent,
            clip_high_percent=clip_high_percent,
        )
        return Image.fromarray(rgb, mode='RGB')
    else:
        # Apply display transforms even in grayscale mode
        norm = _apply_display_transforms(
            arr,
            contrast=contrast,
            gamma=gamma,
            clip_low_percent=clip_low_percent,
            clip_high_percent=clip_high_percent,
        )
        img = (norm * 255.0).astype(np.uint8)
        return Image.fromarray(img, mode='L').convert('RGB')


def apply_gaussian_highpass(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Subtract Gaussian blur from array (returns highpass with unbounded range)."""
    arr = arr.astype(np.float32, copy=False)
    sigma = float(max(sigma, 0.0))
    if sigma <= 1e-6:
        return arr.copy()
    return arr - gaussian_filter(arr, sigma=sigma)


def downsample_lic(
    arr: np.ndarray,
    target_shape: tuple[int, int],
    supersample: float,
    sigma: float,
) -> np.ndarray:
    """Gaussian blur + bilinear resize from supersampled grid to target resolution."""
    sigma = max(sigma, 0.0)

    # Apply blur even if no resize needed
    filtered = gaussian_filter(arr, sigma=sigma) if sigma > 0 else arr

    # Skip resize if shapes already match
    if arr.shape == target_shape:
        return filtered.copy() if filtered is arr else filtered

    scale_y = target_shape[0] / filtered.shape[0]
    scale_x = target_shape[1] / filtered.shape[1]
    return zoom(filtered, (scale_y, scale_x), order=1)
