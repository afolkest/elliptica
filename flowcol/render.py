import numpy as np
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
    "Twilight Peach": np.array(
        [
            (0.15, 0.05, 0.30),
            (0.40, 0.20, 0.55),
            (0.65, 0.45, 0.75),
            (0.90, 0.75, 0.85),
            (1.00, 0.95, 0.90),
            (0.95, 0.80, 0.65),
            (0.85, 0.60, 0.45),
            (0.65, 0.40, 0.35),
            (0.35, 0.20, 0.30),
        ],
        dtype=np.float32,
    ),
    "Twilight Magenta": np.array(
        [
            (0.18, 0.07, 0.22),  # Dark purple
            (0.26, 0.11, 0.35),  # Deep purple
            (0.33, 0.17, 0.48),  # Purple
            (0.40, 0.30, 0.60),  # Medium purple
            (0.50, 0.45, 0.70),  # Light purple
            (0.65, 0.60, 0.80),  # Lavender
            (0.80, 0.75, 0.88),  # Very light lavender
            (0.92, 0.88, 0.94),  # Near white
            (0.95, 0.90, 0.85),  # Peachy white
            (0.88, 0.70, 0.55),  # Peach/gold
            (0.75, 0.50, 0.40),  # Orange-brown
            (0.50, 0.25, 0.35),  # Purple-brown
            (0.35, 0.15, 0.30),  # Dark purple-red
            (0.18, 0.08, 0.21),  # Dark purple (wrap)
        ],
        dtype=np.float32,
    ),
    "Fire & Ice": np.array(
        [
            (0.10, 0.10, 0.50),
            (0.20, 0.20, 0.70),
            (0.40, 0.40, 0.90),
            (0.70, 0.70, 1.00),
            (1.00, 1.00, 1.00),
            (1.00, 0.90, 0.70),
            (1.00, 0.70, 0.40),
            (0.90, 0.40, 0.20),
            (0.70, 0.20, 0.10),
        ],
        dtype=np.float32,
    ),
    "Stark B&W": np.array(
        [
            (0.00, 0.00, 0.00),  # pure black
            (0.00, 0.00, 0.00),  # black
            (0.15, 0.15, 0.15),  # dark gray
            (1.00, 1.00, 1.00),  # white
            (1.00, 1.00, 1.00),  # white
            (0.15, 0.15, 0.15),  # dark gray
            (0.00, 0.00, 0.00),  # black
        ],
        dtype=np.float32,
    ),
    "Ink Wash": np.array(
        [
            (0.00, 0.00, 0.00),  # pure black
            (0.00, 0.00, 0.00),  # pure black
            (0.03, 0.03, 0.03),  # very dark
            (0.08, 0.08, 0.08),  # dark
            (0.90, 0.90, 0.90),  # light
            (0.96, 0.96, 0.96),  # near white
            (1.00, 1.00, 1.00),  # pure white
        ],
        dtype=np.float32,
    ),
    "Inky Washy": np.array(
        [
            (0.00, 0.00, 0.00),  # pure black
            (0.00, 0.00, 0.00),  # pure black
            (0.03, 0.03, 0.03),  # very dark
            (0.08, 0.08, 0.08),  # dark
            (0.99, 0.99, 0.99),  # light
            (0.00, 0.00, 0.00),  # near white
            (0.00, 0.00, 0.00),  # pure white
        ],
        dtype=np.float32,
    ),
    "PuOr": np.array(
        [
            (0.498, 0.231, 0.031),  # Dark orange
            (0.702, 0.345, 0.024),  # Orange
            (0.878, 0.510, 0.078),  # Light orange
            (0.992, 0.722, 0.388),  # Pale orange
            (0.969, 0.969, 0.969),  # White
            (0.847, 0.855, 0.922),  # Pale purple
            (0.698, 0.671, 0.824),  # Light purple
            (0.502, 0.451, 0.675),  # Purple
            (0.329, 0.153, 0.533),  # Dark purple
        ],
        dtype=np.float32,
    ),
    "Coolwarm": np.array(
        [
            (0.231, 0.298, 0.753),  # Cool blue
            (0.404, 0.475, 0.859),  # Light blue
            (0.608, 0.667, 0.925),  # Pale blue
            (0.780, 0.839, 0.965),  # Very pale blue
            (0.933, 0.933, 0.933),  # White/gray
            (0.969, 0.792, 0.729),  # Pale red
            (0.933, 0.573, 0.490),  # Light red
            (0.843, 0.329, 0.333),  # Red
            (0.706, 0.016, 0.149),  # Dark red
        ],
        dtype=np.float32,
    ),
    "Aurora": np.array(
        [
            (0.05, 0.14, 0.14),
            (0.10, 0.24, 0.24),
            (0.22, 0.40, 0.40),
            (0.36, 0.52, 0.58),
            (0.50, 0.56, 0.72),
            (0.64, 0.52, 0.80),
            (0.76, 0.60, 0.86),
            (0.86, 0.72, 0.90),
            (0.94, 0.86, 0.96),
        ],
        dtype=np.float32,
    ),
    "Bronze Teal": np.array(
        [
            (0.08, 0.05, 0.02),
            (0.16, 0.10, 0.05),
            (0.32, 0.20, 0.10),
            (0.56, 0.36, 0.14),
            (0.78, 0.56, 0.22),
            (0.56, 0.66, 0.54),
            (0.36, 0.68, 0.64),
            (0.22, 0.58, 0.58),
            (0.14, 0.44, 0.46),
        ],
        dtype=np.float32,
    ),
    "Electric": np.array(
        [
            (0.00, 0.00, 0.00),
            (0.10, 0.00, 0.20),
            (0.20, 0.00, 0.40),
            (0.40, 0.10, 0.60),
            (0.60, 0.30, 0.80),
            (0.80, 0.60, 0.90),
            (0.90, 0.80, 0.95),
            (0.95, 0.90, 1.00),
            (1.00, 1.00, 1.00),
        ],
        dtype=np.float32,
    ),
    "Jade Lavender Gold": np.array(
        [
            (0.02, 0.22, 0.18),
            (0.08, 0.44, 0.34),
            (0.26, 0.66, 0.52),
            (0.86, 0.80, 0.96),
            (0.76, 0.66, 0.90),
            (0.96, 0.84, 0.46),
            (1.00, 0.90, 0.62),
        ],
        dtype=np.float32,
    ),
    "Seafoam Sunset Plum": np.array(
        [
            (0.06, 0.24, 0.22),
            (0.22, 0.60, 0.58),
            (0.58, 0.90, 0.86),
            (1.00, 0.70, 0.50),
            (0.98, 0.54, 0.38),
            (0.70, 0.52, 0.78),
            (0.52, 0.36, 0.64),
        ],
        dtype=np.float32,
    ),
    "Viridian Ochre Cerulean": np.array(
        [
            (0.02, 0.16, 0.12),
            (0.06, 0.32, 0.26),
            (0.10, 0.50, 0.40),
            (0.86, 0.72, 0.26),
            (0.98, 0.84, 0.42),
            (0.20, 0.48, 0.74),
            (0.10, 0.34, 0.60),
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

    if lowpass_sigma > 0.0:
        base = gaussian_filter(base, sigma=lowpass_sigma)

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
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    clip_percent: float = 0.5,
) -> np.ndarray:
    """Map scalar field to RGB using the provided LUT.

    Args:
        arr: Input array to colorize
        palette: Color palette name
        brightness: Brightness adjustment (0.0 = no change, additive)
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
    if not np.isclose(brightness, 0.0):
        norm = np.clip(norm + brightness, 0.0, 1.0)
    gamma = max(float(gamma), 1e-3)
    if not np.isclose(gamma, 1.0):
        norm = np.power(norm, gamma, dtype=np.float32)
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
    clip_percent: float = 0.5,
) -> np.ndarray:
    """Apply clip/brightness/contrast/gamma transforms to normalize array to [0,1]."""
    arr = arr.astype(np.float32, copy=False)

    # Percentile-based normalization (clipping)
    if clip_percent > 0.0:
        vmin = float(np.percentile(arr, clip_percent))
        vmax = float(np.percentile(arr, 100.0 - clip_percent))
        if vmax > vmin:
            norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
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
        norm = np.power(norm, gamma, dtype=np.float32)

    return np.clip(norm, 0.0, 1.0)


def array_to_pil(
    arr: np.ndarray,
    *,
    use_color: bool = False,
    palette: str | None = None,
    contrast: float = 1.0,
    gamma: float = 1.0,
    clip_percent: float = 0.5,
) -> Image.Image:
    """Convert scalar array to PIL Image, optionally colorized.

    Framework-agnostic version for Streamlit, headless rendering, etc.

    Note: Default clip_percent=0.5 matches gauss_law_morph behavior.
    """
    if use_color:
        rgb = colorize_array(arr, palette=palette, contrast=contrast, gamma=gamma, clip_percent=clip_percent)
        return Image.fromarray(rgb, mode='RGB')
    else:
        # Apply display transforms even in grayscale mode
        norm = _apply_display_transforms(arr, contrast=contrast, gamma=gamma, clip_percent=clip_percent)
        img = (norm * 255.0).astype(np.uint8)
        return Image.fromarray(img, mode='L').convert('RGB')


def array_to_surface(
    arr: np.ndarray,
    *,
    use_color: bool = False,
    palette: str | None = None,
    contrast: float = 1.0,
    gamma: float = 1.0,
    clip_percent: float = 0.5,
):
    """Convert scalar array to pygame surface, optionally colorized.

    Note: UI always passes clip_percent explicitly from state (defaults to 0.0 there).
          This default of 0.5 is for direct API usage to match gauss_law_morph behavior.
    """
    import pygame
    pil_img = array_to_pil(arr, use_color=use_color, palette=palette, contrast=contrast, gamma=gamma, clip_percent=clip_percent)
    return pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode)


def save_render(
    arr: np.ndarray,
    project: Project,
    multiplier: float,
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

    pil_img = array_to_pil(arr, use_color=use_color, palette=palette, contrast=contrast, gamma=gamma, clip_percent=clip_percent)
    pil_img.save(filepath)

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
    sigma = max(sigma, 0.0)

    # Apply blur even if no resize needed
    filtered = gaussian_filter(arr, sigma=sigma) if sigma > 0 else arr

    # Skip resize if shapes already match
    if arr.shape == target_shape:
        return filtered.copy() if filtered is arr else filtered

    scale_y = target_shape[0] / filtered.shape[0]
    scale_x = target_shape[1] / filtered.shape[1]
    return zoom(filtered, (scale_y, scale_x), order=1)


def apply_conductor_smear(
    rgb: np.ndarray,
    lic_gray: np.ndarray,
    project,
    palette_name: str | None,
    render_shape: tuple[int, int],
    color_enabled: bool = True,
    lic_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply smear effect to texture inside conductor masks.

    Blurs the LIC grayscale, re-normalizes it to create smooth gradients,
    then applies color (if enabled) or grayscale. This creates the characteristic
    "melted blob" appearance.

    Args:
        rgb: Current RGB image (H, W, 3) uint8
        lic_gray: Original LIC grayscale (H, W) float32
        project: Project containing conductors
        palette_name: Color palette to use if color_enabled is True
        render_shape: (height, width) of render resolution
        color_enabled: Whether to apply color palette or use grayscale
        lic_percentiles: Precomputed (vmin, vmax) for normalization, or None to compute

    Returns:
        Modified RGB image with smear applied
    """

    # Convert to float for blending
    out = rgb.astype(np.float32) / 255.0

    render_h, render_w = render_shape
    canvas_w, canvas_h = project.canvas_resolution
    scale_x = render_w / canvas_w
    scale_y = render_h / canvas_h

    # Process each conductor with smear enabled
    for conductor in project.conductors:
        if not conductor.smear_enabled:
            continue

        # Build conductor mask at render resolution
        x = conductor.position[0] * scale_x
        y = conductor.position[1] * scale_y

        # Scale mask to render resolution
        if not np.isclose(scale_x, 1.0) or not np.isclose(scale_y, 1.0):
            scaled_mask = zoom(conductor.mask, (scale_y, scale_x), order=1)
        else:
            scaled_mask = conductor.mask.copy()

        mask_h, mask_w = scaled_mask.shape

        # Place mask in render coordinates
        ix, iy = int(round(x)), int(round(y))
        x0, y0 = max(0, ix), max(0, iy)
        x1, y1 = min(ix + mask_w, render_w), min(iy + mask_h, render_h)

        mx0, my0 = max(0, -ix), max(0, -iy)
        mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

        if x1 <= x0 or y1 <= y0:
            continue

        mask_slice = scaled_mask[my0:my1, mx0:mx1]

        # Create full mask for this conductor
        full_mask = np.zeros((render_h, render_w), dtype=np.float32)
        full_mask[y0:y1, x0:x1] = mask_slice
        mask_bool = full_mask > 0.5

        if not np.any(mask_bool):
            continue

        # Blur LIC grayscale globally (to avoid boundary artifacts)
        sigma_px = max(conductor.smear_sigma, 0.1)
        lic_blur = gaussian_filter(lic_gray.astype(np.float32), sigma=sigma_px)

        # Re-normalize and colorize (creates "melted blob" effect)
        # Use precomputed percentiles if available, otherwise compute on-the-fly
        if color_enabled and palette_name:
            rgb_blur = colorize_array(lic_blur, palette=palette_name).astype(np.float32) / 255.0
        else:
            # Grayscale: normalize to [0, 1] and broadcast to RGB
            arr = lic_blur.astype(np.float32)
            if lic_percentiles is not None:
                vmin, vmax = lic_percentiles
            else:
                vmin = float(np.percentile(arr, 0.5))
                vmax = float(np.percentile(arr, 99.5))

            if vmax > vmin:
                norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                norm = np.clip((arr - arr.min()) / (arr.max() - arr.min() + 1e-10), 0.0, 1.0)
            rgb_blur = np.stack([norm, norm, norm], axis=-1)

        # Apply smear at full strength inside mask (no distance-based feathering)
        # This is GPU-friendly and creates clean "melted blob" effect
        weight = full_mask[..., None]  # Broadcast mask to RGB channels
        out = out * (1.0 - weight) + rgb_blur * weight

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
