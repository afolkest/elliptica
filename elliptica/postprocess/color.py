"""Global colorization functions for display rendering."""

from dataclasses import dataclass
import time
import numpy as np
import torch

from elliptica.gpu import GPUContext
from elliptica.gpu.pipeline import build_base_rgb_gpu
from elliptica.render import colorize_array, array_to_pil, _normalize_unit, _get_palette_lut


@dataclass
class ColorParams:
    """Pure color parameters for backend colorization functions.

    This type lives in the backend (elliptica/postprocess/) to avoid UI coupling.
    UI layer converts DisplaySettings -> ColorParams at the boundary.
    """
    clip_low_percent: float
    clip_high_percent: float
    brightness: float
    contrast: float
    gamma: float
    color_enabled: bool
    palette: str


def build_base_rgb(scalar_array: np.ndarray, color_params: ColorParams, display_array_gpu=None) -> np.ndarray:
    """Apply global colorization to scalar LIC field (GPU or JIT-accelerated).

    Args:
        scalar_array: Grayscale LIC array (float32)
        color_params: ColorParams with color_enabled, palette, gamma, contrast, clip range, brightness
        display_array_gpu: Optional GPU tensor to use instead of scalar_array (faster!)

    Returns:
        RGB uint8 array ready for display/compositing
    """
    # Try GPU-accelerated path if tensor provided
    use_gpu = False
    if display_array_gpu is not None:
        if GPUContext.is_available():
            use_gpu = True

    if use_gpu:
        # GPU path - much faster!
        start = time.time()

        lut_numpy = _get_palette_lut(color_params.palette) if color_params.color_enabled else None
        lut_gpu = None
        if lut_numpy is not None:
            lut_gpu = GPUContext.to_gpu(lut_numpy)

        rgb_gpu = build_base_rgb_gpu(
            display_array_gpu,
            color_params.clip_low_percent,
            color_params.clip_high_percent,
            color_params.brightness,
            color_params.contrast,
            color_params.gamma,
            color_params.color_enabled,
            lut_gpu,
        )

        # Convert to uint8 and download
        rgb_uint8_tensor = (rgb_gpu * 255.0).clamp(0, 255).to(torch.uint8)

        # Synchronize GPU (platform-specific)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        result = GPUContext.to_cpu(rgb_uint8_tensor)

        # Uncomment for performance debugging:
        # elapsed = time.time() - start
        # print(f"🎨 GPU colorization: {elapsed*1000:.1f}ms")
        return result
    else:
        # CPU fallback (original code)
        arr = scalar_array.astype(np.float32, copy=False)

        # Clip/normalize to [0, 1]
        low = max(float(color_params.clip_low_percent), 0.0)
        high = max(float(color_params.clip_high_percent), 0.0)
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

        if not color_params.color_enabled:
            # Grayscale mode with JIT-accelerated transforms
            rgb = grayscale_to_rgb_jit(norm, color_params.brightness, color_params.contrast, color_params.gamma)
            return rgb
        else:
            # Color mode: apply contrast/gamma, then LUT
            norm_adjusted = apply_contrast_gamma_jit(norm, color_params.brightness, color_params.contrast, color_params.gamma)
            lut = _get_palette_lut(color_params.palette)
            rgb = apply_palette_lut_jit(norm_adjusted, lut)
            return rgb


# REMOVED: apply_region_overlays() and helper functions - replaced by unified GPU/CPU version in gpu/overlay.py
# The GPU implementation works on both GPU (device='mps'/'cuda') and CPU (device='cpu')
# using PyTorch operations, eliminating the need for these NumPy-based duplicates:
# - _blend_region()
# - _fill_region()
# - apply_region_overlays()
