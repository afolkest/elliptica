"""Global colorization functions for display rendering."""

from dataclasses import dataclass
import numpy as np
from flowcol.render import colorize_array, array_to_pil, _normalize_unit, _get_palette_lut
from flowcol.postprocess.fast import apply_contrast_gamma_jit, apply_palette_lut_jit, grayscale_to_rgb_jit


@dataclass
class ColorParams:
    """Pure color parameters for backend colorization functions.

    This type lives in the backend (flowcol/postprocess/) to avoid UI coupling.
    UI layer converts DisplaySettings -> ColorParams at the boundary.
    """
    clip_percent: float
    brightness: float
    contrast: float
    gamma: float
    color_enabled: bool
    palette: str


def build_base_rgb(scalar_array: np.ndarray, color_params: ColorParams, display_array_gpu=None) -> np.ndarray:
    """Apply global colorization to scalar LIC field (GPU or JIT-accelerated).

    Args:
        scalar_array: Grayscale LIC array (float32)
        color_params: ColorParams with color_enabled, palette, gamma, contrast, clip_percent, brightness
        display_array_gpu: Optional GPU tensor to use instead of scalar_array (faster!)

    Returns:
        RGB uint8 array ready for display/compositing
    """
    # Try GPU-accelerated path if tensor provided
    use_gpu = False
    if display_array_gpu is not None:
        try:
            from flowcol.gpu import GPUContext
            from flowcol.gpu.pipeline import build_base_rgb_gpu

            if GPUContext.is_available():
                use_gpu = True
        except Exception:
            pass

    if use_gpu:
        # GPU path - much faster!
        import time
        import torch
        start = time.time()

        lut_numpy = _get_palette_lut(color_params.palette) if color_params.color_enabled else None
        lut_gpu = None
        if lut_numpy is not None:
            lut_gpu = GPUContext.to_gpu(lut_numpy)

        rgb_gpu = build_base_rgb_gpu(
            display_array_gpu,
            color_params.clip_percent,
            color_params.brightness,
            color_params.contrast,
            color_params.gamma,
            color_params.color_enabled,
            lut_gpu,
        )

        # Convert to uint8 and download
        rgb_uint8_tensor = (rgb_gpu * 255.0).clamp(0, 255).to(torch.uint8)
        torch.mps.synchronize()  # Wait for GPU work to complete
        result = GPUContext.to_cpu(rgb_uint8_tensor)

        # Uncomment for performance debugging:
        # elapsed = time.time() - start
        # print(f"🎨 GPU colorization: {elapsed*1000:.1f}ms")
        return result
    else:
        # CPU fallback (original code)
        arr = scalar_array.astype(np.float32, copy=False)

        # Clip/normalize to [0, 1]
        if color_params.clip_percent > 0.0:
            vmin = float(np.percentile(arr, color_params.clip_percent))
            vmax = float(np.percentile(arr, 100.0 - color_params.clip_percent))
            if vmax > vmin:
                norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
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


def _blend_region(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend overlay over base using mask as alpha.

    Args:
        base: Base RGB uint8 array (H, W, 3)
        overlay: Overlay RGB uint8 array (H, W, 3)
        mask: Float mask (H, W) in [0, 1]

    Returns:
        Blended RGB uint8 array
    """
    alpha = mask[..., None].astype(np.float32)
    base_f = base.astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    blended = base_f * (1.0 - alpha) + overlay_f * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _fill_region(base: np.ndarray, color_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill region with solid color using mask.

    Args:
        base: Base RGB uint8 array (H, W, 3)
        color_uint8: Solid color as uint8 array (3,)
        mask: Float mask (H, W) in [0, 1]

    Returns:
        Filled RGB uint8 array
    """
    result = base.copy()
    alpha = mask[..., None].astype(np.float32)
    base_f = base.astype(np.float32)
    color_f = color_uint8.astype(np.float32)
    filled = base_f * (1.0 - alpha) + color_f * alpha
    return np.clip(filled, 0, 255).astype(np.uint8)


def apply_region_overlays(
    base_rgb: np.ndarray,
    scalar_array: np.ndarray,
    conductor_masks: list[np.ndarray],
    interior_masks: list[np.ndarray],
    conductor_color_settings: dict,
    conductors: list,
    color_params: ColorParams,
) -> np.ndarray:
    """Composite per-region color overrides over base RGB.

    For each conductor:
    - If interior region style enabled: apply it (palette or solid)
    - If surface region style enabled: apply it (palette or solid)

    Layers from bottom to top:
    1. Base RGB (global)
    2. Interior regions (where enabled)
    3. Surface regions (where enabled)

    Args:
        base_rgb: Base RGB uint8 array (from global colorization)
        scalar_array: Original scalar LIC array
        conductor_masks: List of surface masks at display resolution
        interior_masks: List of interior masks at display resolution
        conductor_color_settings: dict[conductor_id -> ConductorColorSettings]
        conductors: List of Conductor objects
        color_params: ColorParams for gamma/contrast/clip/brightness

    Returns:
        Final composited RGB uint8 array
    """
    result = base_rgb.copy()

    for idx, conductor in enumerate(conductors):
        if conductor.id is None:
            continue

        settings = conductor_color_settings.get(conductor.id)
        if settings is None:
            continue

        # Apply interior region first (lower layer)
        if settings.interior.enabled and idx < len(interior_masks) and interior_masks[idx] is not None:
            mask = interior_masks[idx]
            if np.any(mask > 0):
                if settings.interior.use_palette:
                    # Re-colorize scalar array in this region
                    region_rgb = colorize_array(
                        scalar_array,
                        palette=settings.interior.palette,
                        brightness=color_params.brightness,
                        gamma=color_params.gamma,
                        contrast=color_params.contrast,
                        clip_percent=color_params.clip_percent,
                    )
                    result = _blend_region(result, region_rgb, mask)
                else:
                    # Solid color fill
                    color_uint8 = (np.array(settings.interior.solid_color) * 255).astype(np.uint8)
                    result = _fill_region(result, color_uint8, mask)

        # Apply surface region (upper layer)
        if settings.surface.enabled and idx < len(conductor_masks) and conductor_masks[idx] is not None:
            mask = conductor_masks[idx]
            if np.any(mask > 0):
                if settings.surface.use_palette:
                    # Re-colorize scalar array in this region
                    region_rgb = colorize_array(
                        scalar_array,
                        palette=settings.surface.palette,
                        brightness=color_params.brightness,
                        gamma=color_params.gamma,
                        contrast=color_params.contrast,
                        clip_percent=color_params.clip_percent,
                    )
                    result = _blend_region(result, region_rgb, mask)
                else:
                    # Solid color fill
                    color_uint8 = (np.array(settings.surface.solid_color) * 255).astype(np.uint8)
                    result = _fill_region(result, color_uint8, mask)

    return result
