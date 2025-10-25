"""Global colorization functions for display rendering."""

import numpy as np
from flowcol.render import colorize_array, array_to_pil


def build_base_rgb(scalar_array: np.ndarray, settings) -> np.ndarray:
    """Apply global colorization to scalar LIC field.

    Args:
        scalar_array: Grayscale LIC array (float32)
        settings: DisplaySettings with color_enabled, palette, gamma, contrast, clip_percent

    Returns:
        RGB uint8 array ready for display/compositing
    """
    if not settings.color_enabled:
        # Grayscale mode with display transforms
        pil_img = array_to_pil(
            scalar_array,
            use_color=False,
            gamma=settings.gamma,
            contrast=settings.contrast,
            clip_percent=settings.clip_percent,
        )
        # Convert back to numpy RGB array
        return np.array(pil_img, dtype=np.uint8)
    else:
        # Colorized mode
        rgb = colorize_array(
            scalar_array,
            palette=settings.palette,
            gamma=settings.gamma,
            contrast=settings.contrast,
            clip_percent=settings.clip_percent,
        )
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
    display_settings,
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
        display_settings: DisplaySettings for gamma/contrast/clip

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
                        gamma=display_settings.gamma,
                        contrast=display_settings.contrast,
                        clip_percent=display_settings.clip_percent,
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
                        gamma=display_settings.gamma,
                        contrast=display_settings.contrast,
                        clip_percent=display_settings.clip_percent,
                    )
                    result = _blend_region(result, region_rgb, mask)
                else:
                    # Solid color fill
                    color_uint8 = (np.array(settings.surface.solid_color) * 255).astype(np.uint8)
                    result = _fill_region(result, color_uint8, mask)

    return result
