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
