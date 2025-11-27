"""OKLCH color space conversions, gamut mapping, and expression-based color mapping.

This module provides:
- OKLCH <-> sRGB conversions
- Gamut mapping (clip or chroma compression)
- ColorMapping: map field arrays to RGB via OKLCH expressions
- Backend-agnostic: works with numpy arrays or torch tensors

Example:
    import numpy as np
    from elliptica.colorspace import ColorMapping

    # Define how fields map to color channels
    mapping = ColorMapping(
        L="0.3 + 0.5 * normalize(lic)",
        C="0.15 * normalize(mag)",
        H="180",
    )

    # Render to RGB
    rgb = mapping.render({'lic': lic_array, 'mag': mag_array})
"""

from .oklch import (
    oklch_to_oklab,
    oklab_to_oklch,
    oklab_to_linear_rgb,
    linear_rgb_to_oklab,
    linear_to_srgb,
    srgb_to_linear,
    oklch_to_srgb,
    srgb_to_oklch,
    oklch_to_linear_rgb,
)

from .gamut import (
    is_in_gamut,
    gamut_clip,
    gamut_compress,
    gamut_compress_fast,
    gamut_map_to_srgb,
    max_chroma_for_lh,
    max_chroma_fast,
    get_max_chroma_lut,
)

from .mapping import ColorMapping, ColorConfig

__all__ = [
    # High-level API
    'ColorMapping',
    'ColorConfig',
    # OKLCH conversions
    'oklch_to_oklab',
    'oklab_to_oklch',
    'oklab_to_linear_rgb',
    'linear_rgb_to_oklab',
    'linear_to_srgb',
    'srgb_to_linear',
    'oklch_to_srgb',
    'srgb_to_oklch',
    'oklch_to_linear_rgb',
    # Gamut mapping
    'is_in_gamut',
    'gamut_clip',
    'gamut_compress',
    'gamut_compress_fast',
    'gamut_map_to_srgb',
    'max_chroma_for_lh',
    'max_chroma_fast',
    'get_max_chroma_lut',
]
