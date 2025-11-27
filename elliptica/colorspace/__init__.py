"""OKLCH color space conversions and gamut mapping.

This module provides:
- OKLCH <-> sRGB conversions
- Gamut mapping (clip or chroma compression)
- Backend-agnostic: works with numpy arrays or torch tensors

Example:
    import numpy as np
    from flowcol.colorspace import oklch_to_srgb, gamut_map_to_srgb

    # Create OKLCH values
    L = np.full((100, 100), 0.7)
    C = np.full((100, 100), 0.15)
    H = np.linspace(0, 360, 100)[None, :] * np.ones((100, 1))

    # Convert to sRGB with gamut handling
    rgb = gamut_map_to_srgb(L, C, H, method='compress')
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

__all__ = [
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
