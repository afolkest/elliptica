"""Gamut mapping for out-of-gamut OKLCH values.

Not all (L, C, H) combinations produce valid sRGB. High chroma at
extreme lightness is particularly problematic.

Strategies:
- clip: Hard-clip RGB to [0,1] — fast but can shift hue/lightness
- compress: Reduce C until in-gamut — preserves L and H intent
"""

import numpy as np
from typing import Literal

from . import _backend as B
from ._backend import Array
from .oklch import oklch_to_srgb, oklch_to_linear_rgb, srgb_to_oklch

# === Gamut checking ===

def is_in_gamut(L: Array, C: Array, H: Array, tolerance: float = 1e-4) -> Array:
    """Check if OKLCH values produce valid sRGB (all channels in [0,1])."""
    rgb = oklch_to_srgb(L, C, H)
    in_range = (rgb >= -tolerance) & (rgb <= 1 + tolerance)
    return B.all_along_axis(in_range, axis=-1)


# === Gamut mapping methods ===

def gamut_clip(L: Array, C: Array, H: Array) -> Array:
    """Convert to sRGB and hard-clip to [0,1].

    Fast but may distort colors (hue shifts, flattened gradients).

    Returns:
        RGB array (..., 3) with values clamped to [0,1]
    """
    rgb = oklch_to_srgb(L, C, H)
    return B.clip(rgb, 0.0, 1.0)


def gamut_compress(
    L: Array,
    C: Array,
    H: Array,
    method: Literal['clip', 'chroma'] = 'chroma'
) -> tuple[Array, Array, Array]:
    """Bring out-of-gamut colors into sRGB gamut.

    Args:
        L, C, H: OKLCH values
        method: 'clip' for RGB clipping, 'chroma' for chroma reduction

    Returns:
        (L, C, H) tuple with adjusted values
    """
    if method == 'clip':
        rgb_clipped = gamut_clip(L, C, H)
        return srgb_to_oklch(rgb_clipped)

    elif method == 'chroma':
        max_C = max_chroma_for_lh(L, H)
        C_compressed = B.minimum(C, max_C)
        return L, C_compressed, H

    raise ValueError(f"Unknown gamut method: {method}")


# === Max chroma computation ===

def max_chroma_for_lh(L: Array, H: Array, steps: int = 16) -> Array:
    """Find maximum valid chroma for given L and H via binary search.

    For real-time use, prefer max_chroma_fast() which uses a precomputed LUT.
    """
    lo = B.zeros_like(L)
    hi = B.full_like(L, 0.5)  # 0.5 is always out of gamut

    for _ in range(steps):
        mid = (lo + hi) / 2
        valid = is_in_gamut(L, mid, H)
        lo = B.where(valid, mid, lo)
        hi = B.where(~valid, mid, hi)

    return lo


# === Precomputed LUT for fast gamut mapping ===

_MAX_CHROMA_LUT: np.ndarray | None = None
_LUT_L_STEPS = 256
_LUT_H_STEPS = 360


def _build_max_chroma_lut() -> np.ndarray:
    """Build max chroma lookup table. Called once on first use."""
    L = np.linspace(0, 1, _LUT_L_STEPS)[:, None]
    H = np.arange(_LUT_H_STEPS, dtype=np.float32)[None, :]

    L_grid = np.broadcast_to(L, (_LUT_L_STEPS, _LUT_H_STEPS)).astype(np.float32)
    H_grid = np.broadcast_to(H, (_LUT_L_STEPS, _LUT_H_STEPS)).astype(np.float32)

    # Flatten, compute, reshape
    lut = max_chroma_for_lh(
        L_grid.ravel(),
        H_grid.ravel(),
        steps=20
    ).reshape(_LUT_L_STEPS, _LUT_H_STEPS)

    return lut.astype(np.float32)


def get_max_chroma_lut() -> np.ndarray:
    """Get or build the max chroma LUT."""
    global _MAX_CHROMA_LUT
    if _MAX_CHROMA_LUT is None:
        _MAX_CHROMA_LUT = _build_max_chroma_lut()
    return _MAX_CHROMA_LUT


def max_chroma_fast(L: Array, H: Array) -> Array:
    """Fast max chroma lookup via LUT + bilinear interpolation.

    Much faster than binary search for large arrays.
    """
    lut = get_max_chroma_lut()

    # Convert to numpy for interpolation
    L_np = B.to_numpy(L)
    H_np = B.to_numpy(H)

    # Map to LUT indices
    L_idx = np.clip(L_np * (_LUT_L_STEPS - 1), 0, _LUT_L_STEPS - 1)
    H_idx = H_np % 360

    # Bilinear interpolation indices
    L_lo = np.floor(L_idx).astype(int)
    L_hi = np.minimum(L_lo + 1, _LUT_L_STEPS - 1)
    L_frac = L_idx - L_lo

    H_lo = np.floor(H_idx).astype(int) % _LUT_H_STEPS
    H_hi = (H_lo + 1) % _LUT_H_STEPS
    H_frac = H_idx - np.floor(H_idx)

    # Bilinear interpolation
    c00 = lut[L_lo, H_lo]
    c01 = lut[L_lo, H_hi]
    c10 = lut[L_hi, H_lo]
    c11 = lut[L_hi, H_hi]

    c0 = c00 * (1 - H_frac) + c01 * H_frac
    c1 = c10 * (1 - H_frac) + c11 * H_frac
    result = c0 * (1 - L_frac) + c1 * L_frac

    return B.from_numpy(result.astype(np.float32), L)


def gamut_compress_fast(L: Array, C: Array, H: Array) -> tuple[Array, Array, Array]:
    """Fast gamut compression using precomputed LUT.

    Reduces chroma to stay in gamut while preserving L and H.
    """
    max_C = max_chroma_fast(L, H)
    C_compressed = B.minimum(C, max_C)
    return L, C_compressed, H


def gamut_map_to_srgb(
    L: Array,
    C: Array,
    H: Array,
    method: Literal['clip', 'compress'] = 'compress'
) -> Array:
    """Map OKLCH to sRGB with gamut handling.

    This is the main entry point for OKLCH -> sRGB conversion with gamut safety.

    Args:
        L: Lightness (0-1)
        C: Chroma (0-~0.4)
        H: Hue degrees (0-360)
        method: 'clip' for fast RGB clipping, 'compress' for chroma reduction

    Returns:
        RGB array (..., 3) with values in [0, 1]
    """
    if method == 'clip':
        return gamut_clip(L, C, H)

    elif method == 'compress':
        L_safe, C_safe, H_safe = gamut_compress_fast(L, C, H)
        rgb = oklch_to_srgb(L_safe, C_safe, H_safe)
        # Final clip for numerical safety
        return B.clip(rgb, 0.0, 1.0)

    raise ValueError(f"Unknown method: {method}")
