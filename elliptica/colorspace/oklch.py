"""OKLCH color space conversions.

Reference: https://bottosson.github.io/posts/oklab/

All functions accept numpy arrays or torch tensors.
GPU acceleration automatic when torch GPU tensors are passed.
"""

from math import pi
from . import _backend as B
from ._backend import Array

# === OKLab <-> Linear RGB matrices ===
# From Björn Ottosson's reference implementation

# Linear RGB -> LMS
_RGB_TO_LMS = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)

# LMS cube root -> OKLab
_LMS_TO_OKLAB = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)

# OKLab -> LMS cube root
_OKLAB_TO_LMS = (
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)

# LMS -> Linear RGB
_LMS_TO_RGB = (
    (+4.0767416621, -3.3077115913, +0.2309699292),
    (-1.2684380046, +2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, +1.7076147010),
)


# === Core Conversions ===

def oklch_to_oklab(L: Array, C: Array, H: Array) -> tuple[Array, Array, Array]:
    """OKLCH -> OKLab. H in degrees."""
    H_rad = H * (pi / 180)
    a = C * B.cos(H_rad)
    b = C * B.sin(H_rad)
    return L, a, b


def oklab_to_oklch(L: Array, a: Array, b: Array) -> tuple[Array, Array, Array]:
    """OKLab -> OKLCH. Returns H in degrees [0, 360)."""
    C = B.sqrt(a**2 + b**2)
    H_rad = B.atan2(b, a)
    H = H_rad * (180 / pi)
    # Wrap to [0, 360)
    H = H % 360
    return L, C, H


def oklab_to_linear_rgb(L: Array, a: Array, b: Array) -> tuple[Array, Array, Array]:
    """OKLab -> Linear RGB via LMS intermediate."""
    # OKLab -> LMS (cube root space)
    l_ = L + _OKLAB_TO_LMS[0][1] * a + _OKLAB_TO_LMS[0][2] * b
    m_ = L + _OKLAB_TO_LMS[1][1] * a + _OKLAB_TO_LMS[1][2] * b
    s_ = L + _OKLAB_TO_LMS[2][1] * a + _OKLAB_TO_LMS[2][2] * b

    # Cube to get LMS
    l, m, s = l_**3, m_**3, s_**3

    # LMS -> Linear RGB
    r = _LMS_TO_RGB[0][0]*l + _LMS_TO_RGB[0][1]*m + _LMS_TO_RGB[0][2]*s
    g = _LMS_TO_RGB[1][0]*l + _LMS_TO_RGB[1][1]*m + _LMS_TO_RGB[1][2]*s
    b = _LMS_TO_RGB[2][0]*l + _LMS_TO_RGB[2][1]*m + _LMS_TO_RGB[2][2]*s

    return r, g, b


def linear_rgb_to_oklab(r: Array, g: Array, b: Array) -> tuple[Array, Array, Array]:
    """Linear RGB -> OKLab via LMS intermediate."""
    # Linear RGB -> LMS
    l = _RGB_TO_LMS[0][0]*r + _RGB_TO_LMS[0][1]*g + _RGB_TO_LMS[0][2]*b
    m = _RGB_TO_LMS[1][0]*r + _RGB_TO_LMS[1][1]*g + _RGB_TO_LMS[1][2]*b
    s = _RGB_TO_LMS[2][0]*r + _RGB_TO_LMS[2][1]*g + _RGB_TO_LMS[2][2]*b

    # Cube root (sign-preserving for edge cases)
    l_, m_, s_ = B.cbrt(l), B.cbrt(m), B.cbrt(s)

    # LMS cube root -> OKLab
    L = _LMS_TO_OKLAB[0][0]*l_ + _LMS_TO_OKLAB[0][1]*m_ + _LMS_TO_OKLAB[0][2]*s_
    a = _LMS_TO_OKLAB[1][0]*l_ + _LMS_TO_OKLAB[1][1]*m_ + _LMS_TO_OKLAB[1][2]*s_
    b = _LMS_TO_OKLAB[2][0]*l_ + _LMS_TO_OKLAB[2][1]*m_ + _LMS_TO_OKLAB[2][2]*s_

    return L, a, b


def linear_to_srgb(x: Array) -> Array:
    """Linear RGB -> sRGB gamma encoding (per channel)."""
    threshold = 0.0031308
    low = x * 12.92
    high = 1.055 * B.pow(B.maximum(x, B.full_like(x, 1e-10)), 1/2.4) - 0.055
    return B.where(x <= threshold, low, high)


def srgb_to_linear(x: Array) -> Array:
    """sRGB -> Linear RGB gamma decoding (per channel)."""
    threshold = 0.04045
    low = x / 12.92
    high = B.pow((x + 0.055) / 1.055, 2.4)
    return B.where(x <= threshold, low, high)


# === Convenience Composites ===

def oklch_to_srgb(L: Array, C: Array, H: Array) -> Array:
    """OKLCH -> sRGB in one call.

    Args:
        L: Lightness (0-1)
        C: Chroma (0-~0.4)
        H: Hue in degrees (0-360)

    Returns:
        RGB array with shape (..., 3), values may be outside [0,1] if out of gamut
    """
    L_ok, a, b = oklch_to_oklab(L, C, H)
    r, g, b = oklab_to_linear_rgb(L_ok, a, b)
    r_srgb = linear_to_srgb(r)
    g_srgb = linear_to_srgb(g)
    b_srgb = linear_to_srgb(b)
    return B.stack([r_srgb, g_srgb, b_srgb], axis=-1)


def srgb_to_oklch(rgb: Array) -> tuple[Array, Array, Array]:
    """sRGB -> OKLCH.

    Args:
        rgb: RGB array with shape (..., 3), values in [0,1]

    Returns:
        (L, C, H) tuple
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)
    L, a, b = linear_rgb_to_oklab(r_lin, g_lin, b_lin)
    return oklab_to_oklch(L, a, b)


def oklch_to_linear_rgb(L: Array, C: Array, H: Array) -> tuple[Array, Array, Array]:
    """OKLCH -> Linear RGB (no gamma encoding)."""
    L_ok, a, b = oklch_to_oklab(L, C, H)
    return oklab_to_linear_rgb(L_ok, a, b)
