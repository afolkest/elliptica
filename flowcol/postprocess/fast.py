"""JIT-accelerated postprocessing functions using Numba."""

import numpy as np
import numba


@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def apply_contrast_gamma_jit(
    norm: np.ndarray,
    brightness: float,
    contrast: float,
    gamma: float,
) -> np.ndarray:
    """Apply brightness, contrast and gamma correction to normalized [0,1] array.

    Args:
        norm: Normalized array in [0, 1]
        brightness: Brightness adjustment (0.0 = no change, additive)
        contrast: Contrast multiplier
        gamma: Gamma exponent

    Returns:
        Transformed array in [0, 1]
    """
    h, w = norm.shape
    out = np.empty((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            val = norm[i, j]

            # Contrast adjustment
            if contrast != 1.0:
                val = (val - 0.5) * contrast + 0.5
                val = max(0.0, min(1.0, val))

            # Brightness adjustment (after contrast, before gamma)
            if brightness != 0.0:
                val = val + brightness
                val = max(0.0, min(1.0, val))

            # Gamma correction
            if gamma != 1.0:
                val = val ** gamma
                val = max(0.0, min(1.0, val))

            out[i, j] = val

    return out


@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def apply_palette_lut_jit(
    norm: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """Map normalized values through color LUT.

    Args:
        norm: Normalized array in [0, 1], shape (H, W)
        lut: Color lookup table, shape (N, 3), values in [0, 1]

    Returns:
        RGB uint8 array, shape (H, W, 3)
    """
    h, w = norm.shape
    lut_size = lut.shape[0]
    out = np.empty((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            val = norm[i, j]

            # Map to LUT index
            idx = int(val * (lut_size - 1))
            idx = max(0, min(lut_size - 1, idx))

            # Lookup and convert to uint8
            out[i, j, 0] = int(lut[idx, 0] * 255.0)
            out[i, j, 1] = int(lut[idx, 1] * 255.0)
            out[i, j, 2] = int(lut[idx, 2] * 255.0)

    return out


@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def grayscale_to_rgb_jit(
    norm: np.ndarray,
    brightness: float,
    contrast: float,
    gamma: float,
) -> np.ndarray:
    """Convert grayscale array to RGB with brightness/contrast/gamma.

    Args:
        norm: Normalized array in [0, 1], shape (H, W)
        brightness: Brightness adjustment (0.0 = no change, additive)
        contrast: Contrast multiplier
        gamma: Gamma exponent

    Returns:
        RGB uint8 array, shape (H, W, 3)
    """
    h, w = norm.shape
    out = np.empty((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            val = norm[i, j]

            # Contrast adjustment
            if contrast != 1.0:
                val = (val - 0.5) * contrast + 0.5
                val = max(0.0, min(1.0, val))

            # Brightness adjustment (after contrast, before gamma)
            if brightness != 0.0:
                val = val + brightness
                val = max(0.0, min(1.0, val))

            # Gamma correction
            if gamma != 1.0:
                val = val ** gamma
                val = max(0.0, min(1.0, val))

            # Convert to uint8
            gray = int(val * 255.0)
            out[i, j, 0] = gray
            out[i, j, 1] = gray
            out[i, j, 2] = gray

    return out
