"""
Relaxation utilities for PDE solutions.
"""

import numba
import numpy as np
from scipy.ndimage import binary_dilation


@numba.njit(cache=True)
def relax_potential_band(phi: np.ndarray, relax_mask: np.ndarray, iterations: int, omega: float) -> None:
    """
    Apply Gauss-Seidel relaxation to a specific band of pixels.
    
    Used to smooth out interpolation artifacts near boundaries when upscaling
    low-resolution preview solutions.
    """
    height, width = phi.shape
    for it in range(iterations):
        parity = it & 1
        for i in range(height):
            for j in range(width):
                if not relax_mask[i, j]:
                    continue
                if ((i + j) & 1) != parity:
                    continue

                sum_val = 0.0
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < height and 0 <= jj < width:
                        sum_val += phi[ii, jj]
                    else:
                        sum_val += phi[i, j]
                avg = 0.25 * sum_val
                phi[i, j] = (1.0 - omega) * phi[i, j] + omega * avg


def build_relaxation_mask(dirichlet_mask: np.ndarray, band_width: int) -> np.ndarray:
    """
    Build a mask of pixels to relax, surrounding the Dirichlet boundaries.
    """
    if band_width <= 0:
        return np.zeros_like(dirichlet_mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(dirichlet_mask, structure=structure, iterations=band_width)
    return np.logical_and(dilated, ~dirichlet_mask)
