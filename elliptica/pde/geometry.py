"""
Geometry utilities for PDE boundary handling.

This module provides reusable functions for computing geometric properties
of arbitrary-shaped boundaries, such as normal vectors at boundary pixels.
These utilities are PDE-agnostic and can be used by any solver that needs
to handle Neumann (flux) boundary conditions on interior objects.
"""

import numpy as np
from scipy.ndimage import binary_dilation, convolve


def find_boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Find pixels that are on the boundary of a mask (inside mask, adjacent to outside).

    Args:
        mask: Boolean 2D array where True = inside the object

    Returns:
        Boolean 2D array where True = boundary pixel
    """
    # A boundary pixel is one that is inside the mask but has at least one
    # neighbor outside the mask
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.uint8)

    # Count neighbors that are outside the mask
    outside_neighbor_count = convolve((~mask).astype(np.uint8), kernel, mode='constant', cval=1)

    # Boundary = inside mask AND has at least one outside neighbor
    return mask & (outside_neighbor_count > 0)


def compute_boundary_normals(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute outward-pointing unit normal vectors at each pixel of a mask.

    Uses the gradient of the mask (treated as a level set) to estimate normals.
    The normal points from inside the mask toward outside.

    Args:
        mask: Boolean 2D array where True = inside the object

    Returns:
        Tuple of (nx, ny) arrays with the same shape as mask.
        At boundary pixels, these are unit normals pointing outward.
        At non-boundary pixels, values are 0.
    """
    # Convert mask to float for gradient computation
    # We want normals pointing outward (from True region toward False)
    mask_float = mask.astype(np.float64)

    # Compute gradient of the mask (approximates the level set gradient)
    # np.gradient returns (dy, dx) by default
    gy, gx = np.gradient(mask_float)

    # The gradient points from low (False=0) to high (True=1), i.e., inward.
    # We want outward normals, so negate.
    nx = -gx
    ny = -gy

    # Normalize to unit length
    magnitude = np.sqrt(nx**2 + ny**2)
    # Avoid division by zero
    magnitude = np.maximum(magnitude, 1e-10)
    nx = nx / magnitude
    ny = ny / magnitude

    # Zero out normals at non-boundary pixels (where mask is False or interior)
    boundary = find_boundary_pixels(mask)
    nx = np.where(boundary, nx, 0.0)
    ny = np.where(boundary, ny, 0.0)

    return nx, ny


def find_interior_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Find pixels that are strictly interior to a mask (no outside neighbors).

    Args:
        mask: Boolean 2D array where True = inside the object

    Returns:
        Boolean 2D array where True = interior pixel (not boundary)
    """
    boundary = find_boundary_pixels(mask)
    return mask & ~boundary


def classify_boundary_neighbors(
    mask: np.ndarray,
    boundary_pixel: tuple[int, int]
) -> dict[str, bool]:
    """
    For a boundary pixel, classify which of its 4-connected neighbors are inside vs outside.

    Args:
        mask: Boolean 2D array
        boundary_pixel: (i, j) coordinates of the boundary pixel

    Returns:
        Dict with keys 'up', 'down', 'left', 'right' -> bool (True = neighbor is inside mask)
    """
    i, j = boundary_pixel
    h, w = mask.shape

    return {
        'up': (i > 0) and mask[i-1, j],
        'down': (i < h-1) and mask[i+1, j],
        'left': (j > 0) and mask[i, j-1],
        'right': (j < w-1) and mask[i, j+1],
    }
