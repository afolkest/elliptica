"""Anisotropic edge blur for smoothing artifacts near conductor boundaries."""

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def apply_anisotropic_edge_blur(
    lic_array: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    conductor_masks: list[np.ndarray] | None,
    sigma: float,
    falloff_distance: float,
    strength: float,
) -> np.ndarray:
    """Apply anisotropic blur perpendicular to field lines near conductor edges.

    Args:
        lic_array: Grayscale LIC array (H, W)
        ex: Electric field X component (H, W)
        ey: Electric field Y component (H, W)
        conductor_masks: List of conductor masks (or None)
        sigma: Gaussian sigma for perpendicular blur (pixels)
        falloff_distance: Distance from edge where blur falls to zero (pixels)
        strength: Global strength multiplier (0-2)

    Returns:
        Blurred LIC array
    """
    if sigma <= 0 or falloff_distance <= 0 or strength <= 0:
        return lic_array

    if conductor_masks is None or len(conductor_masks) == 0:
        return lic_array

    h, w = lic_array.shape

    # Compute combined distance field from all conductors
    combined_mask = np.zeros((h, w), dtype=np.float32)
    for mask in conductor_masks:
        if mask is not None and mask.shape == (h, w):
            combined_mask = np.maximum(combined_mask, mask)

    if not np.any(combined_mask > 0):
        return lic_array

    # Distance transform from conductor edges
    # Distance is 0 at edge, increases outward
    distance_field = distance_transform_edt(1.0 - (combined_mask > 0.01))

    # Compute blend weight: 1 at edge, decays with distance
    # weight = strength * (1 - exp(-distance / falloff))
    # Invert: we want high weight near edges
    weight = strength * np.exp(-distance_field / falloff_distance)
    weight = np.clip(weight, 0.0, 1.0)

    # Early exit if no significant weight anywhere
    if np.max(weight) < 0.01:
        return lic_array

    # Compute perpendicular direction at each pixel
    # Field direction: (ex, ey)
    # Perpendicular: (-ey, ex) normalized
    field_mag = np.sqrt(ex**2 + ey**2) + 1e-8
    perp_x = -ey / field_mag
    perp_y = ex / field_mag

    # Apply directional blur
    blurred = _directional_blur(lic_array, perp_x, perp_y, sigma)

    # Blend based on distance weight
    result = lic_array * (1.0 - weight) + blurred * weight

    return result.astype(np.float32)


def _directional_blur(
    array: np.ndarray,
    dir_x: np.ndarray,
    dir_y: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Apply 1D Gaussian blur along specified direction field.

    Args:
        array: Input array (H, W)
        dir_x: X component of blur direction at each pixel (H, W)
        dir_y: Y component of blur direction at each pixel (H, W)
        sigma: Gaussian sigma in pixels

    Returns:
        Blurred array
    """
    h, w = array.shape

    # Sample radius (3 sigma covers ~99.7% of Gaussian)
    radius = int(np.ceil(3 * sigma))
    if radius == 0:
        return array

    # Create sample offsets along direction
    t_samples = np.linspace(-radius, radius, 2 * radius + 1)
    n_samples = len(t_samples)

    # Gaussian weights
    weights = np.exp(-0.5 * (t_samples / sigma) ** 2)
    weights /= np.sum(weights)

    # Create coordinate grids
    y_grid, x_grid = np.ogrid[:h, :w]
    y_grid = y_grid.astype(np.float32)
    x_grid = x_grid.astype(np.float32)

    # Vectorized: create all sample coordinates at once
    # Shape: (n_samples, H, W)
    sample_y_all = y_grid[None, :, :] + dir_y[None, :, :] * t_samples[:, None, None]
    sample_x_all = x_grid[None, :, :] + dir_x[None, :, :] * t_samples[:, None, None]

    # Clamp to boundaries
    sample_y_all = np.clip(sample_y_all, 0, h - 1)
    sample_x_all = np.clip(sample_x_all, 0, w - 1)

    # Flatten to 1D for map_coordinates
    # Coordinates shape: (2, n_samples * H * W)
    coords = np.stack([
        sample_y_all.ravel(),
        sample_x_all.ravel()
    ], axis=0)

    # Single map_coordinates call for all samples
    samples_flat = map_coordinates(
        array,
        coords,
        order=1,
        mode='nearest'
    )

    # Reshape back to (n_samples, H, W)
    samples_all = samples_flat.reshape(n_samples, h, w)

    # Weighted sum along sample dimension
    result = np.sum(samples_all * weights[:, None, None], axis=0)

    return result.astype(np.float32)
