"""GPU-accelerated anisotropic edge blur."""

import torch
import numpy as np
from typing import Optional


def apply_anisotropic_edge_blur_gpu(
    lic_tensor: torch.Tensor,
    ex_tensor: torch.Tensor,
    ey_tensor: torch.Tensor,
    conductor_masks: list[np.ndarray] | None,
    sigma: float,
    falloff_distance: float,
    strength: float,
) -> torch.Tensor:
    """Apply anisotropic blur perpendicular to field lines near conductor edges (GPU).

    Args:
        lic_tensor: Grayscale LIC tensor (H, W) on GPU
        ex_tensor: Electric field X component (H, W) on GPU
        ey_tensor: Electric field Y component (H, W) on GPU
        conductor_masks: List of conductor masks as NumPy arrays (will be uploaded)
        sigma: Gaussian sigma for perpendicular blur (pixels)
        falloff_distance: Distance from edge where blur falls to zero (pixels)
        strength: Global strength multiplier (0-2)

    Returns:
        Blurred LIC tensor on GPU
    """
    if sigma <= 0 or falloff_distance <= 0 or strength <= 0:
        return lic_tensor

    if conductor_masks is None or len(conductor_masks) == 0:
        return lic_tensor

    h, w = lic_tensor.shape
    device = lic_tensor.device

    # Compute combined distance field from all conductors
    combined_mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    for mask in conductor_masks:
        if mask is not None and mask.shape == (h, w):
            mask_tensor = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
            combined_mask = torch.maximum(combined_mask, mask_tensor)

    if not torch.any(combined_mask > 0):
        return lic_tensor

    # Distance transform (approximation on GPU)
    distance_field = _distance_transform_gpu(combined_mask > 0.01)

    # Compute blend weight: high near edges, decays with distance
    weight = strength * torch.exp(-distance_field / falloff_distance)
    weight = torch.clamp(weight, 0.0, 1.0)

    # Early exit if no significant weight anywhere
    if torch.max(weight) < 0.01:
        return lic_tensor

    # Compute perpendicular direction at each pixel
    # Field direction: (ex, ey)
    # Perpendicular: (-ey, ex) normalized
    field_mag = torch.sqrt(ex_tensor**2 + ey_tensor**2) + 1e-8
    perp_x = -ey_tensor / field_mag
    perp_y = ex_tensor / field_mag

    # Apply directional blur on GPU
    blurred = _directional_blur_gpu(lic_tensor, perp_x, perp_y, sigma)

    # Blend based on distance weight
    result = lic_tensor * (1.0 - weight) + blurred * weight

    return result


def _distance_transform_gpu(mask: torch.Tensor, max_distance: float = 100.0) -> torch.Tensor:
    """Approximate distance transform on GPU using max pooling.

    Args:
        mask: Binary mask tensor (H, W) on GPU
        max_distance: Maximum distance to compute (limits iterations)

    Returns:
        Distance field tensor (H, W) on GPU
    """
    # Start with 0 inside mask, max_distance outside
    distance = torch.where(mask, 0.0, max_distance)

    # Use iterative max pooling for fast GPU distance approximation
    # Each iteration propagates distance by 1 pixel in all directions
    iterations = min(int(max_distance), 15)  # Cap iterations for performance

    # Prepare for max pooling (need 4D: batch, channel, height, width)
    dist_4d = distance.unsqueeze(0).unsqueeze(0)

    for i in range(iterations):
        # Use max pooling with stride=1 to find minimum in 3x3 neighborhood
        # (Note: we use -distance and -max_pool to get minimum)
        neg_dist = -dist_4d
        neg_pooled = torch.nn.functional.max_pool2d(
            neg_dist, kernel_size=3, stride=1, padding=1
        )
        pooled = -neg_pooled

        # Add 1.0 to propagated distances
        new_dist_4d = torch.minimum(dist_4d, pooled + 1.0)

        # Early stopping if converged
        if torch.allclose(new_dist_4d, dist_4d, atol=0.01):
            break

        dist_4d = new_dist_4d

    # Convert back to 2D
    distance = dist_4d.squeeze(0).squeeze(0)

    return distance


def _directional_blur_gpu(
    array: torch.Tensor,
    dir_x: torch.Tensor,
    dir_y: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Apply 1D Gaussian blur along specified direction field on GPU.

    Args:
        array: Input tensor (H, W) on GPU
        dir_x: X component of blur direction at each pixel (H, W) on GPU
        dir_y: Y component of blur direction at each pixel (H, W) on GPU
        sigma: Gaussian sigma in pixels

    Returns:
        Blurred tensor on GPU
    """
    h, w = array.shape
    device = array.device

    # Sample radius - use fewer samples for speed
    radius = max(1, int(np.ceil(2 * sigma)))
    n_samples = min(2 * radius + 1, 11)  # Cap at 11 samples

    # Create sample offsets
    t_samples = torch.linspace(-radius, radius, n_samples, device=device, dtype=torch.float32)

    # Gaussian weights
    weights = torch.exp(-0.5 * (t_samples / sigma) ** 2)
    weights = weights / torch.sum(weights)

    # Create coordinate grids
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Accumulate weighted samples
    result = torch.zeros_like(array, dtype=torch.float32)

    # Process each sample
    for t, weight in zip(t_samples, weights):
        # Compute sample coordinates
        sample_y = y_grid + dir_y * t
        sample_x = x_grid + dir_x * t

        # Clamp to boundaries and round to nearest (nearest-neighbor sampling)
        sample_y = torch.clamp(torch.round(sample_y).long(), 0, h - 1)
        sample_x = torch.clamp(torch.round(sample_x).long(), 0, w - 1)

        # Direct tensor indexing (fast on GPU)
        result += array[sample_y, sample_x] * weight

    return result


__all__ = ['apply_anisotropic_edge_blur_gpu']
