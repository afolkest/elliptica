"""Mask rasterization and interior detection for boundary colorization."""

import numpy as np
from scipy.ndimage import binary_fill_holes
from typing import Optional

from elliptica.mask_utils import place_mask_in_grid


def derive_interior(mask: np.ndarray, thickness: float = 0.1) -> Optional[np.ndarray]:
    """Derive interior from hollow boundary mask using morphological hole filling.

    For hollow boundaries (rings), detects the empty region inside.
    For solid boundaries (disks), returns None.

    Uses standard morphological operation: binary_fill_holes identifies regions
    surrounded by the mask but not part of it.

    Args:
        mask: Binary/grayscale boundary mask
        thickness: Unused, kept for API compatibility

    Returns:
        Interior mask as float32 array, or None if no hollow interior exists
    """
    # Convert to binary
    binary = (mask > 0.5).astype(bool)

    # Early exit if mask is empty
    if not np.any(binary):
        return None

    # Fill holes: identifies interior regions surrounded by boundary
    filled = binary_fill_holes(binary)

    # Interior is the difference between filled and original
    interior = filled & ~binary

    # If no interior exists (solid shape), return None
    if not np.any(interior):
        return None

    return interior.astype(np.float32)


def rasterize_boundary_masks(
    boundaries,
    shape: tuple[int, int],
    margin: float,
    scale: float,
    offset_x: int = 0,
    offset_y: int = 0,
    domain_size: tuple[float, float] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rasterize boundary masks onto display grid.

    Args:
        boundaries: List of BoundaryObject objects
        shape: Target (height, width) for rasterized masks
        margin: Physical margin used in render (pre-scale)
        scale: multiplier * supersample used for render
        offset_x: Crop offset in x direction (to align with cropped result.array)
        offset_y: Crop offset in y direction (to align with cropped result.array)
        domain_size: Optional (domain_w, domain_h) for computing exact scale factors.
            If provided, scale_x and scale_y are computed to match the LIC/field solver.

    Returns:
        (surface_masks, interior_masks) - lists of binary masks at display resolution
    """
    height, width = shape
    surface_masks = []
    interior_masks = []

    # Compute scale factors that match the field solver / LIC mask exactly
    # The field solver uses: compute_dim / domain_dim (after int rounding of compute_dim)
    # This ensures masks align pixel-perfectly with where the field is actually blocked
    if domain_size is not None:
        domain_w, domain_h = domain_size
        compute_w_full = int(round(domain_w * scale))
        compute_h_full = int(round(domain_h * scale))
        scale_x = compute_w_full / domain_w if domain_w > 0 else scale
        scale_y = compute_h_full / domain_h if domain_h > 0 else scale
    else:
        scale_x = scale
        scale_y = scale

    for boundary in boundaries:
        # Get edge smoothing sigma (same as used in field solver and LIC mask)
        edge_smooth_sigma = getattr(boundary, 'edge_smooth_sigma', 0.0)

        # Rasterize surface mask with edge smoothing
        surface = _rasterize_single_mask(
            boundary.mask,
            boundary.position,
            (height, width),
            margin,
            scale_x,
            scale_y,
            offset_x,
            offset_y,
            edge_smooth_sigma=edge_smooth_sigma,
        )
        surface_masks.append(surface)

        # Rasterize interior mask if present (no smoothing - interior is derived region)
        if boundary.interior_mask is not None:
            interior = _rasterize_single_mask(
                boundary.interior_mask,
                boundary.position,
                (height, width),
                margin,
                scale_x,
                scale_y,
                offset_x,
                offset_y,
                edge_smooth_sigma=0.0,  # Interior doesn't need smoothing
            )
            interior_masks.append(interior)
        else:
            # Empty interior mask
            interior_masks.append(np.zeros((height, width), dtype=np.float32))

    return surface_masks, interior_masks


def _rasterize_single_mask(
    mask: np.ndarray,
    position: tuple[float, float],
    target_shape: tuple[int, int],
    margin: float,
    scale_x: float,
    scale_y: float,
    offset_x: int = 0,
    offset_y: int = 0,
    edge_smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Rasterize a single mask onto target grid with proper alignment.

    Args:
        mask: Source mask array
        position: (x, y) position on canvas (pre-margin, pre-scale)
        target_shape: (height, width) of output
        margin: Physical margin (pre-scale)
        scale_x: X scaling factor (compute_w / domain_w)
        scale_y: Y scaling factor (compute_h / domain_h)
        offset_x: Crop offset in x direction (to align with cropped array)
        offset_y: Crop offset in y direction (to align with cropped array)
        edge_smooth_sigma: Edge smoothing in canvas pixels (will be scaled)

    Returns:
        Rasterized mask at target_shape resolution
    """
    result = place_mask_in_grid(
        mask, position, target_shape,
        margin=(margin, margin), scale=(scale_x, scale_y),
        edge_smooth_sigma=edge_smooth_sigma,
        offset=(offset_x, offset_y),
    )
    if result is None:
        return np.zeros(target_shape, dtype=np.float32)
    placed, (y0, y1, x0, x1) = result
    output = np.zeros(target_shape, dtype=np.float32)
    output[y0:y1, x0:x1] = placed
    return output
