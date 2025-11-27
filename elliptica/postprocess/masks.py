"""Mask rasterization and interior detection for conductor colorization."""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes, zoom
from typing import Optional

from elliptica.mask_utils import blur_mask


def derive_interior(mask: np.ndarray, thickness: float = 0.1) -> Optional[np.ndarray]:
    """Derive interior from hollow conductor mask using morphological hole filling.

    For hollow conductors (rings), detects the empty region inside.
    For solid conductors (disks), returns None.

    Uses standard morphological operation: binary_fill_holes identifies regions
    surrounded by the mask but not part of it.

    Args:
        mask: Binary/grayscale conductor mask
        thickness: Unused, kept for API compatibility

    Returns:
        Interior mask as float32 array, or None if no hollow interior exists
    """
    # Convert to binary
    binary = (mask > 0.5).astype(bool)

    # Early exit if mask is empty
    if not np.any(binary):
        return None

    # Fill holes: identifies interior regions surrounded by conductor
    filled = binary_fill_holes(binary)

    # Interior is the difference between filled and original
    interior = filled & ~binary

    # If no interior exists (solid shape), return None
    if not np.any(interior):
        return None

    return interior.astype(np.float32)


def rasterize_conductor_masks(
    conductors,
    shape: tuple[int, int],
    margin: float,
    scale: float,
    offset_x: int = 0,
    offset_y: int = 0,
    domain_size: tuple[float, float] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rasterize conductor masks onto display grid.

    Args:
        conductors: List of Conductor objects
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
        # Reconstruct compute dimensions (full domain before cropping)
        compute_w = width + offset_x + int(round(margin * scale))  # Approximate
        compute_h = height + offset_y + int(round(margin * scale))
        # Actually, we need the original compute dimensions. Use domain_size directly.
        compute_w_full = int(round(domain_w * scale))
        compute_h_full = int(round(domain_h * scale))
        scale_x = compute_w_full / domain_w if domain_w > 0 else scale
        scale_y = compute_h_full / domain_h if domain_h > 0 else scale
    else:
        scale_x = scale
        scale_y = scale

    for conductor in conductors:
        # Get edge smoothing sigma (same as used in field solver and LIC mask)
        edge_smooth_sigma = getattr(conductor, 'edge_smooth_sigma', 0.0)

        # Rasterize surface mask with edge smoothing
        surface = _rasterize_single_mask(
            conductor.mask,
            conductor.position,
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
        if conductor.interior_mask is not None:
            interior = _rasterize_single_mask(
                conductor.interior_mask,
                conductor.position,
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
    target_h, target_w = target_shape
    pos_x, pos_y = position

    # Apply margin offset and scaling, then subtract crop offset to align with cropped result
    # This matches the field solver: x = (pos + margin) * scale
    grid_x = (pos_x + margin) * scale_x - offset_x
    grid_y = (pos_y + margin) * scale_y - offset_y

    # Scale mask using zoom - use the same (scale_y, scale_x) as field solver
    scaled_mask = zoom(mask, (scale_y, scale_x), order=0)
    scaled_mask = np.clip(scaled_mask, 0.0, 1.0).astype(np.float32)

    # Apply edge smoothing (same as field solver and LIC mask)
    # Field solver uses: scale_factor = (scale_x + scale_y) / 2.0
    if edge_smooth_sigma > 0:
        scale_factor = (scale_x + scale_y) / 2.0
        scaled_sigma = edge_smooth_sigma * scale_factor
        scaled_mask = blur_mask(scaled_mask, scaled_sigma)

    # Compute integer placement on target grid
    # Use actual scaled mask dimensions (after zoom)
    new_h, new_w = scaled_mask.shape
    x0 = int(round(grid_x))
    y0 = int(round(grid_y))
    x1 = x0 + new_w
    y1 = y0 + new_h

    # Compute valid paste region (intersection with target)
    paste_x0 = max(0, x0)
    paste_y0 = max(0, y0)
    paste_x1 = min(target_w, x1)
    paste_y1 = min(target_h, y1)

    # Check if completely outside
    if paste_x0 >= paste_x1 or paste_y0 >= paste_y1:
        return np.zeros(target_shape, dtype=np.float32)

    # Compute source region to copy (handle partial overlap)
    src_x0 = paste_x0 - x0
    src_y0 = paste_y0 - y0
    src_x1 = src_x0 + (paste_x1 - paste_x0)
    src_y1 = src_y0 + (paste_y1 - paste_y0)

    # Create output and paste
    output = np.zeros(target_shape, dtype=np.float32)
    output[paste_y0:paste_y1, paste_x0:paste_x1] = scaled_mask[src_y0:src_y1, src_x0:src_x1]

    return output
