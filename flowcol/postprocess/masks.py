"""Mask rasterization and interior detection for conductor colorization."""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes, zoom
from typing import Optional

from flowcol.mask_utils import blur_mask


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
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rasterize conductor masks onto display grid.

    Args:
        conductors: List of Conductor objects
        shape: Target (height, width) for rasterized masks
        margin: Physical margin used in render (pre-scale)
        scale: multiplier * supersample used for render
        offset_x: Crop offset in x direction (to align with cropped result.array)
        offset_y: Crop offset in y direction (to align with cropped result.array)

    Returns:
        (surface_masks, interior_masks) - lists of binary masks at display resolution
    """
    height, width = shape
    surface_masks = []
    interior_masks = []

    for conductor in conductors:
        # Get edge smoothing sigma (same as used in field solver and LIC mask)
        edge_smooth_sigma = getattr(conductor, 'edge_smooth_sigma', 0.0)

        # Rasterize surface mask with edge smoothing
        surface = _rasterize_single_mask(
            conductor.mask,
            conductor.position,
            (height, width),
            margin,
            scale,
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
                scale,
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
    scale: float,
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
        scale: Scaling factor applied to canvas
        offset_x: Crop offset in x direction (to align with cropped array)
        offset_y: Crop offset in y direction (to align with cropped array)
        edge_smooth_sigma: Edge smoothing in canvas pixels (will be scaled)

    Returns:
        Rasterized mask at target_shape resolution
    """
    target_h, target_w = target_shape
    pos_x, pos_y = position

    # Apply margin offset and scaling, then subtract crop offset to align with cropped result
    grid_x = (pos_x + margin) * scale - offset_x
    grid_y = (pos_y + margin) * scale - offset_y

    # Scale mask using zoom (vectorized, no holes)
    mask_h, mask_w = mask.shape
    new_h = max(1, int(round(mask_h * scale)))
    new_w = max(1, int(round(mask_w * scale)))

    zoom_y = new_h / mask_h
    zoom_x = new_w / mask_w
    scaled_mask = zoom(mask, (zoom_y, zoom_x), order=1)
    scaled_mask = np.clip(scaled_mask, 0.0, 1.0).astype(np.float32)

    # Apply edge smoothing (same as field solver and LIC mask)
    # This ensures smear region matches where field actually stops
    if edge_smooth_sigma > 0:
        # Scale sigma to match render resolution
        scaled_sigma = edge_smooth_sigma * scale
        scaled_mask = blur_mask(scaled_mask, scaled_sigma)

    # Compute integer placement on target grid
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
