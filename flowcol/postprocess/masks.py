"""Mask rasterization and interior detection for conductor colorization."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Optional


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
    from scipy.ndimage import binary_fill_holes

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
        # Rasterize surface mask
        surface = _rasterize_single_mask(
            conductor.mask,
            conductor.position,
            (height, width),
            margin,
            scale,
            offset_x,
            offset_y,
        )
        surface_masks.append(surface)

        # Rasterize interior mask if present
        if conductor.interior_mask is not None:
            interior = _rasterize_single_mask(
                conductor.interior_mask,
                conductor.position,
                (height, width),
                margin,
                scale,
                offset_x,
                offset_y,
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

    Returns:
        Rasterized mask at target_shape resolution
    """
    target_h, target_w = target_shape
    pos_x, pos_y = position

    # Apply margin offset and scaling, then subtract crop offset to align with cropped result
    grid_x = (pos_x + margin) * scale - offset_x
    grid_y = (pos_y + margin) * scale - offset_y

    # Scale mask dimensions
    mask_h, mask_w = mask.shape
    scaled_h = mask_h * scale
    scaled_w = mask_w * scale

    # Compute integer bounds on target grid
    x0 = int(np.floor(grid_x))
    y0 = int(np.floor(grid_y))
    x1 = int(np.ceil(grid_x + scaled_w))
    y1 = int(np.ceil(grid_y + scaled_h))

    # Clip to target bounds
    x0_clip = max(0, x0)
    y0_clip = max(0, y0)
    x1_clip = min(target_w, x1)
    y1_clip = min(target_h, y1)

    # Check if mask is completely outside target
    if x0_clip >= x1_clip or y0_clip >= y1_clip:
        return np.zeros(target_shape, dtype=np.float32)

    # Create output mask
    output = np.zeros(target_shape, dtype=np.float32)

    # Simple nearest-neighbor sampling (good enough for binary masks)
    for out_y in range(y0_clip, y1_clip):
        for out_x in range(x0_clip, x1_clip):
            # Map output pixel back to source mask coordinates
            src_x = (out_x - grid_x) / scale
            src_y = (out_y - grid_y) / scale

            # Nearest neighbor
            src_x_idx = int(np.round(src_x))
            src_y_idx = int(np.round(src_y))

            if 0 <= src_x_idx < mask_w and 0 <= src_y_idx < mask_h:
                output[out_y, out_x] = mask[src_y_idx, src_x_idx]

    return output
