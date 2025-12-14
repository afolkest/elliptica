"""GPU-accelerated rendering pipeline using ColorConfig.

Provides the bridge between the existing postprocessing pipeline and the
expression-based ColorConfig system.

Example:
    from elliptica.colorspace.pipeline import render_with_color_config

    # Build field bindings from render cache
    bindings = build_field_bindings(cache.result.array, cache.result.ex, cache.result.ey)

    # Render using ColorConfig
    rgb = render_with_color_config(color_config, bindings, region_masks)
"""

import torch
import numpy as np
from typing import Optional

from . import _backend as B
from ._backend import Array
from .mapping import ColorConfig


def build_field_bindings(
    lic: Array,
    ex: Optional[Array] = None,
    ey: Optional[Array] = None,
    solution: Optional[dict[str, Array]] = None,
) -> dict[str, Array]:
    """Build field bindings dict for ColorConfig expressions.

    Creates the standard field variables that expressions can reference:
    - lic: LIC texture (raw, not normalized - use clipnorm() in expressions)
    - mag: Field magnitude sqrt(ex² + ey²) (only if ex/ey provided)
    - ex, ey: Raw field components (only if provided)
    - Additional fields from solution dict (if provided)

    Args:
        lic: LIC texture array (H, W)
        ex: Field X component (H, W), optional
        ey: Field Y component (H, W), optional
        solution: Optional dict of additional solution fields

    Returns:
        Dict mapping variable names to arrays
    """
    bindings: dict[str, Array] = {'lic': lic}

    if ex is not None and ey is not None:
        # Compute magnitude
        if B.is_torch(ex):
            mag = torch.sqrt(ex**2 + ey**2)
        else:
            mag = np.sqrt(ex**2 + ey**2)
        bindings['mag'] = mag
        bindings['ex'] = ex
        bindings['ey'] = ey

    # Add solution fields (e.g., phi, E from PDE solver)
    if solution is not None:
        for name, arr in solution.items():
            if name not in bindings:  # Don't override standard names
                bindings[name] = arr

    return bindings


def build_region_masks(
    boundary_masks: Optional[list[Array]],
    interior_masks: Optional[list[Array]],
    boundaries: list,
) -> dict[str, Array]:
    """Build region masks dict for ColorConfig.

    Creates named masks for each boundary's surface and interior regions.
    Names follow the pattern: 'boundary_{id}_surface', 'boundary_{id}_interior'

    Args:
        boundary_masks: List of surface masks per boundary
        interior_masks: List of interior masks per boundary
        boundaries: List of BoundaryObject objects (used for IDs)

    Returns:
        Dict mapping region names to mask arrays
    """
    masks: dict[str, Array] = {}

    for idx, boundary in enumerate(boundaries):
        bid = boundary.id if boundary.id is not None else idx

        if boundary_masks and idx < len(boundary_masks) and boundary_masks[idx] is not None:
            masks[f'boundary_{bid}_surface'] = boundary_masks[idx]

        if interior_masks and idx < len(interior_masks) and interior_masks[idx] is not None:
            masks[f'boundary_{bid}_interior'] = interior_masks[idx]

    return masks


def render_with_color_config(
    config: ColorConfig,
    bindings: dict[str, Array],
    region_masks: Optional[dict[str, Array]] = None,
) -> Array:
    """Render using ColorConfig expressions.

    This is the main entry point for expression-based color rendering.
    Handles both numpy arrays and torch tensors (GPU acceleration automatic).

    Args:
        config: ColorConfig with global and optional region mappings
        bindings: Dict mapping variable names to field arrays
        region_masks: Dict mapping region names to mask arrays

    Returns:
        RGB array with shape (H, W, 3) and values in [0, 1]
    """
    return config.render(bindings, region_masks)


def render_with_color_config_gpu(
    config: ColorConfig,
    scalar_tensor: torch.Tensor,
    ex_tensor: Optional[torch.Tensor] = None,
    ey_tensor: Optional[torch.Tensor] = None,
    boundary_masks_gpu: Optional[list[Optional[torch.Tensor]]] = None,
    interior_masks_gpu: Optional[list[Optional[torch.Tensor]]] = None,
    boundaries: Optional[list] = None,
    solution: Optional[dict[str, np.ndarray]] = None,
) -> torch.Tensor:
    """GPU rendering using ColorConfig expressions.

    Higher-level function that builds bindings from common inputs.
    All computation stays on GPU when possible.

    Args:
        config: ColorConfig with global and optional region mappings
        scalar_tensor: LIC texture tensor (H, W) on GPU
        ex_tensor: Field X component tensor (H, W) on GPU, optional
        ey_tensor: Field Y component tensor (H, W) on GPU, optional
        boundary_masks_gpu: List of surface mask tensors on GPU
        interior_masks_gpu: List of interior mask tensors on GPU
        boundaries: List of BoundaryObject objects (for region naming)
        solution: Optional dict of CPU solution arrays (will be uploaded to GPU)

    Returns:
        RGB tensor with shape (H, W, 3) and values in [0, 1] on GPU
    """
    # Build bindings with GPU tensors
    bindings: dict[str, torch.Tensor] = {'lic': scalar_tensor}

    if ex_tensor is not None and ey_tensor is not None:
        mag = torch.sqrt(ex_tensor**2 + ey_tensor**2)
        bindings['mag'] = mag
        bindings['ex'] = ex_tensor
        bindings['ey'] = ey_tensor

    # Add solution fields to bindings (handles both numpy arrays and torch tensors)
    if solution is not None:
        device = scalar_tensor.device
        dtype = scalar_tensor.dtype
        for name, arr in solution.items():
            if name not in bindings:
                if isinstance(arr, torch.Tensor):
                    # Already a tensor, ensure on correct device
                    bindings[name] = arr.to(device=device, dtype=dtype)
                else:
                    # Numpy array, upload to GPU
                    bindings[name] = torch.from_numpy(arr).to(device=device, dtype=dtype)

    # Build region masks if we have boundaries
    region_masks: dict[str, torch.Tensor] = {}
    if boundaries is not None:
        for idx, boundary in enumerate(boundaries):
            bid = boundary.id if boundary.id is not None else idx

            if boundary_masks_gpu and idx < len(boundary_masks_gpu):
                mask = boundary_masks_gpu[idx]
                if mask is not None:
                    # Threshold soft mask to binary (match existing pipeline behavior)
                    region_masks[f'boundary_{bid}_surface'] = (mask > 0.5).float()

            if interior_masks_gpu and idx < len(interior_masks_gpu):
                mask = interior_masks_gpu[idx]
                if mask is not None:
                    region_masks[f'boundary_{bid}_interior'] = (mask > 0.5).float()

    # Render using ColorConfig
    rgb = config.render(bindings, region_masks if region_masks else None)

    return rgb


__all__ = [
    'build_field_bindings',
    'build_region_masks',
    'render_with_color_config',
    'render_with_color_config_gpu',
]
