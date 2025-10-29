"""GPU-accelerated conductor smear effects."""

import torch
import numpy as np
from typing import Tuple
from scipy.ndimage import zoom

from flowcol.gpu import GPUContext
from flowcol.gpu.ops import gaussian_blur_gpu, percentile_clip_gpu, apply_palette_lut_gpu, grayscale_to_rgb_gpu


def apply_conductor_smear_gpu(
    rgb_tensor: torch.Tensor,
    lic_gray_tensor: torch.Tensor,
    conductor_masks: list[np.ndarray],
    conductors: list,
    render_shape: Tuple[int, int],
    canvas_resolution: Tuple[int, int],
    lut_tensor: torch.Tensor | None,
    lic_percentiles: Tuple[float, float] | None = None,
) -> torch.Tensor:
    """Apply smear effect to texture inside conductor masks on GPU.

    Args:
        rgb_tensor: Current RGB image (H, W, 3) float32 in [0, 1] on GPU
        lic_gray_tensor: Original LIC grayscale (H, W) float32 on GPU
        conductor_masks: List of conductor masks (CPU arrays)
        conductors: List of Conductor objects
        render_shape: (height, width) of render resolution
        canvas_resolution: (width, height) of canvas
        lut_tensor: Color palette LUT on GPU, or None for grayscale
        lic_percentiles: Precomputed (vmin, vmax) for normalization

    Returns:
        Modified RGB tensor (H, W, 3) float32 in [0, 1] on GPU
    """
    out = rgb_tensor.clone()

    render_h, render_w = render_shape
    canvas_w, canvas_h = canvas_resolution
    scale_x = render_w / canvas_w
    scale_y = render_h / canvas_h

    # Precompute percentiles if needed and any conductor has smear
    if lic_percentiles is None and any(c.smear_enabled for c in conductors):
        # Use GPU quantile (much faster than CPU percentile!)
        flat = lic_gray_tensor.flatten()
        quantiles = torch.tensor([0.005, 0.995], device=lic_gray_tensor.device, dtype=lic_gray_tensor.dtype)
        vmin_tensor, vmax_tensor = torch.quantile(flat, quantiles)
        vmin, vmax = vmin_tensor.item(), vmax_tensor.item()
    elif lic_percentiles is not None:
        vmin, vmax = lic_percentiles
    else:
        vmin, vmax = 0.0, 1.0

    for idx, conductor in enumerate(conductors):
        if not conductor.smear_enabled:
            continue

        if idx >= len(conductor_masks) or conductor_masks[idx] is None:
            continue

        # Use pre-rasterized mask passed in (already at correct resolution)
        mask_cpu = conductor_masks[idx]

        # Upload mask to GPU
        full_mask = GPUContext.to_gpu(mask_cpu)
        mask_bool = full_mask > 0.5

        if not torch.any(mask_bool):
            continue

        # Blur LIC grayscale globally (to avoid boundary artifacts)
        sigma_px = max(conductor.smear_sigma, 0.1)
        lic_blur = gaussian_blur_gpu(lic_gray_tensor, sigma_px)

        # Re-normalize and colorize (creates "melted blob" effect)
        if lut_tensor is not None:
            # Color mode: normalize using precomputed percentiles
            if vmax > vmin:
                norm = torch.clamp((lic_blur - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                # Fallback: normalize by tensor's own range
                tmin = lic_blur.min()
                tmax = lic_blur.max()
                if tmax > tmin:
                    norm = (lic_blur - tmin) / (tmax - tmin)
                else:
                    norm = torch.zeros_like(lic_blur)

            # Apply color palette on GPU
            rgb_blur = apply_palette_lut_gpu(norm, lut_tensor)
        else:
            # Grayscale mode: normalize and convert to RGB
            if vmax > vmin:
                norm = torch.clamp((lic_blur - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                tmin = lic_blur.min()
                tmax = lic_blur.max()
                if tmax > tmin:
                    norm = (lic_blur - tmin) / (tmax - tmin)
                else:
                    norm = torch.zeros_like(lic_blur)

            rgb_blur = grayscale_to_rgb_gpu(norm)

        # Apply smear at full strength inside mask (no distance-based feathering)
        weight = full_mask.unsqueeze(-1)  # Broadcast mask to RGB channels (H, W, 1)
        out = out * (1.0 - weight) + rgb_blur * weight

        # Clean up large temporary tensors to avoid GPU memory accumulation
        del full_mask, mask_bool, lic_blur, rgb_blur, weight

    return torch.clamp(out, 0.0, 1.0)


__all__ = ['apply_conductor_smear_gpu']
