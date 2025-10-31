"""GPU-accelerated conductor smear effects."""

import torch
import numpy as np
from typing import Tuple
from scipy.ndimage import zoom

from flowcol.gpu import GPUContext
from flowcol.gpu.ops import gaussian_blur_gpu, percentile_clip_gpu, apply_palette_lut_gpu, grayscale_to_rgb_gpu, apply_contrast_gamma_gpu
from flowcol.render import _get_palette_lut


def apply_conductor_smear_gpu(
    rgb_tensor: torch.Tensor,
    lic_gray_tensor: torch.Tensor,
    conductor_masks: list[np.ndarray],
    conductors: list,
    render_shape: Tuple[int, int],
    canvas_resolution: Tuple[int, int],
    lut_tensor: torch.Tensor | None,
    lic_percentiles: Tuple[float, float] | None = None,
    conductor_color_settings: dict | None = None,
    conductor_masks_gpu: list[torch.Tensor | None] | None = None,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
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
        conductor_color_settings: Per-conductor color settings dict (to detect custom colors)
        conductor_masks_gpu: Optional pre-uploaded GPU masks (avoids repeated CPU→GPU transfers)
        brightness: Brightness adjustment (additive, 0.0 = no change)
        contrast: Contrast multiplier (1.0 = no change)
        gamma: Gamma exponent (1.0 = no change)

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

        # Use pre-uploaded GPU mask if available, otherwise upload from CPU
        if conductor_masks_gpu is not None and idx < len(conductor_masks_gpu) and conductor_masks_gpu[idx] is not None:
            # Use pre-uploaded GPU mask (fast path!)
            full_mask = conductor_masks_gpu[idx]
        else:
            # Upload mask to GPU (fallback path)
            mask_cpu = conductor_masks[idx]
            full_mask = GPUContext.to_gpu(mask_cpu)

        mask_bool = full_mask > 0.5

        if not torch.any(mask_bool):
            continue

        sigma_px = max(conductor.smear_sigma, 0.1)

        # ALWAYS blur the LIC grayscale to get the field texture (never blur RGB globally!)
        lic_blur = gaussian_blur_gpu(lic_gray_tensor, sigma_px)

        # Normalize the blurred LIC
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

        # Apply brightness/contrast/gamma adjustments (matches overlay behavior)
        # This ensures smear colors match region overlay colors when using the same palette
        adjusted = apply_contrast_gamma_gpu(norm, brightness, contrast, gamma)

        # Check if this conductor has custom color settings
        has_custom_settings = (
            conductor_color_settings is not None
            and conductor.id in conductor_color_settings
        )

        if has_custom_settings:
            # Get custom settings for this conductor
            settings = conductor_color_settings[conductor.id]

            # Use surface settings for smear (surface is the conductor body)
            if settings.surface.enabled:
                if settings.surface.use_palette:
                    # Custom palette: apply that palette to adjusted LIC
                    custom_lut = _get_palette_lut(settings.surface.palette)
                    custom_lut_tensor = GPUContext.to_gpu(custom_lut)
                    rgb_blur = apply_palette_lut_gpu(adjusted, custom_lut_tensor)
                else:
                    # Solid custom color: colorize the adjusted LIC with the custom color
                    # This preserves the LIC texture while using the custom color
                    solid_color = settings.surface.solid_color  # (r, g, b) in [0, 1]
                    color_tensor = torch.tensor(solid_color, dtype=torch.float32, device=adjusted.device)
                    # Broadcast: (H, W) * (3,) -> (H, W, 3) with color modulated by intensity
                    rgb_blur = adjusted.unsqueeze(-1) * color_tensor
            else:
                # Custom settings exist but surface not enabled - fall back to global
                if lut_tensor is not None:
                    rgb_blur = apply_palette_lut_gpu(adjusted, lut_tensor)
                else:
                    rgb_blur = grayscale_to_rgb_gpu(adjusted)
        else:
            # No custom settings - use global palette/grayscale
            if lut_tensor is not None:
                rgb_blur = apply_palette_lut_gpu(adjusted, lut_tensor)
            else:
                rgb_blur = grayscale_to_rgb_gpu(adjusted)

        # Apply smear at full strength inside mask (no distance-based feathering)
        weight = full_mask.unsqueeze(-1)  # Broadcast mask to RGB channels (H, W, 1)
        out = out * (1.0 - weight) + rgb_blur * weight

        # Clean up large temporary tensors to avoid GPU memory accumulation
        del full_mask, mask_bool, lic_blur, norm, adjusted, rgb_blur, weight

    return torch.clamp(out, 0.0, 1.0)


__all__ = ['apply_conductor_smear_gpu']
