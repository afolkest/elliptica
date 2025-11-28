"""GPU-accelerated region smear effects."""

import torch
import numpy as np
from typing import Tuple, Optional

from elliptica.gpu import GPUContext
from elliptica.gpu.ops import gaussian_blur_gpu, quantile_safe, apply_palette_lut_gpu, grayscale_to_rgb_gpu, apply_contrast_gamma_gpu
from elliptica.render import _get_palette_lut


def compute_mask_bbox_cpu(mask: np.ndarray, pad: int, max_h: int, max_w: int) -> Optional[Tuple[int, int, int, int]]:
    """Compute bounding box of non-zero region in mask (CPU operation, no GPU sync).

    Args:
        mask: Binary mask array (H, W)
        pad: Padding to add around bounding box
        max_h: Maximum height (for clamping)
        max_w: Maximum width (for clamping)

    Returns:
        (y_min, y_max, x_min, x_max) or None if mask is empty
    """
    coords = np.argwhere(mask > 0.5)
    if coords.shape[0] == 0:
        return None

    y_min = int(coords[:, 0].min())
    y_max = int(coords[:, 0].max())
    x_min = int(coords[:, 1].min())
    x_max = int(coords[:, 1].max())

    # Add padding for blur kernel (3*sigma on each side is sufficient)
    y_min_pad = max(0, y_min - pad)
    y_max_pad = min(max_h, y_max + pad + 1)
    x_min_pad = max(0, x_min - pad)
    x_max_pad = min(max_w, x_max + pad + 1)

    return (y_min_pad, y_max_pad, x_min_pad, x_max_pad)


def _has_any_smear_enabled(conductor_color_settings: dict | None, conductors: list) -> bool:
    """Check if any region has smear enabled."""
    if conductor_color_settings is None:
        return False
    for conductor in conductors:
        if conductor.id in conductor_color_settings:
            settings = conductor_color_settings[conductor.id]
            if settings.surface.smear_enabled or settings.interior.smear_enabled:
                return True
    return False


def _apply_smear_to_region(
    out: torch.Tensor,
    lic_gray_tensor: torch.Tensor,
    mask_cpu: np.ndarray,
    mask_gpu: torch.Tensor | None,
    region_style,
    render_shape: Tuple[int, int],
    vmin: float,
    vmax: float,
    lut_tensor: torch.Tensor | None,
    global_brightness: float,
    global_contrast: float,
    gamma: float,
    ex_tensor: torch.Tensor | None = None,
    ey_tensor: torch.Tensor | None = None,
    solution_gpu: dict[str, torch.Tensor] | None = None,
    global_lightness_expr: str | None = None,
) -> torch.Tensor:
    """Apply smear effect to a single region.

    Args:
        out: Current output RGB tensor (modified in place)
        lic_gray_tensor: Original LIC grayscale on GPU
        mask_cpu: Region mask on CPU
        mask_gpu: Pre-uploaded GPU mask (or None)
        region_style: RegionStyle with smear settings
        render_shape: (height, width)
        vmin, vmax: Percentiles for normalization
        lut_tensor: Global color palette LUT
        global_brightness, global_contrast, gamma: Global postprocess settings
        ex_tensor, ey_tensor: Electric field components for lightness expressions
        solution_gpu: PDE solution fields for lightness expressions
        global_lightness_expr: Global lightness expression (fallback)

    Returns:
        Modified output tensor
    """
    render_h, render_w = render_shape

    # Use pre-uploaded GPU mask if available, otherwise upload from CPU
    if mask_gpu is not None:
        full_mask = mask_gpu
    else:
        full_mask = GPUContext.to_gpu(mask_cpu)

    mask_bool = full_mask > 0.5
    if not torch.any(mask_bool):
        return out

    # Convert fractional sigma to pixels
    sigma_px = max(region_style.smear_sigma * render_w, 0.1)

    # Compute bounding box on CPU
    pad = int(3 * sigma_px) + 1
    bbox = compute_mask_bbox_cpu(mask_cpu, pad, render_h, render_w)
    if bbox is None:
        return out

    y_min_pad, y_max_pad, x_min_pad, x_max_pad = bbox

    # Extract and blur region
    lic_region = lic_gray_tensor[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
    lic_blur_region = gaussian_blur_gpu(lic_region, sigma_px)

    # Create full-size blur tensor
    lic_blur = torch.zeros_like(lic_gray_tensor)
    lic_blur[y_min_pad:y_max_pad, x_min_pad:x_max_pad] = lic_blur_region

    # Normalize
    if vmax > vmin:
        norm = torch.clamp((lic_blur - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        tmin = lic_blur.min()
        tmax = lic_blur.max()
        norm = (lic_blur - tmin) / (tmax - tmin) if tmax > tmin else torch.zeros_like(lic_blur)

    # Resolve per-region brightness/contrast
    region_brightness = region_style.brightness if region_style.brightness is not None else global_brightness
    region_contrast = region_style.contrast if region_style.contrast is not None else global_contrast

    # Apply brightness/contrast/gamma
    adjusted = apply_contrast_gamma_gpu(norm, region_brightness, region_contrast, gamma)

    # Colorize based on region settings
    if region_style.enabled:
        if region_style.use_palette:
            custom_lut = _get_palette_lut(region_style.palette)
            custom_lut_tensor = GPUContext.to_gpu(custom_lut)
            rgb_blur = apply_palette_lut_gpu(adjusted, custom_lut_tensor)
        else:
            solid_color = region_style.solid_color
            color_tensor = torch.tensor(solid_color, dtype=torch.float32, device=adjusted.device)
            rgb_blur = adjusted.unsqueeze(-1) * color_tensor
    else:
        # Use global palette/grayscale
        if lut_tensor is not None:
            rgb_blur = apply_palette_lut_gpu(adjusted, lut_tensor)
        else:
            rgb_blur = grayscale_to_rgb_gpu(adjusted)

    # Apply lightness expression (region-specific or global fallback)
    region_expr = region_style.lightness_expr if region_style.lightness_expr is not None else global_lightness_expr
    if region_expr is not None:
        from elliptica.gpu.overlay import _apply_lightness_expr_to_rgb
        rgb_blur = _apply_lightness_expr_to_rgb(
            rgb_blur, region_expr, lic_gray_tensor,
            ex_tensor, ey_tensor, solution_gpu
        )

    # Apply smear inside mask
    weight = (full_mask > 0.5).float().unsqueeze(-1)
    out = out * (1.0 - weight) + rgb_blur * weight

    # Clean up
    del lic_blur, norm, adjusted, rgb_blur, weight

    return out


def apply_region_smear_gpu(
    rgb_tensor: torch.Tensor,
    lic_gray_tensor: torch.Tensor,
    conductor_masks: list[np.ndarray] | None,
    interior_masks: list[np.ndarray] | None,
    conductors: list,
    render_shape: Tuple[int, int],
    canvas_resolution: Tuple[int, int],
    lut_tensor: torch.Tensor | None,
    lic_percentiles: Tuple[float, float] | None = None,
    conductor_color_settings: dict | None = None,
    conductor_masks_gpu: list[torch.Tensor | None] | None = None,
    interior_masks_gpu: list[torch.Tensor | None] | None = None,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    ex_tensor: torch.Tensor | None = None,
    ey_tensor: torch.Tensor | None = None,
    solution_gpu: dict[str, torch.Tensor] | None = None,
    global_lightness_expr: str | None = None,
) -> torch.Tensor:
    """Apply smear effect to regions (both surfaces and interiors) on GPU.

    Smear is now a per-region effect stored in conductor_color_settings.

    Args:
        rgb_tensor: Current RGB image (H, W, 3) float32 in [0, 1] on GPU
        lic_gray_tensor: Original LIC grayscale (H, W) float32 on GPU
        conductor_masks: List of conductor surface masks (CPU arrays)
        interior_masks: List of interior masks (CPU arrays)
        conductors: List of Conductor objects
        render_shape: (height, width) of render resolution
        canvas_resolution: (width, height) of canvas
        lut_tensor: Color palette LUT on GPU, or None for grayscale
        lic_percentiles: Precomputed (vmin, vmax) for normalization
        conductor_color_settings: Per-conductor color settings dict (contains smear settings)
        conductor_masks_gpu: Optional pre-uploaded surface masks on GPU
        interior_masks_gpu: Optional pre-uploaded interior masks on GPU
        brightness: Global brightness adjustment
        contrast: Global contrast multiplier
        gamma: Gamma exponent
        ex_tensor, ey_tensor: Electric field components for lightness expressions
        solution_gpu: PDE solution fields for lightness expressions
        global_lightness_expr: Global lightness expression (fallback)

    Returns:
        Modified RGB tensor (H, W, 3) float32 in [0, 1] on GPU
    """
    if conductor_color_settings is None:
        return rgb_tensor

    out = rgb_tensor.clone()
    render_h, render_w = render_shape

    # Precompute percentiles if needed
    if lic_percentiles is None and _has_any_smear_enabled(conductor_color_settings, conductors):
        flat = lic_gray_tensor.flatten()
        quantiles = torch.tensor([0.005, 0.995], device=lic_gray_tensor.device, dtype=lic_gray_tensor.dtype)
        vmin_tensor, vmax_tensor = quantile_safe(flat, quantiles)
        vmin, vmax = vmin_tensor.item(), vmax_tensor.item()
    elif lic_percentiles is not None:
        vmin, vmax = lic_percentiles
    else:
        vmin, vmax = 0.0, 1.0

    for idx, conductor in enumerate(conductors):
        if conductor.id not in conductor_color_settings:
            continue

        settings = conductor_color_settings[conductor.id]

        # Apply smear to surface if enabled
        if settings.surface.smear_enabled:
            if conductor_masks is not None and idx < len(conductor_masks) and conductor_masks[idx] is not None:
                mask_cpu = conductor_masks[idx]
                mask_gpu = conductor_masks_gpu[idx] if conductor_masks_gpu and idx < len(conductor_masks_gpu) else None
                out = _apply_smear_to_region(
                    out, lic_gray_tensor, mask_cpu, mask_gpu, settings.surface,
                    render_shape, vmin, vmax, lut_tensor, brightness, contrast, gamma,
                    ex_tensor, ey_tensor, solution_gpu, global_lightness_expr
                )

        # Apply smear to interior if enabled
        if settings.interior.smear_enabled:
            if interior_masks is not None and idx < len(interior_masks) and interior_masks[idx] is not None:
                mask_cpu = interior_masks[idx]
                mask_gpu = interior_masks_gpu[idx] if interior_masks_gpu and idx < len(interior_masks_gpu) else None
                out = _apply_smear_to_region(
                    out, lic_gray_tensor, mask_cpu, mask_gpu, settings.interior,
                    render_shape, vmin, vmax, lut_tensor, brightness, contrast, gamma,
                    ex_tensor, ey_tensor, solution_gpu, global_lightness_expr
                )

    return torch.clamp(out, 0.0, 1.0)


# Legacy alias for compatibility
apply_conductor_smear_gpu = apply_region_smear_gpu


__all__ = ['apply_region_smear_gpu', 'apply_conductor_smear_gpu']
