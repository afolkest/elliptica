"""GPU-accelerated region overlay blending."""

import torch
import numpy as np
from typing import Tuple

from flowcol.gpu import GPUContext
from flowcol.gpu.pipeline import build_base_rgb_gpu


def blend_region_gpu(
    base: torch.Tensor,
    overlay: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Blend overlay over base using mask as alpha on GPU.

    Args:
        base: Base RGB tensor (H, W, 3) in [0, 1] on GPU
        overlay: Overlay RGB tensor (H, W, 3) in [0, 1] on GPU
        mask: Float mask (H, W) in [0, 1] on GPU

    Returns:
        Blended RGB tensor (H, W, 3) in [0, 1] on GPU
    """
    alpha = mask.unsqueeze(-1)  # (H, W, 1) for broadcasting
    blended = base * (1.0 - alpha) + overlay * alpha
    return torch.clamp(blended, 0.0, 1.0)


def fill_region_gpu(
    base: torch.Tensor,
    color_rgb: Tuple[float, float, float],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fill region with solid color using mask on GPU.

    Args:
        base: Base RGB tensor (H, W, 3) in [0, 1] on GPU
        color_rgb: Solid color as (r, g, b) in [0, 1]
        mask: Float mask (H, W) in [0, 1] on GPU

    Returns:
        Filled RGB tensor (H, W, 3) in [0, 1] on GPU
    """
    # Create color tensor on GPU
    device = base.device
    color_tensor = torch.tensor(color_rgb, dtype=torch.float32, device=device)

    # Broadcast color to full image size
    h, w = base.shape[:2]
    color_full = color_tensor.view(1, 1, 3).expand(h, w, 3)

    # Blend using mask
    alpha = mask.unsqueeze(-1)
    filled = base * (1.0 - alpha) + color_full * alpha
    return torch.clamp(filled, 0.0, 1.0)


def apply_region_overlays_gpu(
    base_rgb: torch.Tensor,
    scalar_tensor: torch.Tensor,
    conductor_masks: list[torch.Tensor],
    interior_masks: list[torch.Tensor],
    conductor_color_settings: dict,
    conductors: list,
    clip_percent: float,
    brightness: float,
    contrast: float,
    gamma: float,
) -> torch.Tensor:
    """Composite per-region color overrides over base RGB on GPU.

    Args:
        base_rgb: Base RGB tensor (H, W, 3) in [0, 1] on GPU
        scalar_tensor: Original scalar LIC tensor (H, W) on GPU
        conductor_masks: List of surface mask tensors on GPU
        interior_masks: List of interior mask tensors on GPU
        conductor_color_settings: dict[conductor_id -> ConductorColorSettings]
        conductors: List of Conductor objects
        clip_percent: Percentile clipping for colorization
        brightness: Brightness adjustment
        contrast: Contrast adjustment
        gamma: Gamma correction

    Returns:
        Final composited RGB tensor (H, W, 3) in [0, 1] on GPU
    """
    result = base_rgb.clone()

    # OPTIMIZATION: Pre-compute RGB for each unique palette ONCE on GPU
    palette_cache: dict[str, torch.Tensor] = {}

    # Collect unique palettes needed
    unique_palettes = set()
    for conductor in conductors:
        if conductor.id is None:
            continue
        settings = conductor_color_settings.get(conductor.id)
        if settings is None:
            continue

        if settings.interior.enabled and settings.interior.use_palette:
            unique_palettes.add(settings.interior.palette)
        if settings.surface.enabled and settings.surface.use_palette:
            unique_palettes.add(settings.surface.palette)

    # Pre-compute RGB for each unique palette on GPU
    from flowcol.render import _get_palette_lut
    for palette_name in unique_palettes:
        lut_numpy = _get_palette_lut(palette_name)
        lut_tensor = GPUContext.to_gpu(lut_numpy)

        # Build full RGB using GPU pipeline (stays on GPU!)
        palette_cache[palette_name] = build_base_rgb_gpu(
            scalar_tensor,
            clip_percent,
            brightness,
            contrast,
            gamma,
            color_enabled=True,
            lut=lut_tensor,
        )

    # Blend each region using cached palette RGB
    for idx, conductor in enumerate(conductors):
        if conductor.id is None:
            continue

        settings = conductor_color_settings.get(conductor.id)
        if settings is None:
            continue

        # Apply interior region first (lower layer)
        if settings.interior.enabled and idx < len(interior_masks) and interior_masks[idx] is not None:
            mask = interior_masks[idx]
            if torch.any(mask > 0):
                if settings.interior.use_palette:
                    # Use pre-computed palette RGB (stays on GPU!)
                    region_rgb = palette_cache[settings.interior.palette]
                    result = blend_region_gpu(result, region_rgb, mask)
                else:
                    # Solid color fill
                    result = fill_region_gpu(result, settings.interior.solid_color, mask)

        # Apply surface region (upper layer)
        if settings.surface.enabled and idx < len(conductor_masks) and conductor_masks[idx] is not None:
            mask = conductor_masks[idx]
            if torch.any(mask > 0):
                if settings.surface.use_palette:
                    # Use pre-computed palette RGB (stays on GPU!)
                    region_rgb = palette_cache[settings.surface.palette]
                    result = blend_region_gpu(result, region_rgb, mask)
                else:
                    # Solid color fill
                    result = fill_region_gpu(result, settings.surface.solid_color, mask)

    return torch.clamp(result, 0.0, 1.0)


__all__ = [
    'blend_region_gpu',
    'fill_region_gpu',
    'apply_region_overlays_gpu',
]
