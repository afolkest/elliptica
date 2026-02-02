"""GPU-accelerated postprocessing pipeline functions."""

import torch
from typing import Tuple

from elliptica.gpu import GPUContext
from elliptica.gpu.ops import (
    gaussian_blur_gpu,
    percentile_clip_gpu,
    apply_contrast_gamma_gpu,
    apply_palette_lut_gpu,
    grayscale_to_rgb_gpu,
)


def downsample_lic_gpu(
    tensor: torch.Tensor,
    target_shape: Tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """GPU-accelerated Gaussian blur + bilinear resize.

    Args:
        tensor: Input tensor (H, W) on GPU
        target_shape: Target (height, width)
        sigma: Gaussian blur sigma

    Returns:
        Downsampled tensor (target_H, target_W) on GPU
    """
    # Apply Gaussian blur on GPU
    if sigma > 0:
        blurred = gaussian_blur_gpu(tensor, sigma)
    else:
        blurred = tensor

    # Skip resize if shapes already match
    if blurred.shape == target_shape:
        return blurred.clone() if blurred is tensor else blurred

    # Use PyTorch's interpolate for bilinear resizing
    # Needs (B, C, H, W) format
    blurred_4d = blurred.unsqueeze(0).unsqueeze(0)

    # mode='bilinear' for smooth interpolation, align_corners=False matches scipy behavior
    resized_4d = torch.nn.functional.interpolate(
        blurred_4d,
        size=target_shape,
        mode='bilinear',
        align_corners=False,
    )

    # Convert back to 2D
    return resized_4d.squeeze(0).squeeze(0)


def build_base_rgb_gpu(
    tensor: torch.Tensor,
    clip_low_percent: float,
    clip_high_percent: float,
    brightness: float,
    contrast: float,
    gamma: float,
    color_enabled: bool,
    lut: torch.Tensor | None,
    normalized_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """GPU-accelerated colorization pipeline.

    Args:
        tensor: Input grayscale tensor (H, W) on GPU
        clip_low_percent: Percentile clipping from low end (e.g., 0.5 for 0.5%)
        clip_high_percent: Percentile clipping from high end (e.g., 0.5 for 99.5%)
        brightness: Brightness adjustment (0.0 = no change, additive)
        contrast: Contrast multiplier
        gamma: Gamma exponent
        color_enabled: Whether to apply color palette
        lut: Color lookup table (N, 3) on GPU, or None for grayscale
        normalized_tensor: Optional pre-normalized tensor (skips expensive percentile computation)

    Returns:
        RGB tensor (H, W, 3) with values in [0, 1] on GPU
    """
    # Percentile clip and normalize to [0, 1] (unless already provided)
    if normalized_tensor is not None:
        normalized = normalized_tensor
    else:
        normalized, _, _ = percentile_clip_gpu(tensor, clip_low_percent, clip_high_percent)

    # Apply brightness, contrast and gamma
    adjusted = apply_contrast_gamma_gpu(normalized, brightness, contrast, gamma)

    # Convert to RGB
    if color_enabled and lut is not None:
        # Apply color palette
        rgb = apply_palette_lut_gpu(adjusted, lut)
    else:
        # Grayscale mode
        rgb = grayscale_to_rgb_gpu(adjusted)

    return rgb


__all__ = [
    'downsample_lic_gpu',
    'build_base_rgb_gpu',
]
