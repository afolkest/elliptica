"""GPU-accelerated postprocessing pipeline functions."""

import torch
import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter, zoom

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
    clip_percent: float,
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
        clip_percent: Percentile clipping (e.g., 2.0 for 2%-98%)
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
        normalized, _, _ = percentile_clip_gpu(tensor, clip_percent)

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


def downsample_lic_hybrid(
    arr: np.ndarray,
    target_shape: Tuple[int, int],
    supersample: float,
    sigma: float,
    use_gpu: bool = True,
) -> np.ndarray:
    """Hybrid downsample with automatic GPU/CPU fallback.

    This is a drop-in replacement for the CPU-only downsample_lic.

    Args:
        arr: Input array (H, W)
        target_shape: Target (height, width)
        supersample: Supersample factor (not used in GPU path)
        sigma: Gaussian blur sigma
        use_gpu: Whether to attempt GPU acceleration

    Returns:
        Downsampled array (target_H, target_W)
    """
    if use_gpu and GPUContext.is_available():
        # GPU path
        tensor = GPUContext.to_gpu(arr)
        result_tensor = downsample_lic_gpu(tensor, target_shape, sigma)
        return GPUContext.to_cpu(result_tensor)
    else:
        # CPU fallback
        sigma = max(sigma, 0.0)
        filtered = gaussian_filter(arr, sigma=sigma) if sigma > 0 else arr

        if arr.shape == target_shape:
            return filtered.copy() if filtered is arr else filtered

        scale_y = target_shape[0] / filtered.shape[0]
        scale_x = target_shape[1] / filtered.shape[1]
        return zoom(filtered, (scale_y, scale_x), order=1)


__all__ = [
    'downsample_lic_gpu',
    'build_base_rgb_gpu',
    'downsample_lic_hybrid',
]
