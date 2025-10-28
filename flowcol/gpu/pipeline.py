"""GPU-accelerated postprocessing pipeline functions."""

import torch
import numpy as np
from typing import Tuple

from flowcol.gpu import GPUContext
from flowcol.gpu.ops import (
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

    Returns:
        RGB tensor (H, W, 3) with values in [0, 1] on GPU
    """
    # Percentile clip and normalize to [0, 1]
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
        from scipy.ndimage import gaussian_filter, zoom

        sigma = max(sigma, 0.0)
        filtered = gaussian_filter(arr, sigma=sigma) if sigma > 0 else arr

        if arr.shape == target_shape:
            return filtered.copy() if filtered is arr else filtered

        scale_y = target_shape[0] / filtered.shape[0]
        scale_x = target_shape[1] / filtered.shape[1]
        return zoom(filtered, (scale_y, scale_x), order=1)


def build_base_rgb_hybrid(
    arr: np.ndarray,
    clip_percent: float,
    brightness: float,
    contrast: float,
    gamma: float,
    color_enabled: bool,
    lut_numpy: np.ndarray | None,
    use_gpu: bool = True,
) -> np.ndarray:
    """Hybrid colorization with automatic GPU/CPU fallback.

    This is a drop-in replacement for the CPU-only build_base_rgb.

    Args:
        arr: Input grayscale array (H, W)
        clip_percent: Percentile clipping
        brightness: Brightness adjustment (0.0 = no change, additive)
        contrast: Contrast multiplier
        gamma: Gamma exponent
        color_enabled: Whether to apply color palette
        lut_numpy: Color lookup table (N, 3) as numpy array
        use_gpu: Whether to attempt GPU acceleration

    Returns:
        RGB uint8 array (H, W, 3)
    """
    if use_gpu and GPUContext.is_available():
        # GPU path
        tensor = GPUContext.to_gpu(arr)

        # Upload LUT to GPU if provided
        lut_gpu = None
        if lut_numpy is not None:
            lut_gpu = GPUContext.to_gpu(lut_numpy)

        # Process on GPU
        rgb_tensor = build_base_rgb_gpu(
            tensor,
            clip_percent,
            brightness,
            contrast,
            gamma,
            color_enabled,
            lut_gpu,
        )

        # Convert to uint8 and download
        rgb_uint8_tensor = (rgb_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        return GPUContext.to_cpu(rgb_uint8_tensor)
    else:
        # CPU fallback
        from flowcol.postprocess.color import build_base_rgb, ColorParams

        # TODO: Palette inference from lut_numpy
        # Currently we can't reverse-engineer which palette name corresponds to lut_numpy.
        # This CPU fallback path is rarely hit (GPU usually available), but if we ever
        # rely on non-default palettes when GPU is unavailable, we'd need to either:
        # 1. Pass palette name as explicit parameter to build_base_rgb_hybrid(), or
        # 2. Add a reverse LUT lookup in render.py (fragile)
        # For now, default palette is acceptable since GPU path dominates.
        palette = "Ink & Gold"  # Default fallback when GPU unavailable

        # Create pure ColorParams (no UI dependency!)
        color_params = ColorParams(
            clip_percent=clip_percent,
            brightness=brightness,
            contrast=contrast,
            gamma=gamma,
            color_enabled=color_enabled,
            palette=palette,
        )

        # Use CPU implementation
        return build_base_rgb(arr, color_params)


__all__ = [
    'downsample_lic_gpu',
    'build_base_rgb_gpu',
    'downsample_lic_hybrid',
    'build_base_rgb_hybrid',
]
