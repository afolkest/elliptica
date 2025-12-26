"""GPU-accelerated image processing operations."""

import torch
import numpy as np
from torchvision.transforms.functional import gaussian_blur
from typing import Tuple


def quantile_safe(tensor: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Compute torch.quantile with GPU-friendly fallback.

    Torch's MPS backend currently raises a RuntimeError when quantile inputs
    exceed an internal size limit. When that happens we fall back to performing
    the computation on CPU and move the small result tensor back to the original
    device.
    """
    try:
        return torch.quantile(tensor, quantiles)
    except RuntimeError as error:
        message = str(error).lower()
        if "tensor is too large" in message:
            tensor_cpu = tensor.detach().to('cpu')
            quantiles_cpu = quantiles.detach().to('cpu')

            try:
                result_cpu = torch.quantile(tensor_cpu, quantiles_cpu)
            except RuntimeError as cpu_error:
                if "tensor is too large" not in str(cpu_error).lower():
                    raise
                # Final fallback: use NumPy which handles large reductions fine.
                result_np = np.quantile(
                    tensor_cpu.numpy(),
                    quantiles_cpu.numpy(),
                    method='linear',
                )
                result_cpu = torch.tensor(result_np, dtype=tensor_cpu.dtype)

            return result_cpu.to(tensor.device)
        raise


def gaussian_blur_gpu(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur on GPU using torchvision.

    Args:
        tensor: 2D tensor (H, W) on GPU
        sigma: Gaussian blur standard deviation

    Returns:
        Blurred tensor (H, W) on GPU
    """
    if sigma <= 0:
        return tensor

    # torchvision expects (B, C, H, W) format
    tensor_4d = tensor.unsqueeze(0).unsqueeze(0)

    # Calculate kernel size from sigma (rule of thumb: 6*sigma + 1, ensure odd)
    kernel_size = max(3, (int(6 * sigma)) | 1)

    blurred_4d = gaussian_blur(tensor_4d, kernel_size=[kernel_size, kernel_size], sigma=sigma)

    # Return to 2D format
    return blurred_4d.squeeze(0).squeeze(0)


def percentile_clip_gpu(
    tensor: torch.Tensor,
    clip_low_percent: float,
    clip_high_percent: float,
) -> Tuple[torch.Tensor, float, float]:
    """Clip and normalize tensor using percentile bounds.

    Args:
        tensor: 2D tensor (H, W) on GPU
        clip_low_percent: Percentage to clip from the low end (e.g., 0.5 for 0.5%)
        clip_high_percent: Percentage to clip from the high end (e.g., 0.5 for 99.5%)

    Returns:
        Tuple of (normalized tensor in [0,1], vmin, vmax)
    """
    low = max(float(clip_low_percent), 0.0)
    high = max(float(clip_high_percent), 0.0)

    if low <= 0.0 and high <= 0.0:
        vmin = tensor.min().item()
        vmax = tensor.max().item()
    else:
        # torch.quantile expects percentiles in [0, 1]
        lower = max(0.0, min(low / 100.0, 1.0))
        upper = max(0.0, min(1.0 - (high / 100.0), 1.0))
        if upper <= lower:
            vmin = tensor.min().item()
            vmax = tensor.max().item()
        else:
            quantiles = torch.tensor([lower, upper], device=tensor.device, dtype=tensor.dtype)
            vmin_tensor, vmax_tensor = quantile_safe(tensor.flatten(), quantiles)
            vmin = vmin_tensor.item()
            vmax = vmax_tensor.item()

    if vmax > vmin:
        normalized = torch.clamp((tensor - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        # Fallback: normalize by tensor's own range
        tmin = tensor.min()
        tmax = tensor.max()
        if tmax > tmin:
            normalized = (tensor - tmin) / (tmax - tmin)
        else:
            normalized = torch.zeros_like(tensor)

    return normalized, vmin, vmax


def apply_contrast_gamma_gpu(tensor: torch.Tensor, brightness: float, contrast: float, gamma: float) -> torch.Tensor:
    """Apply brightness, contrast and gamma correction on GPU.

    Args:
        tensor: Normalized tensor in [0, 1] (H, W) on GPU
        brightness: Brightness adjustment (0.0 = no change, additive)
        contrast: Contrast multiplier (1.0 = no change)
        gamma: Gamma exponent (1.0 = no change)

    Returns:
        Adjusted tensor in [0, 1] on GPU
    """
    result = tensor

    # Contrast adjustment: (val - 0.5) * contrast + 0.5
    if contrast != 1.0:
        result = (result - 0.5) * contrast + 0.5
        result = torch.clamp(result, 0.0, 1.0)

    # Brightness adjustment (after contrast, before gamma)
    if brightness != 0.0:
        result = result + brightness
        result = torch.clamp(result, 0.0, 1.0)

    # Gamma correction: val ^ gamma
    if gamma != 1.0:
        result = torch.pow(result, gamma)
        result = torch.clamp(result, 0.0, 1.0)

    return result


def apply_palette_lut_gpu(tensor: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Apply color palette lookup table on GPU.

    Args:
        tensor: Normalized tensor in [0, 1] (H, W) on GPU
        lut: Color lookup table (N, 3) with RGB values in [0, 1] on GPU

    Returns:
        RGB tensor (H, W, 3) with values in [0, 1] on GPU
    """
    # Clamp input to [0, 1] to ensure valid indices
    tensor_clamped = torch.clamp(tensor, 0.0, 1.0)

    # Map [0, 1] to [0, lut_size-1] indices
    lut_size = lut.shape[0]
    indices = (tensor_clamped * (lut_size - 1)).long()

    # Index into LUT - this is vectorized and very fast on GPU
    rgb = lut[indices.flatten()]  # (H*W, 3)

    # Reshape back to (H, W, 3)
    h, w = tensor.shape
    rgb = rgb.reshape(h, w, 3)

    return rgb


def grayscale_to_rgb_gpu(tensor: torch.Tensor) -> torch.Tensor:
    """Convert grayscale tensor to RGB on GPU.

    Args:
        tensor: Normalized tensor in [0, 1] (H, W) on GPU

    Returns:
        RGB tensor (H, W, 3) with values in [0, 1] on GPU
    """
    # Stack the same values across 3 channels
    rgb = tensor.unsqueeze(-1).expand(-1, -1, 3)
    return rgb


def apply_highpass_gpu(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian high-pass filter on GPU.

    Subtracts Gaussian blur from the input tensor to emphasize high-frequency details.

    Args:
        tensor: 2D tensor (H, W) on GPU
        sigma: Gaussian blur standard deviation

    Returns:
        High-pass filtered tensor (H, W) on GPU with unbounded range
    """
    if sigma <= 0:
        return tensor.clone()

    # High-pass = original - lowpass(blur)
    blurred = gaussian_blur_gpu(tensor, sigma)
    return tensor - blurred


__all__ = [
    'gaussian_blur_gpu',
    'percentile_clip_gpu',
    'apply_contrast_gamma_gpu',
    'apply_palette_lut_gpu',
    'grayscale_to_rgb_gpu',
    'apply_highpass_gpu',
    'quantile_safe',
]
