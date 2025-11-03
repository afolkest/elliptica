"""Unified GPU postprocessing pipeline - chains all GPU operations together."""

import torch
import numpy as np
from typing import Tuple

from flowcol.gpu import GPUContext
from flowcol.gpu.pipeline import build_base_rgb_gpu
from flowcol.gpu.overlay import apply_region_overlays_gpu
from flowcol.gpu.smear import apply_conductor_smear_gpu
from flowcol.render import _get_palette_lut


def apply_full_postprocess_gpu(
    scalar_tensor: torch.Tensor,
    conductor_masks_cpu: list[np.ndarray] | None,
    interior_masks_cpu: list[np.ndarray] | None,
    conductor_color_settings: dict,
    conductors: list,
    render_shape: Tuple[int, int],
    canvas_resolution: Tuple[int, int],
    clip_percent: float,
    brightness: float,
    contrast: float,
    gamma: float,
    color_enabled: bool,
    palette: str,
    lic_percentiles: Tuple[float, float] | None = None,
    conductor_masks_gpu: list[torch.Tensor | None] | None = None,
    interior_masks_gpu: list[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Apply full postprocessing pipeline on GPU.

    This is the unified entry point that chains:
    1. Base RGB colorization (GPU)
    2. Region overlays (GPU)
    3. Conductor smear (GPU)

    Everything stays on GPU until the final result.

    Args:
        scalar_tensor: LIC grayscale field (H, W) on GPU
        conductor_masks_cpu: List of conductor masks (CPU arrays)
        interior_masks_cpu: List of interior masks (CPU arrays)
        conductor_color_settings: Per-conductor color settings
        conductors: List of Conductor objects
        render_shape: (height, width) of render resolution
        canvas_resolution: (width, height) of canvas
        clip_percent: Percentile clipping
        brightness: Brightness adjustment
        contrast: Contrast adjustment
        gamma: Gamma correction
        color_enabled: Whether to use color palette
        palette: Color palette name
        lic_percentiles: Precomputed (vmin, vmax) for smear normalization
        conductor_masks_gpu: Optional pre-uploaded GPU masks (avoids repeated CPU→GPU transfers)
        interior_masks_gpu: Optional pre-uploaded GPU interior masks

    Returns:
        Final RGB tensor (H, W, 3) in [0, 1] on GPU
    """
    # OPTIMIZATION: Compute percentile normalization ONCE and reuse
    # This avoids redundant 6-second percentile computation for each palette
    from flowcol.gpu.ops import percentile_clip_gpu
    normalized_tensor, _, _ = percentile_clip_gpu(scalar_tensor, clip_percent)

    # Step 1: Build base RGB colorization on GPU
    lut_tensor = None
    if color_enabled:
        lut_numpy = _get_palette_lut(palette)
        lut_tensor = GPUContext.to_gpu(lut_numpy)

    base_rgb = build_base_rgb_gpu(
        scalar_tensor,
        clip_percent,
        brightness,
        contrast,
        gamma,
        color_enabled,
        lut_tensor,
        normalized_tensor=normalized_tensor,  # Reuse normalized tensor
    )

    # Step 2: Apply region overlays (if any conductors have custom colors)
    # Must come BEFORE smear so that custom colors get smeared too!
    has_overlays = any(
        conductor.id in conductor_color_settings
        for conductor in conductors
    )

    if has_overlays and conductor_masks_cpu is not None and interior_masks_cpu is not None:
        # Upload masks to GPU (or use pre-uploaded masks if available)
        if conductor_masks_gpu is None or interior_masks_gpu is None:
            conductor_masks_gpu = []
            interior_masks_gpu = []

            for mask_cpu in conductor_masks_cpu:
                if mask_cpu is not None:
                    conductor_masks_gpu.append(GPUContext.to_gpu(mask_cpu))
                else:
                    conductor_masks_gpu.append(None)

            for mask_cpu in interior_masks_cpu:
                if mask_cpu is not None:
                    interior_masks_gpu.append(GPUContext.to_gpu(mask_cpu))
                else:
                    interior_masks_gpu.append(None)
        # else: use the pre-uploaded GPU masks (avoids repeated transfers!)

        base_rgb = apply_region_overlays_gpu(
            base_rgb,
            scalar_tensor,
            conductor_masks_gpu,
            interior_masks_gpu,
            conductor_color_settings,
            conductors,
            clip_percent,
            brightness,
            contrast,
            gamma,
            normalized_tensor=normalized_tensor,  # Reuse normalized tensor
        )

    # Step 3: Apply conductor smear (if any conductors have smear enabled)
    # Comes AFTER region overlays so smear applies to custom colored regions
    has_smear = any(c.smear_enabled for c in conductors)
    if has_smear and conductor_masks_cpu is not None:
        base_rgb = apply_conductor_smear_gpu(
            base_rgb,
            scalar_tensor,
            conductor_masks_cpu,
            conductors,
            render_shape,
            canvas_resolution,
            lut_tensor,
            lic_percentiles,
            conductor_color_settings,
            conductor_masks_gpu,  # Pass pre-uploaded GPU masks
            brightness,
            contrast,
            gamma,
        )

    return torch.clamp(base_rgb, 0.0, 1.0)


def apply_full_postprocess_hybrid(
    scalar_array: np.ndarray,
    conductor_masks: list[np.ndarray] | None,
    interior_masks: list[np.ndarray] | None,
    conductor_color_settings: dict,
    conductors: list,
    render_shape: Tuple[int, int],
    canvas_resolution: Tuple[int, int],
    clip_percent: float,
    brightness: float,
    contrast: float,
    gamma: float,
    color_enabled: bool,
    palette: str,
    lic_percentiles: Tuple[float, float] | None = None,
    use_gpu: bool = True,
    scalar_tensor: torch.Tensor | None = None,
    conductor_masks_gpu: list[torch.Tensor | None] | None = None,
    interior_masks_gpu: list[torch.Tensor | None] | None = None,
) -> np.ndarray:
    """Hybrid postprocessing pipeline with automatic GPU/CPU fallback.

    Args:
        scalar_array: LIC grayscale field (H, W) CPU array
        conductor_masks: List of conductor masks (CPU arrays)
        interior_masks: List of interior masks (CPU arrays)
        conductor_color_settings: Per-conductor color settings
        conductors: List of Conductor objects
        render_shape: (height, width) of render resolution
        canvas_resolution: (width, height) of canvas
        clip_percent: Percentile clipping
        brightness: Brightness adjustment
        contrast: Contrast adjustment
        gamma: Gamma correction
        color_enabled: Whether to use color palette
        palette: Color palette name
        lic_percentiles: Precomputed (vmin, vmax) for smear normalization
        use_gpu: Whether to attempt GPU acceleration
        scalar_tensor: Optional pre-uploaded scalar tensor on GPU (saves upload time)
        conductor_masks_gpu: Optional pre-uploaded GPU masks (avoids repeated CPU→GPU transfers)
        interior_masks_gpu: Optional pre-uploaded GPU interior masks

    Returns:
        Final RGB array (H, W, 3) uint8 on CPU
    """
    if use_gpu and GPUContext.is_available():
        # GPU path with error handling
        try:
            if scalar_tensor is None:
                scalar_tensor = GPUContext.to_gpu(scalar_array)

            rgb_tensor = apply_full_postprocess_gpu(
                scalar_tensor,
                conductor_masks,
                interior_masks,
                conductor_color_settings,
                conductors,
                render_shape,
                canvas_resolution,
                clip_percent,
                brightness,
                contrast,
                gamma,
                color_enabled,
                palette,
                lic_percentiles,
                conductor_masks_gpu,
                interior_masks_gpu,
            )

            # Convert to uint8 and download
            rgb_uint8_tensor = (rgb_tensor * 255.0).clamp(0, 255).to(torch.uint8)

            # Synchronize GPU (platform-specific)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()

            return GPUContext.to_cpu(rgb_uint8_tensor)

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # GPU operation failed (OOM, unsupported op) - try CPU device
            print(f"⚠️  GPU postprocessing failed ({e}), retrying with device='cpu'")

            # Retry with CPU device (PyTorch operations work on CPU too!)
            scalar_tensor_cpu = scalar_tensor.to('cpu') if scalar_tensor.device.type != 'cpu' else scalar_tensor

            rgb_tensor_cpu = apply_full_postprocess_gpu(
                scalar_tensor_cpu,
                conductor_masks,
                interior_masks,
                conductor_color_settings,
                conductors,
                render_shape,
                canvas_resolution,
                clip_percent,
                brightness,
                contrast,
                gamma,
                color_enabled,
                palette,
                lic_percentiles,
                None,  # GPU masks not available in fallback
                None,
            )

            # Convert to uint8 and return (already on CPU)
            rgb_uint8_tensor = (rgb_tensor_cpu * 255.0).clamp(0, 255).to(torch.uint8)
            return rgb_uint8_tensor.numpy()
    else:
        # No GPU available - use CPU device with PyTorch
        # PyTorch operations work fine on CPU, often faster than NumPy!
        scalar_tensor_cpu = torch.from_numpy(scalar_array).to(dtype=torch.float32, device='cpu')

        rgb_tensor_cpu = apply_full_postprocess_gpu(
            scalar_tensor_cpu,
            conductor_masks,
            interior_masks,
            conductor_color_settings,
            conductors,
            render_shape,
            canvas_resolution,
            clip_percent,
            brightness,
            contrast,
            gamma,
            color_enabled,
            palette,
            lic_percentiles,
        )

        # Convert to uint8 and return
        rgb_uint8_tensor = (rgb_tensor_cpu * 255.0).clamp(0, 255).to(torch.uint8)
        return rgb_uint8_tensor.numpy()


__all__ = [
    'apply_full_postprocess_gpu',
    'apply_full_postprocess_hybrid',
]
