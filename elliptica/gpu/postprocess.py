"""Unified GPU postprocessing pipeline - chains all GPU operations together."""

import torch
import numpy as np
from typing import Tuple, TYPE_CHECKING

from elliptica.gpu import GPUContext
from elliptica.gpu.pipeline import build_base_rgb_gpu
from elliptica.gpu.overlay import apply_region_overlays_gpu
from elliptica.gpu.smear import apply_region_smear_gpu, _has_any_smear_enabled
from elliptica.render import _get_palette_lut

if TYPE_CHECKING:
    from elliptica.colorspace import ColorConfig


def apply_lightness_expr_gpu(
    rgb: torch.Tensor,
    lightness_expr: str,
    lic_tensor: torch.Tensor,
    ex_tensor: torch.Tensor | None = None,
    ey_tensor: torch.Tensor | None = None,
    solution_gpu: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply a lightness expression as a multiplier to RGB colors.

    Converts RGB → OKLch, multiplies L by expression result, gamut maps back to RGB.

    Args:
        rgb: RGB tensor (H, W, 3) in [0, 1]
        lightness_expr: Expression string that evaluates to a multiplier
        lic_tensor: LIC texture (H, W) for 'lic' variable
        ex_tensor: Field X component (H, W) for 'ex' variable
        ey_tensor: Field Y component (H, W) for 'ey' variable
        solution_gpu: PDE solution fields (phi, etc.) on GPU

    Returns:
        RGB tensor (H, W, 3) in [0, 1]
    """
    from elliptica.colorspace.oklch import srgb_to_oklch
    from elliptica.colorspace.gamut import gamut_map_to_srgb
    from elliptica.expr import compile_expression

    target_shape = lic_tensor.shape  # (H, W)

    # Build bindings for expression evaluation
    bindings = {'lic': lic_tensor}
    if ex_tensor is not None and ey_tensor is not None:
        bindings['ex'] = ex_tensor
        bindings['ey'] = ey_tensor
        bindings['mag'] = torch.sqrt(ex_tensor**2 + ey_tensor**2)

    # Add PDE solution fields (phi, etc.) - resize to match LIC shape if needed
    if solution_gpu:
        for name, tensor in solution_gpu.items():
            if name not in bindings:  # Don't override standard bindings
                if tensor.shape != target_shape:
                    # Resize using bilinear interpolation
                    tensor_4d = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    resized = torch.nn.functional.interpolate(
                        tensor_4d, size=target_shape, mode='bilinear', align_corners=False
                    )
                    tensor = resized.squeeze(0).squeeze(0)  # Back to (H, W)
                bindings[name] = tensor

    # Compile and evaluate expression
    expr_fn = compile_expression(lightness_expr)
    multiplier = expr_fn(bindings)

    # Convert RGB to OKLch
    L, C, H = srgb_to_oklch(rgb)

    # Apply multiplier to L
    L_adjusted = L * multiplier

    # Gamut map back to RGB
    return gamut_map_to_srgb(L_adjusted, C, H, method='compress')


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
    color_config: "ColorConfig | None" = None,
    ex_tensor: torch.Tensor | None = None,
    ey_tensor: torch.Tensor | None = None,
    lightness_expr: str | None = None,
    solution_gpu: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Tuple[float, float]]:
    """Apply full postprocessing pipeline on GPU.

    This is the unified entry point that chains:
    1. Base RGB colorization (GPU) - via palette LUT or ColorConfig expressions
    2. Region overlays (GPU) - via conductor_color_settings or ColorConfig regions
    3. Conductor smear (GPU) - optional, palette mode only

    Everything stays on GPU until the final result.

    Args:
        scalar_tensor: LIC grayscale field (H, W) on GPU
        conductor_masks_cpu: List of conductor masks (CPU arrays)
        interior_masks_cpu: List of interior masks (CPU arrays)
        conductor_color_settings: Per-conductor color settings (legacy palette mode)
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
        color_config: Optional ColorConfig for expression-based coloring.
                     When provided, uses OKLCH expressions instead of palette mode.
                     Ignores brightness/contrast/gamma/palette params.
        ex_tensor: Electric field X component (H, W) on GPU, for ColorConfig mag/ex bindings
        ey_tensor: Electric field Y component (H, W) on GPU, for ColorConfig mag/ey bindings

    Returns:
        Tuple of:
            - Final RGB tensor (H, W, 3) in [0, 1] on GPU
            - Tuple (vmin, vmax) percentiles actually used for normalization
    """
    # OPTIMIZATION: Reuse cached percentiles when they match the requested clip%.
    # Percentile computation can be quite expensive at large resolutions.
    from elliptica.gpu.ops import percentile_clip_gpu

    # Determine whether we already have percentiles that match the requested clip%.
    cached_clip = getattr(scalar_tensor, '_lic_cached_clip_percent', None)
    cached_percentiles_attr = getattr(scalar_tensor, '_lic_cached_percentiles', None)
    percentiles_match_attr = (
        cached_clip is not None and
        cached_percentiles_attr is not None and
        abs(float(cached_clip) - float(clip_percent)) < 0.01
    )

    cached_percentiles: Tuple[float, float] | None = None
    if percentiles_match_attr:
        cached_percentiles = cached_percentiles_attr  # type: ignore[assignment]
    elif lic_percentiles is not None:
        # Caller-provided percentiles are assumed to match clip_percent (caller verifies this).
        cached_percentiles = lic_percentiles

    if cached_percentiles is not None:
        vmin, vmax = cached_percentiles
        if vmax > vmin:
            normalized_tensor = torch.clamp((scalar_tensor - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            normalized_tensor = torch.zeros_like(scalar_tensor)
        used_percentiles = (vmin, vmax)
    else:
        # Slow path: compute fresh percentiles for this clip%.
        normalized_tensor, vmin, vmax = percentile_clip_gpu(scalar_tensor, clip_percent)
        used_percentiles = (vmin, vmax)

    # Persist the latest clip% and percentile bounds directly on the tensor for future reuse.
    scalar_tensor._lic_cached_clip_percent = float(clip_percent)
    scalar_tensor._lic_cached_percentiles = used_percentiles

    # === ColorConfig path: expression-based OKLCH coloring ===
    if color_config is not None:
        from elliptica.colorspace.pipeline import render_with_color_config_gpu

        # Upload masks to GPU if not already done
        if conductor_masks_gpu is None and conductor_masks_cpu is not None:
            conductor_masks_gpu = [
                GPUContext.to_gpu(m) if m is not None else None
                for m in conductor_masks_cpu
            ]
        if interior_masks_gpu is None and interior_masks_cpu is not None:
            interior_masks_gpu = [
                GPUContext.to_gpu(m) if m is not None else None
                for m in interior_masks_cpu
            ]

        # Render using ColorConfig (handles base + regions)
        base_rgb = render_with_color_config_gpu(
            color_config,
            scalar_tensor,
            ex_tensor=ex_tensor,
            ey_tensor=ey_tensor,
            conductor_masks_gpu=conductor_masks_gpu,
            interior_masks_gpu=interior_masks_gpu,
            conductors=conductors,
            solution=solution_gpu,  # Pass GPU tensors directly
        )

        # Note: Smear is not supported in ColorConfig mode (uses palette-specific logic)
        # Future: add smear support for ColorConfig if needed

        return torch.clamp(base_rgb, 0.0, 1.0), used_percentiles

    # === Palette path: LUT-based coloring ===
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

    # Keep original base RGB before global lightness expr (needed for per-region custom exprs)
    base_rgb_no_expr = base_rgb

    # Step 1b: Apply lightness expression if set (palette mode only)
    if lightness_expr is not None:
        base_rgb = apply_lightness_expr_gpu(
            base_rgb,
            lightness_expr,
            scalar_tensor,
            ex_tensor,
            ey_tensor,
            solution_gpu,
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
            ex_tensor=ex_tensor,
            ey_tensor=ey_tensor,
            solution_gpu=solution_gpu,
            global_lightness_expr=lightness_expr,
            base_rgb_no_expr=base_rgb_no_expr,  # For custom expr regions without palette override
        )

    # Step 3: Apply region smear (if any region has smear enabled)
    # Comes AFTER region overlays so smear applies to custom colored regions
    has_smear = _has_any_smear_enabled(conductor_color_settings, conductors)
    if has_smear:
        base_rgb = apply_region_smear_gpu(
            base_rgb,
            scalar_tensor,
            conductor_masks_cpu,
            interior_masks_cpu,
            conductors,
            render_shape,
            canvas_resolution,
            lut_tensor,
            used_percentiles,
            conductor_color_settings,
            conductor_masks_gpu,
            interior_masks_gpu,
            brightness,
            contrast,
            gamma,
            ex_tensor,
            ey_tensor,
            solution_gpu,
            lightness_expr,
        )

    return torch.clamp(base_rgb, 0.0, 1.0), used_percentiles


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
    color_config: "ColorConfig | None" = None,
    ex_tensor: torch.Tensor | None = None,
    ey_tensor: torch.Tensor | None = None,
    lightness_expr: str | None = None,
    solution_gpu: dict[str, torch.Tensor] | None = None,
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
        color_config: Optional ColorConfig for expression-based coloring
        ex_tensor: Electric field X component on GPU (for ColorConfig)
        ey_tensor: Electric field Y component on GPU (for ColorConfig)

    Returns:
        Tuple of:
            - Final RGB array (H, W, 3) uint8 on CPU
            - Tuple (vmin, vmax) percentiles used for normalization
    """
    if use_gpu and GPUContext.is_available():
        # GPU path with error handling
        try:
            if scalar_tensor is None:
                scalar_tensor = GPUContext.to_gpu(scalar_array)

            rgb_tensor, used_percentiles = apply_full_postprocess_gpu(
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
                color_config,
                ex_tensor,
                ey_tensor,
                lightness_expr,
                solution_gpu,
            )

            # Convert to uint8 and download
            rgb_uint8_tensor = (rgb_tensor * 255.0).clamp(0, 255).to(torch.uint8)

            # Synchronize GPU (platform-specific)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()

            return GPUContext.to_cpu(rgb_uint8_tensor), used_percentiles

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # GPU operation failed (OOM, unsupported op) - try CPU device
            print(f"⚠️  GPU postprocessing failed ({e}), retrying with device='cpu'")

            # Retry with CPU device (PyTorch operations work on CPU too!)
            scalar_tensor_cpu = scalar_tensor.to('cpu') if scalar_tensor.device.type != 'cpu' else scalar_tensor
            ex_cpu = ex_tensor.to('cpu') if ex_tensor is not None and ex_tensor.device.type != 'cpu' else ex_tensor
            ey_cpu = ey_tensor.to('cpu') if ey_tensor is not None and ey_tensor.device.type != 'cpu' else ey_tensor

            # Move solution tensors to CPU if needed
            solution_cpu = None
            if solution_gpu:
                solution_cpu = {k: v.to('cpu') if v.device.type != 'cpu' else v for k, v in solution_gpu.items()}

            rgb_tensor_cpu, used_percentiles = apply_full_postprocess_gpu(
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
                color_config,
                ex_cpu,
                ey_cpu,
                lightness_expr,
                solution_cpu,
            )

            # Convert to uint8 and return (already on CPU)
            rgb_uint8_tensor = (rgb_tensor_cpu * 255.0).clamp(0, 255).to(torch.uint8)
            return rgb_uint8_tensor.numpy(), used_percentiles
    else:
        # No GPU available - use CPU device with PyTorch
        # PyTorch operations work fine on CPU, often faster than NumPy!
        scalar_tensor_cpu = torch.from_numpy(scalar_array).to(dtype=torch.float32, device='cpu')

        rgb_tensor_cpu, used_percentiles = apply_full_postprocess_gpu(
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
            color_config=color_config,
            ex_tensor=ex_tensor,
            ey_tensor=ey_tensor,
            lightness_expr=lightness_expr,
            solution_gpu=solution_gpu,
        )

        # Convert to uint8 and return
        rgb_uint8_tensor = (rgb_tensor_cpu * 255.0).clamp(0, 255).to(torch.uint8)
        return rgb_uint8_tensor.numpy(), used_percentiles


__all__ = [
    'apply_full_postprocess_gpu',
    'apply_full_postprocess_hybrid',
]
