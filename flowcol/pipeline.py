"""Pure render pipeline orchestration - no UI dependencies."""

import numpy as np
from dataclasses import dataclass
import time
import os
from flowcol.types import Project
from flowcol.field_pde import compute_field_pde
from flowcol.postprocess.masks import rasterize_conductor_masks
from flowcol.render import (
    compute_lic,
    downsample_lic,
    apply_gaussian_highpass,
    list_color_palettes,
)


MAX_RENDER_DIM = 32768


@dataclass
class RenderResult:
    """Result from rendering operation."""

    array: np.ndarray
    compute_resolution: tuple[int, int]
    canvas_scaled_shape: tuple[int, int]
    margin: float
    offset_x: int = 0  # Crop offset for mask alignment
    offset_y: int = 0  # Crop offset for mask alignment
    poisson_scale: float = 1.0  # Poisson solve scale relative to render grid
    ex: np.ndarray | None = None  # Electric field X component (for anisotropic blur)
    ey: np.ndarray | None = None  # Electric field Y component (for anisotropic blur)
    # Cached conductor masks at canvas resolution (saves redundant rasterization)
    conductor_masks_canvas: list[np.ndarray] | None = None  # Surface masks
    interior_masks_canvas: list[np.ndarray] | None = None  # Interior masks
    # Solution dict from PDE solver (for noise correlation, multiple field extractions, etc.)
    solution: dict[str, np.ndarray] | None = None


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""

    detail_enabled: bool = False
    detail_sigma_factor: float = 0.02


def get_palette_name(palette_index: int) -> str | None:
    """Get palette name from index."""
    palettes = list_color_palettes()
    if not palettes:
        return None
    idx = palette_index % len(palettes)
    return palettes[idx]


def compute_reference_resolution(
    compute_resolution: tuple[int, int],
    canvas_resolution: tuple[int, int],
    render_multiplier: float,
    supersample: float,
) -> float:
    """Compute reference resolution for sigma calculations."""
    compute_h, compute_w = compute_resolution
    if compute_h > 0 and compute_w > 0:
        return float(min(compute_h, compute_w))
    canvas_min = float(min(canvas_resolution))
    if render_multiplier > 0:
        return canvas_min * render_multiplier * supersample
    return canvas_min


def apply_postprocess(
    original: np.ndarray,
    config: PostProcessConfig,
    reference_size: float,
) -> np.ndarray:
    """Apply post-processing pipeline to rendered array.

    Returns processed array without mutating input.
    """
    working = original.astype(np.float32, copy=True)

    if config.detail_enabled:
        detail_sigma = max(config.detail_sigma_factor, 0.0) * reference_size
        if detail_sigma > 0.0:
            working = apply_gaussian_highpass(working, detail_sigma)

    return working


def downsample_for_display(
    highres: np.ndarray,
    target_shape: tuple[int, int],
    supersample: float,
    sigma_factor: float,
) -> np.ndarray:
    """Downsample high-resolution array for display."""
    sigma = sigma_factor * supersample
    return downsample_lic(highres, target_shape, supersample, sigma)


def perform_render(
    project: Project,
    multiplier: float,
    supersample_factor: float,
    num_passes: int,
    margin_factor: float,
    noise_seed: int,
    noise_sigma: float,
    streamlength_factor: float,
    use_mask: bool = True,
    edge_gain_strength: float = 0.0,
    edge_gain_power: float = 2.0,
    poisson_scale: float = 1.0,
) -> RenderResult | None:
    """Execute full render pipeline.

    Pure function - takes parameters, returns rendered array + metadata.
    No state mutation, no pygame dependencies.
    """
    t_start = time.time()

    canvas_w, canvas_h = project.canvas_resolution

    margin_physical = margin_factor * float(min(canvas_w, canvas_h))
    margin_tuple = (margin_physical, margin_physical)
    domain_w = canvas_w + 2.0 * margin_physical
    domain_h = canvas_h + 2.0 * margin_physical
    scale = multiplier * supersample_factor
    compute_w = max(1, int(round(domain_w * scale)))
    compute_h = max(1, int(round(domain_h * scale)))

    if compute_w > MAX_RENDER_DIM or compute_h > MAX_RENDER_DIM:
        return None

    preview_note = "" if poisson_scale >= 0.999 else f" (preview scale {poisson_scale:.2f})"
    print(f"Starting PDE solve ({compute_w}×{compute_h}){preview_note}...")
    t_poisson_start = time.time()
    solution, (ex, ey) = compute_field_pde(
        project,
        multiplier,
        supersample_factor,
        margin_tuple,
        boundary_top=project.boundary_top,
        boundary_bottom=project.boundary_bottom,
        boundary_left=project.boundary_left,
        boundary_right=project.boundary_right,
        poisson_scale=poisson_scale,
    )
    t_poisson_end = time.time()
    print(f"  PDE solve completed in {t_poisson_end - t_poisson_start:.2f}s")

    # Generate conductor mask for LIC blocking if enabled
    lic_mask = None
    if use_mask and project.conductors:
        from flowcol.mask_utils import blur_mask
        from scipy.ndimage import zoom

        lic_mask = np.zeros((compute_h, compute_w), dtype=bool)
        scale_x = compute_w / domain_w if domain_w > 0 else 1.0
        scale_y = compute_h / domain_h if domain_h > 0 else 1.0

        for conductor in project.conductors:
            x = (conductor.position[0] + margin_physical) * scale_x
            y = (conductor.position[1] + margin_physical) * scale_y

            # Scale and blur mask (same logic as field.py)
            if not np.isclose(scale_x, 1.0) or not np.isclose(scale_y, 1.0):
                scaled_mask = zoom(conductor.mask, (scale_y, scale_x), order=0)
            else:
                scaled_mask = conductor.mask

            scale_factor = (scale_x + scale_y) / 2.0
            scaled_sigma = conductor.edge_smooth_sigma * scale_factor
            scaled_mask = blur_mask(scaled_mask, scaled_sigma)

            mask_h, mask_w = scaled_mask.shape
            ix, iy = int(round(x)), int(round(y))
            x0, y0 = max(0, ix), max(0, iy)
            x1, y1 = min(ix + mask_w, compute_w), min(iy + mask_h, compute_h)

            mx0, my0 = max(0, -ix), max(0, -iy)
            mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

            mask_slice = scaled_mask[my0:my1, mx0:mx1]
            lic_mask[y0:y1, x0:x1] |= (mask_slice > 0.5)

    num_passes = max(1, num_passes)
    min_compute = min(compute_w, compute_h)
    streamlength_pixels = max(int(round(streamlength_factor * min_compute)), 1)

    # Determine thread count (None means auto-detect from CPU count)
    from flowcol import defaults
    num_threads = defaults.DEFAULT_NUM_THREADS if defaults.DEFAULT_NUM_THREADS is not None else os.cpu_count()

    mask_status = "with mask blocking" if lic_mask is not None else "no mask"
    halo_status = f", edge_gain={edge_gain_strength:.2f}" if edge_gain_strength > 0 else ""
    print(f"Starting LIC ({num_passes} passes, streamlength={streamlength_pixels}, {mask_status}{halo_status}, threads={num_threads})...")
    t_lic_start = time.time()
    lic_array = compute_lic(
        ex,
        ey,
        streamlength_pixels,
        num_passes=num_passes,
        seed=noise_seed,
        noise_sigma=noise_sigma,
        mask=lic_mask,
        edge_gain_strength=edge_gain_strength,
        edge_gain_power=edge_gain_power,
    )
    t_lic_end = time.time()
    print(f"  LIC completed in {t_lic_end - t_lic_start:.2f}s")

    canvas_scaled_w = max(1, int(round(canvas_w * scale)))
    canvas_scaled_h = max(1, int(round(canvas_h * scale)))
    offset_x = int(round(margin_physical * scale))
    offset_y = int(round(margin_physical * scale))
    offset_x = min(offset_x, max(0, lic_array.shape[1] - canvas_scaled_w))
    offset_y = min(offset_y, max(0, lic_array.shape[0] - canvas_scaled_h))
    crop_x0 = max(0, offset_x)
    crop_y0 = max(0, offset_y)
    crop_x1 = min(crop_x0 + canvas_scaled_w, lic_array.shape[1])
    crop_y1 = min(crop_y0 + canvas_scaled_h, lic_array.shape[0])
    lic_cropped = lic_array[crop_y0:crop_y1, crop_x0:crop_x1]

    # Crop E-field arrays to match LIC
    ex_cropped = ex[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.float32, copy=True)
    ey_cropped = ey[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.float32, copy=True)

    # Rasterize conductor masks at canvas resolution (do this once to avoid redundant rasterization)
    conductor_masks_canvas = None
    interior_masks_canvas = None
    if project.conductors:
        conductor_masks_canvas, interior_masks_canvas = rasterize_conductor_masks(
            project.conductors,
            lic_cropped.shape,
            margin_physical,
            scale,
            crop_x0,
            crop_y0,
        )

    t_end = time.time()
    print(f"Total render time: {t_end - t_start:.2f}s")

    return RenderResult(
        array=lic_cropped.astype(np.float32, copy=True),
        compute_resolution=(compute_h, compute_w),
        canvas_scaled_shape=lic_cropped.shape,
        margin=margin_physical,
        offset_x=crop_x0,
        offset_y=crop_y0,
        poisson_scale=poisson_scale,
        ex=ex_cropped,
        ey=ey_cropped,
        conductor_masks_canvas=conductor_masks_canvas,
        interior_masks_canvas=interior_masks_canvas,
        solution=solution,  # Store the PDE solution dict
    )
