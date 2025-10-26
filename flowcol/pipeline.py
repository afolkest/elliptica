"""Pure render pipeline orchestration - no UI dependencies."""

import numpy as np
from dataclasses import dataclass
import time
from flowcol.types import Project
from flowcol.field import compute_field
from flowcol.render import (
    compute_lic,
    downsample_lic,
    apply_gaussian_highpass,
    apply_highpass_clahe,
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


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""

    detail_enabled: bool = False
    detail_sigma_factor: float = 0.02
    highpass_enabled: bool = False
    highpass_sigma_factor: float = 0.01
    highpass_clip_limit: float = 0.03
    highpass_kernel_rows: int = 64
    highpass_kernel_cols: int = 64
    highpass_num_bins: int = 256
    highpass_strength: float = 1.0


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

    if config.highpass_enabled:
        sigma_px = max(config.highpass_sigma_factor * reference_size, 0.0)
        working = apply_highpass_clahe(
            working,
            sigma_px,
            config.highpass_clip_limit,
            config.highpass_kernel_rows,
            config.highpass_kernel_cols,
            config.highpass_num_bins,
            config.highpass_strength,
        )

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

    print(f"Starting Poisson solve ({compute_w}×{compute_h})...")
    t_poisson_start = time.time()
    ex, ey = compute_field(
        project,
        multiplier,
        supersample_factor,
        margin_tuple,
        boundary_top=project.boundary_top,
        boundary_bottom=project.boundary_bottom,
        boundary_left=project.boundary_left,
        boundary_right=project.boundary_right,
    )
    t_poisson_end = time.time()
    print(f"  Poisson solve completed in {t_poisson_end - t_poisson_start:.2f}s")

    num_passes = max(1, num_passes)
    min_compute = min(compute_w, compute_h)
    streamlength_pixels = max(int(round(streamlength_factor * min_compute)), 1)

    print(f"Starting LIC ({num_passes} passes, streamlength={streamlength_pixels})...")
    t_lic_start = time.time()
    lic_array = compute_lic(
        ex,
        ey,
        streamlength_pixels,
        num_passes=num_passes,
        seed=noise_seed,
        noise_sigma=noise_sigma,
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

    t_end = time.time()
    print(f"Total render time: {t_end - t_start:.2f}s")

    return RenderResult(
        array=lic_cropped.astype(np.float32, copy=True),
        compute_resolution=(compute_h, compute_w),
        canvas_scaled_shape=lic_cropped.shape,
        margin=margin_physical,
        offset_x=crop_x0,
        offset_y=crop_y0,
    )
