"""High-level mutations on AppState reused across UIs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from elliptica.types import BoundaryObject
from elliptica.app.core import AppState, BoundaryColorSettings
from elliptica.postprocess.masks import derive_interior
from elliptica import defaults

MAX_BOUNDARY_DIM = 32768


def add_boundary(state: AppState, boundary: BoundaryObject) -> None:
    """Insert a new boundary object and mark field/render dirty."""

    # Assign unique ID
    boundary.id = state.project.next_boundary_id
    state.project.next_boundary_id += 1

    # Auto-detect interior if not provided
    if boundary.interior_mask is None:
        interior = derive_interior(boundary.mask, thickness=0.1)
        if interior is not None:
            boundary.interior_mask = interior

    state.project.boundary_objects.append(boundary)
    state.set_selected(len(state.project.boundary_objects) - 1)

    # Initialize color settings
    state.boundary_color_settings[boundary.id] = BoundaryColorSettings()

    state.field_dirty = True
    state.render_dirty = True


def remove_boundary(state: AppState, idx: int) -> None:
    """Remove boundary object by index."""
    if 0 <= idx < len(state.project.boundary_objects):
        boundary = state.project.boundary_objects[idx]

        # Clean up color settings
        if boundary.id is not None:
            state.boundary_color_settings.pop(boundary.id, None)

        del state.project.boundary_objects[idx]

        # Update selection: remove deleted index, shift higher indices down
        new_selection = set()
        for sel_idx in state.selected_indices:
            if sel_idx < idx:
                new_selection.add(sel_idx)
            elif sel_idx > idx:
                new_selection.add(sel_idx - 1)
            # sel_idx == idx is dropped (deleted)
        state.selected_indices = new_selection

        state.field_dirty = True
        state.render_dirty = True


def move_boundary(state: AppState, idx: int, dx: float, dy: float) -> None:
    """Translate boundary object by delta."""
    if 0 <= idx < len(state.project.boundary_objects):
        boundary = state.project.boundary_objects[idx]
        boundary.position = (boundary.position[0] + dx, boundary.position[1] + dy)
        state.field_dirty = True
        state.render_dirty = True


def set_boundary_voltage(state: AppState, idx: int, voltage: float) -> None:
    """Assign voltage to boundary object."""
    if 0 <= idx < len(state.project.boundary_objects):
        state.project.boundary_objects[idx].voltage = voltage
        state.field_dirty = True
        state.render_dirty = True


def set_canvas_resolution(state: AppState, width: int, height: int) -> None:
    """Resize project canvas."""
    width = max(int(width), 1)
    height = max(int(height), 1)
    if state.project.canvas_resolution != (width, height):
        state.project.canvas_resolution = (width, height)
        state.field_dirty = True
        state.render_dirty = True
        state.clear_render_cache()


def set_streamlength_factor(state: AppState, factor: float) -> None:
    """Update project streamlength factor."""
    factor = max(float(factor), 1e-6)
    if not np.isclose(state.project.streamlength_factor, factor):
        state.project.streamlength_factor = factor
        state.render_dirty = True


def set_render_multiplier(state: AppState, multiplier: float) -> None:
    multiplier = max(float(multiplier), 1e-3)
    if not np.isclose(state.render_settings.multiplier, multiplier):
        state.render_settings.multiplier = multiplier
        state.render_dirty = True


def set_supersample(state: AppState, supersample: float) -> None:
    supersample = max(float(supersample), 1.0)
    if not np.isclose(state.render_settings.supersample, supersample):
        state.render_settings.supersample = supersample
        state.render_dirty = True


def set_num_passes(state: AppState, passes: int) -> None:
    passes = max(int(passes), 1)
    if state.render_settings.num_passes != passes:
        state.render_settings.num_passes = passes
        state.render_dirty = True


def set_margin(state: AppState, margin: float) -> None:
    margin = max(float(margin), 0.0)
    if not np.isclose(state.render_settings.margin, margin):
        state.render_settings.margin = margin
        state.field_dirty = True
        state.render_dirty = True


def set_noise_seed(state: AppState, seed: int) -> None:
    if state.render_settings.noise_seed != seed:
        state.render_settings.noise_seed = int(seed)
        state.render_dirty = True


def set_noise_sigma(state: AppState, sigma: float) -> None:
    sigma = max(float(sigma), 0.0)
    if not np.isclose(state.render_settings.noise_sigma, sigma):
        state.render_settings.noise_sigma = sigma
        state.render_dirty = True


def set_solve_scale(state: AppState, scale: float) -> None:
    """Update PDE solve resolution scale (0.1–1.0)."""
    scale = float(scale)
    if not np.isfinite(scale):
        return
    scale = max(defaults.MIN_SOLVE_SCALE, min(defaults.MAX_SOLVE_SCALE, scale))
    if not np.isclose(state.render_settings.solve_scale, scale):
        state.render_settings.solve_scale = scale
        state.field_dirty = True
        state.render_dirty = True


def scale_boundary(state: AppState, idx: int, scale_delta: float) -> bool:
    """Scale a boundary mask by a delta factor around its center.

    Args:
        state: Application state
        idx: Boundary object index
        scale_delta: Multiplicative factor (e.g., 1.1 for 10% larger, 0.9 for 10% smaller)
    """
    if not (0 <= idx < len(state.project.boundary_objects)):
        return False
    scale_delta = float(scale_delta)
    if not np.isfinite(scale_delta) or scale_delta <= 0.0:
        return False

    boundary = state.project.boundary_objects[idx]

    # Store original on first scale
    if boundary.original_mask is None:
        boundary.original_mask = boundary.mask.copy()
        if boundary.interior_mask is not None:
            boundary.original_interior_mask = boundary.interior_mask.copy()

    # Calculate new cumulative scale factor
    new_scale_factor = boundary.scale_factor * scale_delta

    # Clamp to reasonable range
    if new_scale_factor < 0.01 or new_scale_factor > 100.0:
        return False

    # Always scale from original to preserve quality
    source_mask = boundary.original_mask
    if source_mask.size == 0:
        return False

    old_h, old_w = boundary.mask.shape
    orig_h, orig_w = source_mask.shape
    new_h = max(1, int(round(orig_h * new_scale_factor)))
    new_w = max(1, int(round(orig_w * new_scale_factor)))

    if new_h == old_h and new_w == old_w:
        return False

    if new_h > MAX_BOUNDARY_DIM or new_w > MAX_BOUNDARY_DIM:
        return False

    scale_y = new_h / orig_h
    scale_x = new_w / orig_w

    scaled_mask = zoom(source_mask, (scale_y, scale_x), order=1)
    scaled_mask = np.clip(scaled_mask, 0.0, 1.0).astype(np.float32)

    if boundary.original_interior_mask is not None:
        scaled_interior = zoom(boundary.original_interior_mask, (scale_y, scale_x), order=1)
        boundary.interior_mask = np.clip(scaled_interior, 0.0, 1.0).astype(np.float32)

    center_x = boundary.position[0] + old_w / 2.0
    center_y = boundary.position[1] + old_h / 2.0

    boundary.mask = scaled_mask
    boundary.scale_factor = new_scale_factor
    boundary.position = (
        center_x - new_w / 2.0,
        center_y - new_h / 2.0,
    )

    state.field_dirty = True
    state.render_dirty = True
    return True


def set_color_enabled(state: AppState, enabled: bool) -> None:
    """Toggle colorization on/off."""
    if state.display_settings.color_enabled != enabled:
        state.display_settings.color_enabled = enabled
        state.invalidate_base_rgb()


def set_palette(state: AppState, palette: str) -> None:
    """Change color palette."""
    if state.display_settings.palette != palette:
        state.display_settings.palette = palette
        state.invalidate_base_rgb()


def set_region_style_enabled(state: AppState, boundary_id: int, region: str, enabled: bool) -> bool:
    """Toggle custom colorization for a region.

    Args:
        boundary_id: Boundary object ID
        region: "surface" or "interior"
        enabled: Enable/disable custom colorization

    Returns:
        True if successful, False if boundary_color_settings missing for this boundary
    """
    settings = state.boundary_color_settings.get(boundary_id)
    if settings is None:
        print(f"⚠️  set_region_style_enabled: no settings for boundary {boundary_id}")
        return False

    if region == "surface":
        settings.surface.enabled = enabled
    elif region == "interior":
        settings.interior.enabled = enabled
    return True


def set_region_palette(state: AppState, boundary_id: int, region: str, palette: str) -> bool:
    """Set palette for region (implies use_palette=True and enabled=True).

    Args:
        boundary_id: Boundary object ID
        region: "surface" or "interior"
        palette: Palette name

    Returns:
        True if successful, False if boundary_color_settings missing for this boundary
    """
    settings = state.boundary_color_settings.get(boundary_id)
    if settings is None:
        print(f"⚠️  set_region_palette: no settings for boundary {boundary_id}")
        return False

    if region == "surface":
        settings.surface.enabled = True  # Auto-enable when palette is selected
        settings.surface.use_palette = True
        settings.surface.palette = palette
    elif region == "interior":
        settings.interior.enabled = True  # Auto-enable when palette is selected
        settings.interior.use_palette = True
        settings.interior.palette = palette
    return True


def set_region_solid_color(state: AppState, boundary_id: int, region: str, rgb: tuple[float, float, float]) -> bool:
    """Set solid color for region (implies use_palette=False and enabled=True).

    Args:
        boundary_id: Boundary object ID
        region: "surface" or "interior"
        rgb: RGB tuple in [0, 1]

    Returns:
        True if successful, False if boundary_color_settings missing for this boundary
    """
    settings = state.boundary_color_settings.get(boundary_id)
    if settings is None:
        print(f"⚠️  set_region_solid_color: no settings for boundary {boundary_id}")
        return False

    if region == "surface":
        settings.surface.enabled = True  # Auto-enable when color is selected
        settings.surface.use_palette = False
        settings.surface.solid_color = rgb
    elif region == "interior":
        settings.interior.enabled = True  # Auto-enable when color is selected
        settings.interior.use_palette = False
        settings.interior.solid_color = rgb
    return True


def set_region_brightness(state: AppState, boundary_id: int, region: str, brightness: float | None) -> bool:
    """Set per-region brightness override (None = inherit from global).

    Args:
        boundary_id: Boundary object ID
        region: "surface" or "interior"
        brightness: Brightness value, or None to inherit from global

    Returns:
        True if successful, False if boundary_color_settings missing for this boundary
    """
    settings = state.boundary_color_settings.get(boundary_id)
    if settings is None:
        print(f"⚠️  set_region_brightness: no settings for boundary {boundary_id}")
        return False

    if region == "surface":
        settings.surface.brightness = brightness
    elif region == "interior":
        settings.interior.brightness = brightness
    return True


def set_region_contrast(state: AppState, boundary_id: int, region: str, contrast: float | None) -> bool:
    """Set per-region contrast override (None = inherit from global).

    Args:
        boundary_id: Boundary object ID
        region: "surface" or "interior"
        contrast: Contrast value, or None to inherit from global

    Returns:
        True if successful, False if boundary_color_settings missing for this boundary
    """
    settings = state.boundary_color_settings.get(boundary_id)
    if settings is None:
        print(f"⚠️  set_region_contrast: no settings for boundary {boundary_id}")
        return False

    if region == "surface":
        settings.surface.contrast = contrast
    elif region == "interior":
        settings.interior.contrast = contrast
    return True


def set_region_gamma(state: AppState, boundary_id: int, region: str, gamma: float | None) -> bool:
    """Set per-region gamma override (None = inherit from global).

    Args:
        boundary_id: Boundary object ID
        region: "surface" or "interior"
        gamma: Gamma value, or None to inherit from global

    Returns:
        True if successful, False if boundary_color_settings missing for this boundary
    """
    settings = state.boundary_color_settings.get(boundary_id)
    if settings is None:
        print(f"⚠️  set_region_gamma: no settings for boundary {boundary_id}")
        return False

    if region == "surface":
        settings.surface.gamma = gamma
    elif region == "interior":
        settings.interior.gamma = gamma
    return True
