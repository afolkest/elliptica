"""High-level mutations on AppState reused across UIs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from flowcol.pipeline import perform_render
from flowcol.types import Conductor
from flowcol.app.core import AppState, RenderCache

MAX_CONDUCTOR_DIM = 32768


def add_conductor(state: AppState, conductor: Conductor) -> None:
    """Insert a new conductor and mark field/render dirty."""
    state.project.conductors.append(conductor)
    state.selected_idx = len(state.project.conductors) - 1
    state.field_dirty = True
    state.render_dirty = True


def remove_conductor(state: AppState, idx: int) -> None:
    """Remove conductor by index."""
    if 0 <= idx < len(state.project.conductors):
        del state.project.conductors[idx]
        if state.selected_idx >= len(state.project.conductors):
            state.selected_idx = len(state.project.conductors) - 1
        state.field_dirty = True
        state.render_dirty = True


def move_conductor(state: AppState, idx: int, dx: float, dy: float) -> None:
    """Translate conductor by delta."""
    if 0 <= idx < len(state.project.conductors):
        cond = state.project.conductors[idx]
        cond.position = (cond.position[0] + dx, cond.position[1] + dy)
        state.field_dirty = True
        state.render_dirty = True


def set_conductor_voltage(state: AppState, idx: int, voltage: float) -> None:
    """Assign voltage to conductor."""
    if 0 <= idx < len(state.project.conductors):
        state.project.conductors[idx].voltage = voltage
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


def scale_conductor(state: AppState, idx: int, scale_delta: float) -> bool:
    """Scale a conductor mask by a delta factor around its center.

    Args:
        state: Application state
        idx: Conductor index
        scale_delta: Multiplicative factor (e.g., 1.1 for 10% larger, 0.9 for 10% smaller)
    """
    if not (0 <= idx < len(state.project.conductors)):
        return False
    scale_delta = float(scale_delta)
    if not np.isfinite(scale_delta) or scale_delta <= 0.0:
        return False

    conductor = state.project.conductors[idx]

    # Store original on first scale
    if conductor.original_mask is None:
        conductor.original_mask = conductor.mask.copy()
        if conductor.interior_mask is not None:
            conductor.original_interior_mask = conductor.interior_mask.copy()

    # Calculate new cumulative scale factor
    new_scale_factor = conductor.scale_factor * scale_delta

    # Clamp to reasonable range
    if new_scale_factor < 0.01 or new_scale_factor > 100.0:
        return False

    # Always scale from original to preserve quality
    source_mask = conductor.original_mask
    if source_mask.size == 0:
        return False

    old_h, old_w = conductor.mask.shape
    orig_h, orig_w = source_mask.shape
    new_h = max(1, int(round(orig_h * new_scale_factor)))
    new_w = max(1, int(round(orig_w * new_scale_factor)))

    if new_h == old_h and new_w == old_w:
        return False

    if new_h > MAX_CONDUCTOR_DIM or new_w > MAX_CONDUCTOR_DIM:
        return False

    scale_y = new_h / orig_h
    scale_x = new_w / orig_w

    scaled_mask = zoom(source_mask, (scale_y, scale_x), order=1)
    scaled_mask = np.clip(scaled_mask, 0.0, 1.0).astype(np.float32)

    if conductor.original_interior_mask is not None:
        scaled_interior = zoom(conductor.original_interior_mask, (scale_y, scale_x), order=1)
        conductor.interior_mask = np.clip(scaled_interior, 0.0, 1.0).astype(np.float32)

    center_x = conductor.position[0] + old_w / 2.0
    center_y = conductor.position[1] + old_h / 2.0

    conductor.mask = scaled_mask
    conductor.scale_factor = new_scale_factor
    conductor.position = (
        center_x - new_w / 2.0,
        center_y - new_h / 2.0,
    )

    state.field_dirty = True
    state.render_dirty = True
    return True


def ensure_render(state: AppState) -> bool:
    """Run the render pipeline if required.

    Returns True on success, False if render failed (e.g., resolution too large).
    """
    if not state.render_dirty and state.render_cache:
        return True

    settings = state.render_settings
    result = perform_render(
        state.project,
        settings.multiplier,
        settings.supersample,
        settings.num_passes,
        settings.margin,
        settings.noise_seed,
        settings.noise_sigma,
        state.project.streamlength_factor,
    )
    if result is None:
        return False

    state.render_cache = RenderCache(
        result=result,
        multiplier=settings.multiplier,
        supersample=settings.supersample,
        display_array=result.array.copy(),
    )
    state.field_dirty = False
    state.render_dirty = False
    state.view_mode = "render"
    return True
