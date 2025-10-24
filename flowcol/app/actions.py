"""High-level mutations on AppState reused across UIs."""

from __future__ import annotations

import numpy as np

from flowcol.pipeline import perform_render
from flowcol.types import Conductor
from flowcol.app.core import AppState, RenderCache


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
