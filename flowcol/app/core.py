"""Toolkit-neutral application state and helpers for FlowCol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from flowcol import defaults
from flowcol.pipeline import RenderResult
from flowcol.types import Project, Conductor


@dataclass
class RenderSettings:
    """User-configurable render parameters."""

    multiplier: float = defaults.RENDER_RESOLUTION_CHOICES[0]
    supersample: float = defaults.SUPERSAMPLE_CHOICES[0]
    num_passes: int = defaults.DEFAULT_RENDER_PASSES
    margin: float = defaults.DEFAULT_PADDING_MARGIN
    noise_seed: int = defaults.DEFAULT_NOISE_SEED
    noise_sigma: float = defaults.DEFAULT_NOISE_SIGMA


@dataclass
class DisplaySettings:
    """Display/postprocessing parameters applied to cached render."""

    downsample_sigma: float = defaults.DEFAULT_DOWNSAMPLE_SIGMA
    clip_percent: float = defaults.DEFAULT_CLIP_PERCENT
    contrast: float = defaults.DEFAULT_CONTRAST
    gamma: float = defaults.DEFAULT_GAMMA
    color_enabled: bool = defaults.DEFAULT_COLOR_ENABLED
    palette: str = defaults.DEFAULT_COLOR_PALETTE


@dataclass
class RenderCache:
    """Latest render output retained for display/export."""

    result: RenderResult
    multiplier: float
    supersample: float
    display_array: Optional[np.ndarray] = None
    # Cached intermediate for fast colorization updates
    base_rgb: Optional[np.ndarray] = None
    # Segmentation masks at display resolution
    conductor_masks: Optional[list[np.ndarray]] = None
    interior_masks: Optional[list[np.ndarray]] = None


@dataclass
class AppState:
    """Central application state shared across UIs."""

    project: Project = field(default_factory=Project)
    render_settings: RenderSettings = field(default_factory=RenderSettings)
    display_settings: DisplaySettings = field(default_factory=DisplaySettings)

    # Interaction state
    selected_idx: int = -1
    view_mode: str = "edit"  # "edit" or "render"

    # Dirty flags – downstream systems decide how to respond.
    field_dirty: bool = True
    render_dirty: bool = True

    # Cached outputs
    render_cache: Optional[RenderCache] = None

    # Per-conductor style settings (keyed by conductor.id)
    conductor_styles: dict[int, dict] = field(default_factory=dict)

    def set_selected(self, idx: int) -> None:
        """Select conductor at index or clear selection."""
        if 0 <= idx < len(self.project.conductors):
            self.selected_idx = idx
        else:
            self.selected_idx = -1

    def get_selected(self) -> Optional[Conductor]:
        """Return currently selected conductor, if any."""
        if 0 <= self.selected_idx < len(self.project.conductors):
            return self.project.conductors[self.selected_idx]
        return None

    def clear_render_cache(self) -> None:
        """Invalidate cached render output."""
        self.render_cache = None
        self.render_dirty = True

    def invalidate_base_rgb(self) -> None:
        """Clear cached base RGB, forcing recompute on next display."""
        if self.render_cache:
            self.render_cache.base_rgb = None
