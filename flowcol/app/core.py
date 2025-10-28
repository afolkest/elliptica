"""Toolkit-neutral application state and helpers for FlowCol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from flowcol import defaults
from flowcol.pipeline import RenderResult
from flowcol.types import Project, Conductor


@dataclass
class RegionStyle:
    """Color settings for a spatial region."""
    enabled: bool = False
    use_palette: bool = True  # True = palette colorization, False = solid color
    palette: str = defaults.DEFAULT_COLOR_PALETTE
    solid_color: tuple[float, float, float] = (0.5, 0.5, 0.5)  # RGB [0,1]


@dataclass
class ConductorColorSettings:
    """Color settings for a conductor's two regions."""
    surface: RegionStyle = field(default_factory=RegionStyle)
    interior: RegionStyle = field(default_factory=RegionStyle)


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
    brightness: float = defaults.DEFAULT_BRIGHTNESS
    contrast: float = defaults.DEFAULT_CONTRAST
    gamma: float = defaults.DEFAULT_GAMMA
    color_enabled: bool = defaults.DEFAULT_COLOR_ENABLED
    palette: str = defaults.DEFAULT_COLOR_PALETTE
    edge_blur_sigma: float = defaults.DEFAULT_EDGE_BLUR_SIGMA
    edge_blur_falloff: float = defaults.DEFAULT_EDGE_BLUR_FALLOFF
    edge_blur_strength: float = defaults.DEFAULT_EDGE_BLUR_STRENGTH
    edge_blur_power: float = defaults.DEFAULT_EDGE_BLUR_POWER

    def to_color_params(self):
        """Convert to pure ColorParams for backend functions."""
        from flowcol.postprocess.color import ColorParams
        return ColorParams(
            clip_percent=self.clip_percent,
            brightness=self.brightness,
            contrast=self.contrast,
            gamma=self.gamma,
            color_enabled=self.color_enabled,
            palette=self.palette,
        )


class RenderCache:
    """Latest render output retained for display/export.

    Single source of truth pattern:
    - GPU tensor (display_array_gpu) is primary when available
    - CPU array (_display_array_cpu) is fallback when GPU unavailable
    - display_array property provides cached lazy download from GPU
    """

    def __init__(
        self,
        result: RenderResult,
        multiplier: float,
        supersample: float,
        display_array: Optional[np.ndarray] = None,
        base_rgb: Optional[np.ndarray] = None,
        conductor_masks: Optional[list[np.ndarray]] = None,
        interior_masks: Optional[list[np.ndarray]] = None,
        full_res_conductor_masks: Optional[list[np.ndarray]] = None,
        full_res_interior_masks: Optional[list[np.ndarray]] = None,
        project_fingerprint: str = "",
        edge_blurred_array: Optional[np.ndarray] = None,
        result_gpu: Optional[torch.Tensor] = None,
        display_array_gpu: Optional[torch.Tensor] = None,
        ex_gpu: Optional[torch.Tensor] = None,
        ey_gpu: Optional[torch.Tensor] = None,
    ):
        self.result = result
        self.multiplier = multiplier
        self.supersample = supersample
        self.base_rgb = base_rgb
        self.conductor_masks = conductor_masks
        self.interior_masks = interior_masks
        self.full_res_conductor_masks = full_res_conductor_masks
        self.full_res_interior_masks = full_res_interior_masks
        self.project_fingerprint = project_fingerprint
        self.edge_blurred_array = edge_blurred_array
        self.result_gpu = result_gpu
        self.ex_gpu = ex_gpu
        self.ey_gpu = ey_gpu

        # Single source of truth: GPU when available, CPU when not
        self.display_array_gpu = display_array_gpu
        self._display_array_cpu: Optional[np.ndarray] = None if display_array_gpu is not None else display_array

        # Cached CPU copy for lazy download from GPU
        self._display_array_cpu_cache: Optional[np.ndarray] = None
        self._cpu_cache_valid: bool = False

    @property
    def display_array(self) -> Optional[np.ndarray]:
        """Lazy CPU access - downloads from GPU if needed, cached."""
        if self.display_array_gpu is not None:
            # GPU is source of truth - use cached download
            if not self._cpu_cache_valid:
                from flowcol.gpu import GPUContext
                self._display_array_cpu_cache = GPUContext.to_cpu(self.display_array_gpu)
                self._cpu_cache_valid = True
            return self._display_array_cpu_cache
        # CPU is source of truth
        return self._display_array_cpu

    @display_array.setter
    def display_array(self, value: Optional[np.ndarray]) -> None:
        """Set CPU array directly (for CPU-only code paths)."""
        if self.display_array_gpu is not None:
            # If GPU exists, don't allow direct CPU writes
            raise ValueError("Cannot set display_array when GPU tensor is primary. Clear GPU first.")
        self._display_array_cpu = value
        self._cpu_cache_valid = False

    def set_display_array_cpu(self, arr: np.ndarray) -> None:
        """Set CPU as primary source (clears GPU tensor)."""
        self.display_array_gpu = None
        self._display_array_cpu = arr
        self._display_array_cpu_cache = None
        self._cpu_cache_valid = False

    def set_display_array_gpu(self, tensor: torch.Tensor) -> None:
        """Set GPU as primary source (invalidates CPU cache)."""
        self.display_array_gpu = tensor
        self._display_array_cpu = None
        self._display_array_cpu_cache = None
        self._cpu_cache_valid = False

    def invalidate_cpu_cache(self) -> None:
        """Mark CPU cache invalid. Call this when GPU tensor is modified."""
        self._cpu_cache_valid = False


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

    # Per-conductor color settings (keyed by conductor.id)
    conductor_color_settings: dict[int, ConductorColorSettings] = field(default_factory=dict)

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
        """Invalidate cached render output and free GPU memory."""
        if self.render_cache:
            # Clear GPU tensors to free VRAM
            self.render_cache.result_gpu = None
            self.render_cache.display_array_gpu = None
            self.render_cache.ex_gpu = None
            self.render_cache.ey_gpu = None
            # Clear CPU cached postprocessing results
            self.render_cache.edge_blurred_array = None
            # Release GPU memory back to system
            from flowcol.gpu import GPUContext
            GPUContext.empty_cache()
        self.render_cache = None
        self.render_dirty = True

    def invalidate_base_rgb(self) -> None:
        """Clear cached base RGB, forcing recompute on next display."""
        if self.render_cache:
            self.render_cache.base_rgb = None
            # Note: We keep display_array_gpu and CPU cache since they're still valid for recomputing base_rgb
            # Only the derived RGB needs recomputation when colorization params change
