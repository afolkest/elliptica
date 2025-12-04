"""Toolkit-neutral application state and helpers for Elliptica."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from elliptica import defaults
from elliptica.gpu import GPUContext
from elliptica.pipeline import RenderResult
from elliptica.postprocess.color import ColorParams
from elliptica.types import Project, Conductor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from elliptica.colorspace import ColorConfig


@dataclass
class RegionStyle:
    """Color and effect settings for a spatial region."""
    enabled: bool = False
    use_palette: bool = True  # True = palette colorization, False = solid color
    palette: str = defaults.DEFAULT_COLOR_PALETTE
    solid_color: tuple[float, float, float] = (0.5, 0.5, 0.5)  # RGB [0,1]

    # Postprocessing overrides (None = inherit from global DisplaySettings)
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    gamma: Optional[float] = None

    # Lightness expression (None = inherit from global, or no expr if global is also None)
    lightness_expr: Optional[str] = None

    # Smear effect (texture blur within region)
    smear_enabled: bool = False
    smear_sigma: float = defaults.DEFAULT_SMEAR_SIGMA


def resolve_region_postprocess_params(
    region: RegionStyle,
    global_settings: DisplaySettings,
) -> tuple[float, float, float]:
    """Resolve per-region postprocessing params, falling back to global settings.

    Args:
        region: Region-specific settings (may have None for params)
        global_settings: Global display settings

    Returns:
        (brightness, contrast, gamma) with global fallback applied
    """
    brightness = region.brightness if region.brightness is not None else global_settings.brightness
    contrast = region.contrast if region.contrast is not None else global_settings.contrast
    gamma = region.gamma if region.gamma is not None else global_settings.gamma
    return brightness, contrast, gamma


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
    use_mask: bool = defaults.DEFAULT_USE_MASK
    edge_gain_strength: float = defaults.DEFAULT_EDGE_GAIN_STRENGTH
    edge_gain_power: float = defaults.DEFAULT_EDGE_GAIN_POWER
    solve_scale: float = defaults.DEFAULT_SOLVE_SCALE


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
    lightness_expr: str | None = None
    saturation: float = 1.0  # Post-colorization chroma multiplier (1.0 = no change)

    def to_color_params(self):
        """Convert to pure ColorParams for backend functions."""
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

    All data kept at full render resolution - DearPyGUI handles scaling for display.
    GPU tensors used when available for fast postprocessing.
    """

    def __init__(
        self,
        result: RenderResult,
        multiplier: float,
        supersample: float,
        base_rgb: Optional[np.ndarray] = None,
        conductor_masks: Optional[list[np.ndarray]] = None,
        interior_masks: Optional[list[np.ndarray]] = None,
        project_fingerprint: str = "",
        result_gpu: Optional[torch.Tensor] = None,
        ex_gpu: Optional[torch.Tensor] = None,
        ey_gpu: Optional[torch.Tensor] = None,
        lic_percentiles: Optional[tuple[float, float]] = None,
        conductor_masks_gpu: Optional[list[Optional[torch.Tensor]]] = None,
        interior_masks_gpu: Optional[list[Optional[torch.Tensor]]] = None,
    ):
        self.result = result  # Full resolution RenderResult
        self.multiplier = multiplier
        self.supersample = supersample
        self.base_rgb = base_rgb  # Cached RGB at full resolution

        # Masks at full render resolution (for region overlays and smear)
        self.conductor_masks = conductor_masks
        self.interior_masks = interior_masks

        self.project_fingerprint = project_fingerprint

        # GPU tensors at full resolution for fast postprocessing
        self.result_gpu = result_gpu  # Full-res LIC on GPU
        self.ex_gpu = ex_gpu  # Electric field X component on GPU
        self.ey_gpu = ey_gpu  # Electric field Y component on GPU
        self.solution_gpu: Optional[dict[str, torch.Tensor]] = None  # PDE solution fields on GPU (phi, etc.)
        self.solution_gpu_lic_shape: Optional[tuple[int, int]] = None  # Shape solution_gpu was resized to
        self.solution_gpu_resized: Optional[dict[str, torch.Tensor]] = None  # Solution fields resized to LIC resolution

        self.lic_percentiles = lic_percentiles  # Precomputed (vmin, vmax) for smear normalization
        self.lic_percentiles_clip_percent: float | None = 0.5 if lic_percentiles is not None else None  # Clip% used for cached percentiles

        # GPU mask tensors (cached to avoid repeated CPU→GPU transfers)
        self.conductor_masks_gpu = conductor_masks_gpu
        self.interior_masks_gpu = interior_masks_gpu


@dataclass
class AppState:
    """Central application state shared across UIs."""

    project: Project = field(default_factory=Project)
    render_settings: RenderSettings = field(default_factory=RenderSettings)
    display_settings: DisplaySettings = field(default_factory=DisplaySettings)

    # Interaction state
    selected_indices: set[int] = field(default_factory=set)  # Selected conductor indices
    selected_region_type: str = "surface"  # "surface" or "interior"
    view_mode: str = "edit"  # "edit" or "render"

    # Dirty flags – downstream systems decide how to respond.
    field_dirty: bool = True
    render_dirty: bool = True

    # Cached outputs
    render_cache: Optional[RenderCache] = None

    # Per-conductor color settings (keyed by conductor.id)
    conductor_color_settings: dict[int, ConductorColorSettings] = field(default_factory=dict)

    # Expression-based color configuration (None = use palette mode)
    # When set, overrides palette/brightness/contrast/gamma with OKLCH expressions
    color_config: Optional["ColorConfig"] = None

    def set_selected(self, idx: int) -> None:
        """Replace selection with single conductor at index, or clear if invalid."""
        self.selected_indices.clear()
        if 0 <= idx < len(self.project.conductors):
            self.selected_indices.add(idx)

    def toggle_selected(self, idx: int) -> None:
        """Toggle conductor in/out of selection (for shift-click)."""
        if not (0 <= idx < len(self.project.conductors)):
            return
        if idx in self.selected_indices:
            self.selected_indices.discard(idx)
        else:
            self.selected_indices.add(idx)

    def clear_selection(self) -> None:
        """Clear all selection."""
        self.selected_indices.clear()

    def get_selected(self) -> Optional[Conductor]:
        """Return selected conductor only if exactly one is selected."""
        if len(self.selected_indices) == 1:
            idx = next(iter(self.selected_indices))
            if 0 <= idx < len(self.project.conductors):
                return self.project.conductors[idx]
        return None

    def get_single_selected_idx(self) -> int:
        """Return selected index if exactly one selected, else -1."""
        if len(self.selected_indices) == 1:
            idx = next(iter(self.selected_indices))
            if 0 <= idx < len(self.project.conductors):
                return idx
        return -1

    def clear_render_cache(self) -> None:
        """Invalidate cached render output and free GPU memory."""
        if self.render_cache:
            # Clear GPU tensors to free VRAM
            self.render_cache.result_gpu = None
            self.render_cache.ex_gpu = None
            self.render_cache.ey_gpu = None
            self.render_cache.conductor_masks_gpu = None
            self.render_cache.interior_masks_gpu = None
            # Release GPU memory back to system
            GPUContext.empty_cache()
        self.render_cache = None
        self.render_dirty = True

    def invalidate_base_rgb(self) -> None:
        """Clear cached base RGB, forcing recompute on next display."""
        if self.render_cache:
            self.render_cache.base_rgb = None
            # Note: We keep display_array_gpu and CPU cache since they're still valid for recomputing base_rgb
            # Only the derived RGB needs recomputation when colorization params change
