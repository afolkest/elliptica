"""Core data types for flowcol."""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from flowcol import defaults


@dataclass
class Conductor:
    mask: np.ndarray
    voltage: float
    position: tuple[float, float] = (0.0, 0.0)
    interior_mask: Optional[np.ndarray] = None


@dataclass
class RenderInfo:
    multiplier: int
    filepath: str
    timestamp: str


@dataclass
class Project:
    conductors: list[Conductor] = field(default_factory=list)
    canvas_resolution: tuple[int, int] = defaults.DEFAULT_CANVAS_RESOLUTION
    streamlength_factor: float = defaults.DEFAULT_STREAMLENGTH_FACTOR
    renders: list[RenderInfo] = field(default_factory=list)


@dataclass
class RenderMenuState:
    is_open: bool = False
    selected_multiplier: float = defaults.RENDER_RESOLUTION_CHOICES[0]
    num_passes: int = defaults.DEFAULT_RENDER_PASSES
    input_focused: bool = False
    streamlength_text: str = f"{defaults.DEFAULT_STREAMLENGTH_FACTOR:.4f}"
    streamlength_input_focused: bool = False
    streamlength_pending_clear: bool = False
    pending_streamlength_factor: float = defaults.DEFAULT_STREAMLENGTH_FACTOR
    seed_input_focused: bool = False
    seed_pending_clear: bool = False
    margin_text: str = f"{defaults.DEFAULT_PADDING_MARGIN:.2f}"
    margin_input_focused: bool = False
    margin_pending_clear: bool = False
    pending_margin_factor: float = defaults.DEFAULT_PADDING_MARGIN
    noise_sigma_text: str = f"{defaults.DEFAULT_NOISE_SIGMA:.1f}"
    noise_sigma_input_focused: bool = False
    noise_sigma_pending_clear: bool = False
    pending_noise_sigma: float = defaults.DEFAULT_NOISE_SIGMA


@dataclass
class HighPassMenuState:
    is_open: bool = False
    sigma_factor: float = defaults.DEFAULT_HIGHPASS_SIGMA_FACTOR
    clip_limit: float = defaults.DEFAULT_CLAHE_CLIP_LIMIT
    kernel_rows: int = defaults.DEFAULT_CLAHE_KERNEL_ROWS
    kernel_cols: int = defaults.DEFAULT_CLAHE_KERNEL_COLS
    num_bins: int = defaults.DEFAULT_CLAHE_BINS
    sigma_factor_text: str = f"{defaults.DEFAULT_HIGHPASS_SIGMA_FACTOR:.4f}"
    clip_text: str = f"{defaults.DEFAULT_CLAHE_CLIP_LIMIT:.2f}"
    kernel_rows_text: str = str(defaults.DEFAULT_CLAHE_KERNEL_ROWS)
    kernel_cols_text: str = str(defaults.DEFAULT_CLAHE_KERNEL_COLS)
    num_bins_text: str = str(defaults.DEFAULT_CLAHE_BINS)
    strength: float = defaults.DEFAULT_CLAHE_STRENGTH
    strength_dragging: bool = False
    focused_field: int = -1
    pending_clear: int = -1


@dataclass
class DownsampleState:
    sigma_factor: float = defaults.DEFAULT_DOWNSAMPLE_SIGMA
    sigma_text: str = f"{defaults.DEFAULT_DOWNSAMPLE_SIGMA:.2f}"
    focused: bool = False
    pending_clear: bool = False
    dragging: bool = False
    dirty: bool = False


@dataclass
class UIState:
    project: Project

    render_mode: str = "edit"
    original_render_data: Optional[np.ndarray] = None
    current_render_data: Optional[np.ndarray] = None
    current_render_multiplier: float = defaults.RENDER_RESOLUTION_CHOICES[0]
    rendered_surface: Optional[object] = None

    render_menu: RenderMenuState = field(default_factory=RenderMenuState)
    highpass_menu: HighPassMenuState = field(default_factory=HighPassMenuState)
    downsample: DownsampleState = field(default_factory=DownsampleState)

    selected_idx: int = -1
    mouse_dragging: bool = False
    slider_dragging: int = -1
    last_mouse_pos: tuple[int, int] = (0, 0)

    field_dirty: bool = True

    supersample_factor: float = defaults.SUPERSAMPLE_CHOICES[0]
    supersample_index: int = 0
    noise_seed: int = defaults.DEFAULT_NOISE_SEED
    noise_seed_text: str = str(defaults.DEFAULT_NOISE_SEED)

    highres_render_data: Optional[np.ndarray] = None
    current_supersample: float = defaults.SUPERSAMPLE_CHOICES[0]
    current_noise_seed: int = defaults.DEFAULT_NOISE_SEED
    current_compute_resolution: tuple[int, int] = (0, 0)
    margin_factor: float = defaults.DEFAULT_PADDING_MARGIN
    current_canvas_scaled: tuple[int, int] = (0, 0)
    current_margin: float = 0.0
    current_render_shape: tuple[int, int] = (0, 0)
    canvas_width_text: str = ""
    canvas_height_text: str = ""
    canvas_focus: int = -1
    canvas_pending_clear: bool = False
    noise_sigma: float = defaults.DEFAULT_NOISE_SIGMA
