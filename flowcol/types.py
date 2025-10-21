"""Core data types for flowcol."""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional


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
    canvas_resolution: tuple[int, int] = (1024, 1024)
    streamlength_factor: float = 30.0 / 1024.0
    renders: list[RenderInfo] = field(default_factory=list)          


@dataclass
class RenderMenuState:
    is_open: bool = False
    selected_multiplier: float = 1.0
    num_passes: int = 1
    input_focused: bool = False
    streamlength_text: str = "0.0293"
    streamlength_input_focused: bool = False
    streamlength_pending_clear: bool = False
    pending_streamlength_factor: float = 30.0 / 1024.0
    seed_input_focused: bool = False
    seed_pending_clear: bool = False
    margin_text: str = "0.10"
    margin_input_focused: bool = False
    margin_pending_clear: bool = False
    pending_margin_factor: float = 0.10


@dataclass
class HighPassMenuState:
    is_open: bool = False
    sigma_factor: float = 3.0 / 1024.0
    clip_limit: float = 0.01
    kernel_rows: int = 8
    kernel_cols: int = 8
    num_bins: int = 150
    sigma_factor_text: str = "0.0029"
    clip_text: str = "0.01"
    kernel_rows_text: str = "8"
    kernel_cols_text: str = "8"
    num_bins_text: str = "150"
    focused_field: int = -1
    pending_clear: int = -1


@dataclass
class DownsampleState:
    sigma_factor: float = 0.6
    sigma_text: str = "0.60"
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
    current_render_multiplier: float = 1.0
    rendered_surface: Optional[object] = None

    render_menu: RenderMenuState = field(default_factory=RenderMenuState)
    highpass_menu: HighPassMenuState = field(default_factory=HighPassMenuState)
    downsample: DownsampleState = field(default_factory=DownsampleState)

    selected_idx: int = -1
    mouse_dragging: bool = False
    slider_dragging: int = -1
    last_mouse_pos: tuple[int, int] = (0, 0)

    field_dirty: bool = True

    supersample_factor: float = 1.0
    supersample_index: int = 0
    noise_seed: int = 0
    noise_seed_text: str = "0"

    highres_render_data: Optional[np.ndarray] = None
    current_supersample: float = 1.0
    current_noise_seed: int = 0
    current_compute_resolution: tuple[int, int] = (0, 0)
    margin_factor: float = 0.10
    current_canvas_scaled: tuple[int, int] = (0, 0)
    current_margin: float = 0.0
    current_render_shape: tuple[int, int] = (0, 0)
