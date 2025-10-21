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
    selected_multiplier: int = 1
    num_passes: int = 1
    input_focused: bool = False
    streamlength_text: str = "0.0293"
    streamlength_input_focused: bool = False
    streamlength_pending_clear: bool = False
    pending_streamlength_factor: float = 30.0 / 1024.0


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
class UIState:
    project: Project

    render_mode: str = "edit"
    original_render_data: Optional[np.ndarray] = None
    current_render_data: Optional[np.ndarray] = None
    current_render_multiplier: int = 1
    rendered_surface: Optional[object] = None

    render_menu: RenderMenuState = field(default_factory=RenderMenuState)
    highpass_menu: HighPassMenuState = field(default_factory=HighPassMenuState)

    selected_idx: int = -1
    mouse_dragging: bool = False
    slider_dragging: int = -1
    last_mouse_pos: tuple[int, int] = (0, 0)

    field_dirty: bool = True
