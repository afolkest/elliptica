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
class Project:
    conductors: list[Conductor] = field(default_factory=list)
    canvas_resolution: tuple[int, int] = (1024, 1024)  
    streamlength: int = 30          


@dataclass
class UIState:
    project: Project

    render_mode: str = "edit"
    rendered_surface: Optional[object] = None

    selected_idx: int = -1
    mouse_dragging: bool = False
    slider_dragging: int = -1
    last_mouse_pos: tuple[int, int] = (0, 0)

    field_cache: Optional[tuple[np.ndarray, np.ndarray]] = None
    field_dirty: bool = True



