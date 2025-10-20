"""Core data types for flowcol."""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class Conductor:
    mask: np.ndarray                
    voltage: float                  
    position: tuple[float, float] = (0.0, 0.0)  


@dataclass
class Project:
    conductors: list[Conductor] = field(default_factory=list)
    canvas_resolution: tuple[int, int] = (1024, 1024)  
    streamlength: int = 30          


@dataclass
class UIState:
    project: Project

    preview_resolution: tuple[int, int] = (256, 256)   
    render_resolution: tuple[int, int] = (4096, 4096)  

    selected_idx: int = -1
    mouse_dragging: bool = False
    last_mouse_pos: tuple[int, int] = (0, 0)

    # Cached fields at different resolutions
    field_cache: dict = field(default_factory=dict)  # {resolution: (Ex, Ey)}
    field_dirty: bool = True



