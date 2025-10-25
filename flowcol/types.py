"""Core data types for flowcol - framework-agnostic."""

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
    original_mask: Optional[np.ndarray] = None
    original_interior_mask: Optional[np.ndarray] = None
    scale_factor: float = 1.0  # Current scale relative to original
    id: Optional[int] = None  # Assigned when added to project


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
    next_conductor_id: int = 0  # Incremental counter for conductor IDs
