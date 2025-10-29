"""Core data types for flowcol - framework-agnostic."""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from flowcol import defaults
from flowcol.poisson import DIRICHLET


@dataclass
class Conductor:
    mask: np.ndarray
    voltage: float
    position: tuple[float, float] = (0.0, 0.0)
    interior_mask: Optional[np.ndarray] = None
    original_mask: Optional[np.ndarray] = None
    original_interior_mask: Optional[np.ndarray] = None
    scale_factor: float = 1.0  # Current scale relative to original
    blur_sigma: float = 0.0  # Gaussian blur sigma (pixels or fraction)
    blur_is_fractional: bool = False  # If True, blur_sigma is fraction of canvas size
    smear_enabled: bool = False  # Enable texture smearing inside conductor
    smear_sigma: float = 2.0  # Gaussian blur for LIC texture in pixels
    id: Optional[int] = None  # Assigned when added to project


@dataclass
class RenderInfo:
    multiplier: float
    filepath: str
    timestamp: str


@dataclass
class Project:
    conductors: list[Conductor] = field(default_factory=list)
    canvas_resolution: tuple[int, int] = defaults.DEFAULT_CANVAS_RESOLUTION
    streamlength_factor: float = defaults.DEFAULT_STREAMLENGTH_FACTOR
    renders: list[RenderInfo] = field(default_factory=list)
    next_conductor_id: int = 0  # Incremental counter for conductor IDs
    # Boundary conditions for Poisson solver (DIRICHLET=0 or NEUMANN=1)
    boundary_top: int = DIRICHLET
    boundary_bottom: int = DIRICHLET
    boundary_left: int = DIRICHLET
    boundary_right: int = DIRICHLET
