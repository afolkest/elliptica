"""Core data types for flowcol - framework-agnostic."""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from flowcol import defaults
from flowcol.poisson import DIRICHLET


@dataclass
class BoundaryObject:
    """Generic boundary object for any PDE (replaces Conductor)."""
    mask: np.ndarray
    voltage: float = 0.0  # Primary field for backwards compatibility
    position: tuple[float, float] = (0.0, 0.0)
    interior_mask: Optional[np.ndarray] = None
    original_mask: Optional[np.ndarray] = None
    original_interior_mask: Optional[np.ndarray] = None
    scale_factor: float = 1.0  # Current scale relative to original
    edge_smooth_sigma: float = 1.5  # Edge anti-aliasing blur in pixels (0-5px range)
    smear_enabled: bool = False  # Enable texture smearing inside boundary
    smear_sigma: float = defaults.DEFAULT_SMEAR_SIGMA  # Gaussian blur strength as fraction of canvas width
    id: Optional[int] = None  # Assigned when added to project

    # Generic 'value' is an alias for 'voltage' for future PDE compatibility
    @property
    def value(self) -> float:
        """Generic boundary value (alias for voltage)."""
        return self.voltage

    @value.setter
    def value(self, v: float) -> None:
        """Set generic boundary value."""
        self.voltage = v


# Legacy alias for compatibility
Conductor = BoundaryObject


# Keep original Conductor class for reference (will be removed later)
@dataclass
class _LegacyConductor:
    mask: np.ndarray
    voltage: float
    position: tuple[float, float] = (0.0, 0.0)
    interior_mask: Optional[np.ndarray] = None
    original_mask: Optional[np.ndarray] = None
    original_interior_mask: Optional[np.ndarray] = None
    scale_factor: float = 1.0  # Current scale relative to original
    edge_smooth_sigma: float = 1.5  # Edge anti-aliasing blur in pixels (0-5px range)
    smear_enabled: bool = False  # Enable texture smearing inside conductor
    smear_sigma: float = defaults.DEFAULT_SMEAR_SIGMA  # Gaussian blur strength as fraction of canvas width (resolution-independent)
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

    # === Multi-PDE support ===
    pde_type: str = "poisson"  # Active PDE type
    pde_params: dict = field(default_factory=dict)  # PDE-specific parameters

    # === Compatibility accessors ===
    @property
    def boundary_objects(self) -> list[BoundaryObject]:
        """Generic accessor for boundary objects (same as conductors)."""
        return self.conductors

    @boundary_objects.setter
    def boundary_objects(self, objs: list[BoundaryObject]) -> None:
        """Generic setter for boundary objects."""
        self.conductors = objs

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape for PDE solving (height, width)."""
        return self.canvas_resolution

    @property
    def poisson_scale(self) -> float:
        """Scale factor for Poisson solver (always 1.0 for now)."""
        return 1.0

    @property
    def boundary_conditions(self) -> dict:
        """Get boundary conditions as dict for solver."""
        return {
            'top': self.boundary_top,
            'bottom': self.boundary_bottom,
            'left': self.boundary_left,
            'right': self.boundary_right
        }
