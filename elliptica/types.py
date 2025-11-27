"""Core data types for elliptica - framework-agnostic."""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from elliptica import defaults
from elliptica.poisson import DIRICHLET


@dataclass
class BoundaryObject:
    """Generic boundary object for any PDE (replaces Conductor)."""
    mask: np.ndarray
    params: dict[str, float] = field(default_factory=dict)  # Generic parameters
    position: tuple[float, float] = (0.0, 0.0)
    interior_mask: Optional[np.ndarray] = None
    original_mask: Optional[np.ndarray] = None
    original_interior_mask: Optional[np.ndarray] = None
    scale_factor: float = 1.0  # Current scale relative to original
    edge_smooth_sigma: float = 1.5  # Edge anti-aliasing blur in pixels (0-5px range)
    smear_enabled: bool = False  # Enable texture smearing inside boundary
    id: Optional[int] = None  # Assigned when added to project

    def __init__(self, mask: np.ndarray, voltage: float = 0.0, params: Optional[dict[str, float]] = None, **kwargs):
        self.mask = mask
        self.params = params if params is not None else {}
        if "voltage" not in self.params:
            self.params["voltage"] = voltage
        
        # Handle other fields
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Set defaults for missing fields (simulating dataclass behavior)
        if not hasattr(self, 'position'): self.position = (0.0, 0.0)
        if not hasattr(self, 'interior_mask'): self.interior_mask = None
        if not hasattr(self, 'original_mask'): self.original_mask = None
        if not hasattr(self, 'original_interior_mask'): self.original_interior_mask = None
        if not hasattr(self, 'scale_factor'): self.scale_factor = 1.0
        if not hasattr(self, 'edge_smooth_sigma'): self.edge_smooth_sigma = 1.5
        if not hasattr(self, 'smear_enabled'): self.smear_enabled = False
        if not hasattr(self, 'smear_sigma'): self.smear_sigma = defaults.DEFAULT_SMEAR_SIGMA
        if not hasattr(self, 'id'): self.id = None

    # Backwards compatibility for 'voltage'
    @property
    def voltage(self) -> float:
        return self.params.get("voltage", 0.0)

    @voltage.setter
    def voltage(self, v: float) -> None:
        self.params["voltage"] = v

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
    pde_bc: dict = field(default_factory=dict)  # Per-PDE boundary condition values

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
    def solve_scale(self) -> float:
        """Scale factor for PDE solver (always 1.0 for now)."""
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
