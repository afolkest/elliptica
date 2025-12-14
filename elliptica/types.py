"""Core data types for elliptica - framework-agnostic."""

from dataclasses import dataclass, field
import numpy as np
from elliptica import defaults
from elliptica.poisson import DIRICHLET


class BoundaryObject:
    """Generic boundary object for any PDE.

    Attributes:
        mask: Binary mask defining the boundary region
        params: PDE-specific parameters (e.g. {"voltage": 1.0} for Poisson)
        position: (x, y) position on canvas
        interior_mask: Auto-detected interior region (for ring shapes etc.)
        original_mask: Original mask before scaling
        original_interior_mask: Original interior before scaling
        scale_factor: Current scale relative to original
        edge_smooth_sigma: Edge anti-aliasing blur in pixels (0-5px)
        smear_enabled: Enable texture smearing inside boundary
        smear_sigma: Smear strength (fraction of canvas)
        id: Unique ID assigned when added to project
    """

    def __init__(
        self,
        mask: np.ndarray,
        params: dict[str, float] | None = None,
        position: tuple[float, float] = (0.0, 0.0),
        interior_mask: np.ndarray | None = None,
        original_mask: np.ndarray | None = None,
        original_interior_mask: np.ndarray | None = None,
        scale_factor: float = 1.0,
        edge_smooth_sigma: float = 0.0,
        smear_enabled: bool = False,
        smear_sigma: float = defaults.DEFAULT_SMEAR_SIGMA,
        id: int | None = None,
    ):
        self.mask = mask
        self.params = params if params is not None else {}
        self.position = position
        self.interior_mask = interior_mask
        self.original_mask = original_mask
        self.original_interior_mask = original_interior_mask
        self.scale_factor = scale_factor
        self.edge_smooth_sigma = edge_smooth_sigma
        self.smear_enabled = smear_enabled
        self.smear_sigma = smear_sigma
        self.id = id

    def __repr__(self) -> str:
        return (
            f"BoundaryObject(id={self.id}, params={self.params}, "
            f"position={self.position}, mask={self.mask.shape})"
        )


def clone_boundary_object(boundary: BoundaryObject, preserve_id: bool = True) -> BoundaryObject:
    """Deep-copy a boundary object with all its data.

    Args:
        boundary: The boundary object to clone
        preserve_id: If True, preserve the original id (for rendering snapshots).
                     If False, set id=None so a new id is assigned on add (for clipboard paste).

    Returns:
        A deep copy of the boundary object
    """
    interior = boundary.interior_mask.copy() if boundary.interior_mask is not None else None
    original = boundary.original_mask.copy() if boundary.original_mask is not None else None
    original_interior = boundary.original_interior_mask.copy() if boundary.original_interior_mask is not None else None

    return BoundaryObject(
        mask=boundary.mask.copy(),
        params=boundary.params.copy(),
        position=boundary.position,
        interior_mask=interior,
        original_mask=original,
        original_interior_mask=original_interior,
        scale_factor=boundary.scale_factor,
        edge_smooth_sigma=boundary.edge_smooth_sigma,
        smear_enabled=boundary.smear_enabled,
        smear_sigma=boundary.smear_sigma,
        id=boundary.id if preserve_id else None,
    )



@dataclass
class RenderInfo:
    multiplier: float
    filepath: str
    timestamp: str


@dataclass
class Project:
    boundary_objects: list[BoundaryObject] = field(default_factory=list)
    canvas_resolution: tuple[int, int] = defaults.DEFAULT_CANVAS_RESOLUTION
    streamlength_factor: float = defaults.DEFAULT_STREAMLENGTH_FACTOR
    renders: list[RenderInfo] = field(default_factory=list)
    next_boundary_id: int = 0  # Incremental counter for boundary object IDs
    # Boundary conditions for Poisson solver (DIRICHLET=0 or NEUMANN=1)
    boundary_top: int = DIRICHLET
    boundary_bottom: int = DIRICHLET
    boundary_left: int = DIRICHLET
    boundary_right: int = DIRICHLET

    # === Multi-PDE support ===
    pde_type: str = "poisson"  # Active PDE type
    pde_params: dict = field(default_factory=dict)  # PDE-specific parameters
    pde_bc: dict = field(default_factory=dict)  # Per-PDE boundary condition values

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
