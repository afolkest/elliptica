"""
Base classes for PDE definitions.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np


@dataclass
class BCField:
    """Metadata for a boundary condition field (per edge).

    Attributes:
        visible_when: Optional dict mapping field_name -> required_value.
            If specified, this field is only shown when all conditions are met.
            Example: {"type": 0} means show only when the "type" field equals 0.
    """
    name: str
    display_name: str
    field_type: str  # "enum", "float", "int", "bool"
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    choices: list[tuple[str, Any]] = field(default_factory=list)  # For enum: [(label, value)]
    description: str = ""
    visible_when: dict[str, Any] | None = None


@dataclass
class BoundaryParameter:
    """Metadata for a parameter controllable on boundary objects."""
    name: str
    display_name: str
    min_value: float
    max_value: float
    default_value: float
    description: str = ""


@dataclass
class PDEDefinition:
    """
    Definition of a PDE for the system.

    Each PDE can return arbitrary solution fields in a dict,
    then extract a 2D vector field for LIC visualization.
    """
    # Basic metadata
    name: str
    display_name: str
    description: str

    # Core computation functions
    solve: Callable[[Any], dict[str, np.ndarray]]
    """Solve the PDE, returning dict of solution arrays."""

    extract_lic_field: Callable[[dict[str, np.ndarray], Any], tuple[np.ndarray, np.ndarray]]
    """Extract (ex, ey) vector field for LIC from solution dict."""

    # UI Metadata
    boundary_params: list[BoundaryParameter] = field(default_factory=list)
    
    # Global boundary condition options (e.g., {"Dirichlet": 0, "Neumann": 1})
    # If empty, no global BC controls will be shown.
    global_bc_options: dict[str, int] = field(default_factory=dict)
    
    # Rich boundary condition fields (preferred over global_bc_options)
    bc_fields: list[BCField] = field(default_factory=list)
    """Fields for domain edge boundary conditions (top/bottom/left/right)."""

    # Rich fields for interior boundary objects
    boundary_fields: list[BCField] = field(default_factory=list)
    """Fields for interior boundary objects (enums, floats, etc.). Supplements boundary_params."""
