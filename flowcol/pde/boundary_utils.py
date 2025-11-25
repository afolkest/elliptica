"""
Helpers for boundary condition handling across PDEs.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import PDEDefinition
from flowcol.poisson import DIRICHLET

EDGE_NAMES = ("top", "bottom", "left", "right")


def build_default_bc_map(pde: PDEDefinition) -> Dict[str, Dict[str, Any]]:
    """Construct default BC map for a PDE based on its bc_fields metadata."""
    bc_map: Dict[str, Dict[str, Any]] = {edge: {} for edge in EDGE_NAMES}
    for field in pde.bc_fields:
        for edge in EDGE_NAMES:
            bc_map[edge][field.name] = field.default
    return bc_map


def resolve_bc_map(project, pde: PDEDefinition) -> Dict[str, Dict[str, Any]]:
    """Merge stored per-PDE BCs with defaults and legacy fields."""
    bc_map = build_default_bc_map(pde)
    stored = getattr(project, "pde_bc", {}) or {}
    pde_entry = stored.get(getattr(pde, "name", ""), {})

    # Overlay stored values
    for edge, fields in pde_entry.items():
        if edge not in bc_map:
            continue
        for key, value in fields.items():
            if key in bc_map[edge]:
                bc_map[edge][key] = value

    # Legacy compatibility: map boundary_{edge} ints to "type" if present
    if any("type" in bc_map[e] for e in bc_map):
        for edge in EDGE_NAMES:
            legacy_val = getattr(project, f"boundary_{edge}", None)
            if legacy_val is not None and "type" in bc_map[edge]:
                bc_map[edge]["type"] = legacy_val

    return bc_map


def bc_map_to_legacy(bc_map: Dict[str, Dict[str, Any]], default: int = DIRICHLET) -> Dict[str, int]:
    """Convert bc_map to legacy int codes using the 'type' field if present."""
    legacy: Dict[str, int] = {}
    for edge in EDGE_NAMES:
        edge_map = bc_map.get(edge, {})
        if "type" in edge_map:
            try:
                legacy[edge] = int(edge_map["type"])
            except Exception:
                legacy[edge] = default
        else:
            legacy[edge] = default
    return legacy
