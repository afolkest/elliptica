"""
Helpers for boundary condition handling across PDEs.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.ndimage import zoom

from .base import PDEDefinition
from flowcol.poisson import DIRICHLET
from flowcol.mask_utils import blur_mask

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


def build_dirichlet_from_objects(project: Any) -> tuple[np.ndarray, np.ndarray]:
    """Construct Dirichlet mask and values from boundary objects.

    Args:
        project: Project-like object with attributes:
            - boundary_objects: list of boundary objects with mask, position, voltage/value
            - shape: (grid_h, grid_w) tuple
            - domain_size: (domain_w, domain_h) tuple (optional, defaults to shape)
            - margin: (margin_x, margin_y) tuple (optional, defaults to (0, 0))

    Returns:
        (mask, values) where mask is bool array and values is float array
    """
    boundary_objects = project.boundary_objects
    grid_h, grid_w = project.shape

    mask = np.zeros((grid_h, grid_w), dtype=bool)
    values = np.zeros((grid_h, grid_w), dtype=float)

    if hasattr(project, 'domain_size'):
        domain_w, domain_h = project.domain_size
        grid_scale_x = grid_w / domain_w if domain_w > 0 else 1.0
        grid_scale_y = grid_h / domain_h if domain_h > 0 else 1.0
    else:
        grid_scale_x = grid_scale_y = 1.0
    margin_x, margin_y = project.margin if hasattr(project, 'margin') else (0, 0)

    for obj in boundary_objects:
        if hasattr(obj, 'position'):
            x = (obj.position[0] + margin_x) * grid_scale_x
            y = (obj.position[1] + margin_y) * grid_scale_y
        else:
            x = margin_x * grid_scale_x
            y = margin_y * grid_scale_y

        obj_mask = obj.mask
        if not np.isclose(grid_scale_x, 1.0) or not np.isclose(grid_scale_y, 1.0):
            obj_mask = zoom(obj_mask, (grid_scale_y, grid_scale_x), order=0)

        if hasattr(obj, 'edge_smooth_sigma') and obj.edge_smooth_sigma > 0:
            scale_factor = (grid_scale_x + grid_scale_y) / 2.0
            scaled_sigma = obj.edge_smooth_sigma * scale_factor
            obj_mask = blur_mask(obj_mask, scaled_sigma)

        mask_h, mask_w = obj_mask.shape
        ix, iy = int(round(x)), int(round(y))
        x0, y0 = max(0, ix), max(0, iy)
        x1, y1 = min(ix + mask_w, grid_w), min(iy + mask_h, grid_h)
        mx0, my0 = max(0, -ix), max(0, -iy)
        mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

        mask_slice = obj_mask[my0:my1, mx0:mx1]
        mask_bool = mask_slice > 0.5
        value = obj.voltage if hasattr(obj, 'voltage') else obj.value

        mask[y0:y1, x0:x1] |= mask_bool
        values[y0:y1, x0:x1] = np.where(mask_bool, value, values[y0:y1, x0:x1])

    return mask, values
