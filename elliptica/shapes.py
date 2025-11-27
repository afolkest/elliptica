"""Geometric shape generators for conductors.

Data-driven shape system - add new shapes by extending SHAPES registry.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class ParamSpec:
    """Parameter specification for shape generator."""
    name: str  # Display name
    type: type  # int or float
    default: float
    min_val: float
    max_val: float


@dataclass
class ShapeSpec:
    """Shape specification with generator function and parameters."""
    name: str  # Display name
    params: list[ParamSpec]
    generator: Callable[..., tuple[np.ndarray, np.ndarray | None]]


def make_disk(radius: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Create a solid disk (filled circle).

    Args:
        radius: Disk radius in pixels

    Returns:
        (surface_mask, interior_mask) - interior is None for solid disk
    """
    # Bounding box fits the radius with 1px padding
    size = int(2 * radius) + 3
    h = w = size
    y, x = np.ogrid[:h, :w]
    cy = cx = (size - 1) / 2.0
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2
    mask = mask.astype(np.float32)
    return mask, None


def make_annulus(outer_radius: float, inner_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Create an annulus (ring) with hollow interior.

    Args:
        outer_radius: Outer radius in pixels
        inner_radius: Inner radius in pixels

    Returns:
        (surface_mask, interior_mask)
    """
    # Bounding box fits the outer radius with 1px padding
    size = int(2 * outer_radius) + 3
    h = w = size
    y = np.arange(h, dtype=np.float32) - (h - 1) * 0.5
    x = np.arange(w, dtype=np.float32) - (w - 1) * 0.5
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)

    surface = ((rr >= inner_radius) & (rr <= outer_radius)).astype(np.float32)
    interior = (rr < inner_radius).astype(np.float32)
    return surface, interior


def make_rectangle(width: float, height: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Create a solid filled rectangle.

    Args:
        width: Rectangle width in pixels
        height: Rectangle height in pixels

    Returns:
        (surface_mask, interior_mask) - interior is None for solid rectangle
    """
    h = int(height) + 1
    w = int(width) + 1
    mask = np.ones((h, w), dtype=np.float32)
    return mask, None


def make_hollow_rectangle(width: float, height: float, thickness: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a hollow rectangle (frame).

    Args:
        width: Rectangle width in pixels
        height: Rectangle height in pixels
        thickness: Frame thickness in pixels

    Returns:
        (surface_mask, interior_mask)
    """
    h = int(height) + 1
    w = int(width) + 1
    thickness = int(thickness)

    # Full rectangle
    surface = np.ones((h, w), dtype=np.float32)

    # Interior (hollow part) - shrink by thickness on all sides
    interior = np.zeros((h, w), dtype=np.float32)
    inner_h_start = thickness
    inner_h_end = h - thickness
    inner_w_start = thickness
    inner_w_end = w - thickness

    if inner_h_end > inner_h_start and inner_w_end > inner_w_start:
        interior[inner_h_start:inner_h_end, inner_w_start:inner_w_end] = 1.0

    # Surface is only the frame (full rectangle minus interior)
    surface = surface - interior

    return surface, interior


# Shape registry - add new shapes here!
SHAPES: dict[str, ShapeSpec] = {
    "disk": ShapeSpec(
        name="Disk",
        params=[
            ParamSpec("radius", float, 80.0, 5.0, 1000.0),
        ],
        generator=make_disk,
    ),
    "annulus": ShapeSpec(
        name="Annulus",
        params=[
            ParamSpec("outer_radius", float, 120.0, 10.0, 1000.0),
            ParamSpec("inner_radius", float, 80.0, 5.0, 1000.0),
        ],
        generator=make_annulus,
    ),
    "rectangle": ShapeSpec(
        name="Rectangle",
        params=[
            ParamSpec("width", float, 150.0, 10.0, 2000.0),
            ParamSpec("height", float, 100.0, 10.0, 2000.0),
        ],
        generator=make_rectangle,
    ),
    "hollow_rectangle": ShapeSpec(
        name="Hollow Rectangle",
        params=[
            ParamSpec("width", float, 200.0, 10.0, 2000.0),
            ParamSpec("height", float, 150.0, 10.0, 2000.0),
            ParamSpec("thickness", float, 20.0, 1.0, 500.0),
        ],
        generator=make_hollow_rectangle,
    ),
}
