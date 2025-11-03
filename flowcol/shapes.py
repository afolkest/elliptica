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


def make_disk(size: int, radius: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Create a solid disk (filled circle).

    Args:
        size: Array size (width and height)
        radius: Disk radius in pixels

    Returns:
        (surface_mask, interior_mask) - interior is None for solid disk
    """
    h = w = int(size)
    y, x = np.ogrid[:h, :w]
    cy = cx = (size - 1) / 2.0
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2
    mask = mask.astype(np.float32)
    return mask, None


def make_annulus(size: int, outer_radius: float, inner_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Create an annulus (ring) with hollow interior.

    Args:
        size: Array size (width and height)
        outer_radius: Outer radius in pixels
        inner_radius: Inner radius in pixels

    Returns:
        (surface_mask, interior_mask)
    """
    h = w = int(size)
    y = np.arange(h, dtype=np.float32) - (h - 1) * 0.5
    x = np.arange(w, dtype=np.float32) - (w - 1) * 0.5
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)

    surface = ((rr >= inner_radius) & (rr <= outer_radius)).astype(np.float32)
    interior = (rr < inner_radius).astype(np.float32)
    return surface, interior


# Shape registry - add new shapes here!
SHAPES: dict[str, ShapeSpec] = {
    "disk": ShapeSpec(
        name="Disk",
        params=[
            ParamSpec("size", int, 200, 10, 2000),
            ParamSpec("radius", float, 80.0, 5.0, 1000.0),
        ],
        generator=make_disk,
    ),
    "annulus": ShapeSpec(
        name="Annulus",
        params=[
            ParamSpec("size", int, 300, 10, 2000),
            ParamSpec("outer_radius", float, 120.0, 10.0, 1000.0),
            ParamSpec("inner_radius", float, 80.0, 5.0, 1000.0),
        ],
        generator=make_annulus,
    ),
}
