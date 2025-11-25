"""
Analytic convergence check for the biharmonic solver using a manufactured solution.

We pick a known biharmonic function on [0, 1] x [0, 1]:
    phi(x, y) = x^4 - 6 x^2 y^2 + y^4
which satisfies Δ² phi = 0. We impose phi and its boundary Laplacian on
all four edges and measure error against the numeric solve.
"""

from __future__ import annotations

import numpy as np
import os
import sys

# Ensure repository root is on path
sys.path.append(os.getcwd())

from flowcol.poisson import DIRICHLET
from flowcol.pde.biharmonic_pde import solve_biharmonic


def phi_exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic biharmonic potential."""
    return x**4 - 6.0 * x * x * y * y + y**4


def dphi_dx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 4.0 * x**3 - 12.0 * x * y * y


def dphi_dy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -12.0 * x * x * y + 4.0 * y**3


class DummyProject:
    """Minimal project-like container expected by solve_biharmonic."""

    def __init__(self, h: int, w: int, bc_map):
        self.shape = (h, w)
        self.boundary_objects = []
        self.bc = bc_map
        self.domain_size = (1.0, 1.0)  # physical size
        self.margin = (0.0, 0.0)


def build_bc_map() -> dict:
    """Build edge BCs with φ and Δφ from the analytic solution."""
    edges = {}
    edges["top"] = {
        "type": DIRICHLET,
        "value": lambda x: phi_exact(x, 0.0),
        "laplacian": lambda x: (dphi_dx(x, 0.0) * 0.0 + dphi_dy(x * 0 + x, 0.0) * 0.0),  # placeholder, overwritten below
    }
    edges["bottom"] = {
        "type": DIRICHLET,
        "value": lambda x: phi_exact(x, 1.0),
        "laplacian": lambda x: 0.0,
    }
    edges["left"] = {
        "type": DIRICHLET,
        "value": lambda y: phi_exact(0.0, y),
        "laplacian": lambda y: 0.0,
    }
    edges["right"] = {
        "type": DIRICHLET,
        "value": lambda y: phi_exact(1.0, y),
        "laplacian": lambda y: 0.0,
    }
    return edges


def discretize_bc(edges, xs: np.ndarray, ys: np.ndarray) -> dict:
    """Sample the BC functions onto the grid edges."""
    bc_map = {edge: {"type": DIRICHLET} for edge in edges}
    bc_map["top"]["value"] = edges["top"]["value"](xs)
    bc_map["bottom"]["value"] = edges["bottom"]["value"](xs)
    bc_map["left"]["value"] = edges["left"]["value"](ys)
    bc_map["right"]["value"] = edges["right"]["value"](ys)
    bc_map["top"]["laplacian"] = np.zeros_like(xs)
    bc_map["bottom"]["laplacian"] = np.zeros_like(xs)
    bc_map["left"]["laplacian"] = np.zeros_like(ys)
    bc_map["right"]["laplacian"] = np.zeros_like(ys)
    return bc_map


def run_case(n: int = 64) -> tuple[float, float]:
    """Solve on an n x n grid and return (L2, Linf) errors."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys)

    edges = build_bc_map()
    bc_map = discretize_bc(edges, xs, ys)

    proj = DummyProject(n, n, bc_map)
    phi_num = solve_biharmonic(proj)["phi"]
    phi_true = phi_exact(X, Y).astype(np.float32)

    diff = phi_num - phi_true
    l2 = np.sqrt(np.mean(diff**2))
    linf = np.max(np.abs(diff))
    return l2, linf


def main() -> None:
    for n in (32, 64, 96, 128, 256, 384, 512):
        l2, linf = run_case(n)
        print(f"{n:4d}^2 grid -> L2={l2:.3e}, Linf={linf:.3e}")


if __name__ == "__main__":
    main()
