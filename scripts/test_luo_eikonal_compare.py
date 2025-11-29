#!/usr/bin/env python
"""
Compare the experimental Luo-style factored eikonal solver against the
baseline FMM (skfmm) φ for a few toy scenarios.

Run from repo root (after activating venv):

    PYTHONPATH=. venv/bin/python scripts/test_luo_eikonal_compare.py
"""

from __future__ import annotations

import time
from typing import Tuple
import os

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt

from elliptica.go_luo.factored_eikonal import solve_factored_eikonal


def build_phi(src_mask: np.ndarray, n_field: np.ndarray, order: int = 2) -> np.ndarray:
    """Solve |∇φ| = n with skfmm for a given source mask and n."""
    phi_init = distance_transform_edt(~src_mask).astype(float)
    phi_init[src_mask] = -distance_transform_edt(src_mask)[src_mask]
    speed = 1.0 / n_field
    phi = skfmm.travel_time(phi_init, speed, order=order)
    phi[src_mask] = 0.0
    return phi.astype(np.float64, copy=False)


def make_disk(center: Tuple[int, int], radius: int, size: int) -> np.ndarray:
    h = w = size
    cy, cx = center
    Y, X = np.mgrid[0:h, 0:w]
    r = np.hypot(Y - cy, X - cx)
    return r <= radius


def run_case(
    name: str,
    n_field: np.ndarray,
    src_idx: Tuple[int, int],
    h: float = 1.0,
    gauss_seidel: bool = False,
) -> None:
    """Build φ with FMM and τ with the factored solver; report errors."""
    src_mask = np.zeros_like(n_field, dtype=bool)
    src_mask[src_idx] = True

    t0 = time.time()
    phi = build_phi(src_mask, n_field)
    t_phi = time.time() - t0

    s_field = n_field.astype(np.float64, copy=False)  # slowness == refractive index

    t1 = time.time()
    res = solve_factored_eikonal(
        s_field,
        h=h,
        source_idx=src_idx,
        v_ref=1.0 / s_field[src_idx],
        gauss_seidel=gauss_seidel,
    )
    t_tau = time.time() - t1

    tau = res.tau

    err = tau - phi
    rms = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))
    u_min = float(res.u.min())
    u_max = float(res.u.max())

    print(
        f"{name}: shape={n_field.shape}, src={src_idx}, "
        f"t_phi={t_phi:.3f}s t_tau={t_tau:.3f}s "
        f"GS={gauss_seidel} "
        f"phi[min,max]=({phi.min():.3g},{phi.max():.3g}) "
        f"tau[min,max]=({tau.min():.3g},{tau.max():.3g}) "
        f"u[min,max]=({u_min:.3g},{u_max:.3g}) "
        f"err_rms={rms:.3e} err_max={max_abs:.3e}"
    )


def main() -> None:
    size = 201
    center = (size // 2, size // 2)
    offset = (size // 3, size // 4)
    gs = os.environ.get("LuoGS", "0") == "1"

    # Homogeneous n=1
    n_hom = np.ones((size, size), dtype=np.float64)
    run_case("homog-center", n_hom, center, gauss_seidel=gs)
    run_case("homog-offset", n_hom, offset, gauss_seidel=gs)

    # Simple lens: n=1.5 disk on the right
    lens_center = (size // 2, 3 * size // 4)
    lens_radius = size // 8
    lens_mask = make_disk(lens_center, lens_radius, size)
    n_lens = np.ones((size, size), dtype=np.float64)
    n_lens[lens_mask] = 1.5

    run_case("lens-center", n_lens, center, gauss_seidel=gs)
    run_case("lens-offset", n_lens, offset, gauss_seidel=gs)

    # Slab interface: left n=1, right n=1.5
    slab = np.ones((size, size), dtype=np.float64)
    slab[:, size // 2 :] = 1.5
    run_case("slab-center", slab, center, gauss_seidel=gs)
    run_case("slab-offset", slab, offset, gauss_seidel=gs)


if __name__ == "__main__":
    main()
