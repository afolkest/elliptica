#!/usr/bin/env python
"""
Ad-hoc test harness for compute_amplitude_jacobian (ray-label Jacobian method).

Run from repo root after activating the venv:

    source venv/bin/activate
    PYTHONPATH=. python scripts/test_eikonal_amp_jacobian.py

This compares the Jacobian-based amplitude solver against the characteristic
reference solver for a few representative scenarios and prints error and
runtime metrics.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt

from elliptica.pde.eikonal_amp import (
    compute_amplitude_characteristic,
    compute_amplitude_jacobian,
)


def build_phi(
    size: int,
    src_mask: np.ndarray,
    n_field: np.ndarray,
) -> np.ndarray:
    """Solve eikonal |∇φ| = n with Skfmm for a given source mask and n."""
    phi_init = distance_transform_edt(~src_mask).astype(float)
    phi_init[src_mask] = -distance_transform_edt(src_mask)[src_mask]
    speed = 1.0 / n_field
    phi = skfmm.travel_time(phi_init, speed, order=2)
    phi[src_mask] = 0.0
    return phi.astype(np.float32)


def make_disk(center: Tuple[int, int], radius: int, size: int) -> np.ndarray:
    h = w = size
    cy, cx = center
    Y, X = np.mgrid[0:h, 0:w]
    r = np.hypot(Y - cy, X - cx)
    return r <= radius


def radial_profile(A: np.ndarray, cy: int, cx: int, max_r: int = 40) -> np.ndarray:
    h, w = A.shape
    Y, X = np.mgrid[0:h, 0:w]
    r = np.hypot(Y - cy, X - cx)
    r_int = r.astype(int)
    prof = []
    for R in range(max_r):
        mask = r_int == R
        if mask.any():
            prof.append(float(A[mask].mean()))
        else:
            prof.append(float("nan"))
    return np.array(prof, dtype=float)


def scenario_homogeneous(size: int = 256, radius: int = 8) -> None:
    print(f"\n=== Homogeneous disk (Jacobian), size={size}, radius={radius} ===")
    h = w = size
    centers = [
        (h // 2, w // 2),
        (h // 4, w // 4),
        (3 * h // 4, 3 * w // 4),
    ]
    for cy, cx in centers:
        src = make_disk((cy, cx), radius, size)
        n_field = np.ones((h, w), dtype=np.float32)
        phi = build_phi(size, src, n_field)

        t0 = time.time()
        A_ref = compute_amplitude_characteristic(phi, n_field, src, step_size=0.5)
        t_ref = time.time() - t0

        t1 = time.time()
        A_j = compute_amplitude_jacobian(
            phi,
            n_field,
            src,
            step_size=0.75,
            n_iters=40,
            smooth_sigma=0.5,
        )
        t_j = time.time() - t1

        # Axis symmetry through the source.
        y_mid_ref = A_ref[:, cx]
        x_mid_ref = A_ref[cy, :]
        axis_l1_ref = float(np.mean(np.abs(y_mid_ref - x_mid_ref)))

        y_mid_j = A_j[:, cx]
        x_mid_j = A_j[cy, :]
        axis_l1_j = float(np.mean(np.abs(y_mid_j - x_mid_j)))

        # Radial profiles and error between solvers.
        prof_ref = radial_profile(A_ref, cy, cx)
        prof_j = radial_profile(A_j, cy, cx)
        valid = np.isfinite(prof_ref) & np.isfinite(prof_j)
        if valid.any():
            prof_err_L2 = float(np.sqrt(np.mean((prof_ref[valid] - prof_j[valid]) ** 2)))
            prof_err_max = float(np.max(np.abs(prof_ref[valid] - prof_j[valid])))
        else:
            prof_err_L2 = float("nan")
            prof_err_max = float("nan")

        print(
            f"center=({cy},{cx}) "
            f"t_ref={t_ref:5.3f}s t_j={t_j:5.3f}s "
            f"A_ref[min,max]=({A_ref.min():.4g},{A_ref.max():.4g}) "
            f"A_j[min,max]=({A_j.min():.4g},{A_j.max():.4g}) "
            f"axis_L1_ref={axis_l1_ref:.3e} axis_L1_j={axis_l1_j:.3e} "
            f"radial_L2={prof_err_L2:.3e} radial_max={prof_err_max:.3e}"
        )


def scenario_lens(size: int = 256, src_radius: int = 8, lens_radius: int = 20) -> None:
    print(f"\n=== Lens (Jacobian), size={size}, src_radius={src_radius}, lens_radius={lens_radius} ===")
    h = w = size
    src_center = (h // 2, w // 4)
    lens_center = (h // 2, 3 * w // 4)

    src = make_disk(src_center, src_radius, size)
    lens = make_disk(lens_center, lens_radius, size)

    n_field = np.ones((h, w), dtype=np.float32)
    n_field[lens] = 1.5

    phi = build_phi(size, src, n_field)

    t0 = time.time()
    A_ref = compute_amplitude_characteristic(phi, n_field, src, step_size=0.5)
    t_ref = time.time() - t0

    t1 = time.time()
    A_j = compute_amplitude_jacobian(
        phi,
        n_field,
        src,
        step_size=0.75,
        n_iters=40,
        smooth_sigma=0.5,
    )
    t_j = time.time() - t1

    print(
        f"lens: t_ref={t_ref:5.3f}s t_j={t_j:5.3f}s "
        f"A_ref[min,max]=({A_ref.min():.4g},{A_ref.max():.4g}) "
        f"A_j[min,max]=({A_j.min():.4g},{A_j.max():.4g}) "
        f"A_ref_src_mean={A_ref[src].mean():.4g} A_j_src_mean={A_j[src].mean():.4g} "
        f"A_ref_lens_mean={A_ref[lens].mean():.4g} A_j_lens_mean={A_j[lens].mean():.4g}"
    )


def main() -> None:
    scenario_homogeneous(size=256, radius=8)
    scenario_lens(size=256, src_radius=8, lens_radius=20)


if __name__ == "__main__":
    main()

