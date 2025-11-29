#!/usr/bin/env python
"""
Ad-hoc test harness for compute_amplitude_characteristic_multires.

Run from repo root after activating the venv:

    source venv/bin/activate
    PYTHONPATH=. python scripts/test_eikonal_amp_multires.py

This compares the multi-resolution characteristic wrapper against the
full-resolution characteristic reference for a few scenarios and prints
error and runtime metrics.
"""

from __future__ import annotations

import time
from typing import Tuple, Sequence

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt

from elliptica.pde.eikonal_amp import (
    compute_amplitude_characteristic,
    compute_amplitude_characteristic_multires,
)


def build_phi(
    size: int,
    src_mask: np.ndarray,
    n_field: np.ndarray,
) -> np.ndarray:
    """Solve eikonal |∇φ| = n with skfmm for a given source mask and n."""
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


def scenario_homogeneous(
    size: int = 256,
    radius: int = 8,
    subsamples: Sequence[int] = (2, 3),
) -> None:
    print(f"\n=== Homogeneous disk (multires characteristic), size={size}, radius={radius} ===")
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

        prof_ref = radial_profile(A_ref, cy, cx)

        print(
            f"center=({cy},{cx}) "
            f"t_ref={t_ref:5.3f}s "
            f"A_ref[min,max]=({A_ref.min():.4g},{A_ref.max():.4g})"
        )

        for ds in subsamples:
            t1 = time.time()
            A_ds = compute_amplitude_characteristic_multires(
                phi,
                n_field,
                src,
                downsample=ds,
                step_size=0.5,
            )
            t_ds = time.time() - t1

            # Axis symmetry through the source.
            y_mid = A_ds[:, cx]
            x_mid = A_ds[cy, :]
            axis_l1 = float(np.mean(np.abs(y_mid - x_mid)))

            # Radial profile error vs reference.
            prof_ds = radial_profile(A_ds, cy, cx)
            valid = np.isfinite(prof_ref) & np.isfinite(prof_ds)
            if valid.any():
                prof_err_L2 = float(np.sqrt(np.mean((prof_ref[valid] - prof_ds[valid]) ** 2)))
                prof_err_max = float(np.max(np.abs(prof_ref[valid] - prof_ds[valid])))
            else:
                prof_err_L2 = float("nan")
                prof_err_max = float("nan")

            print(
                f"  ds={ds} t_ds={t_ds:5.3f}s "
                f"A_ds[min,max]=({A_ds.min():.4g},{A_ds.max():.4g}) "
                f"axis_L1={axis_l1:.3e} "
                f"radial_L2={prof_err_L2:.3e} radial_max={prof_err_max:.3e}"
            )


def scenario_lens(
    size: int = 256,
    src_radius: int = 8,
    lens_radius: int = 20,
    subsamples: Sequence[int] = (2, 3),
) -> None:
    print(f"\n=== Lens (multires characteristic), size={size}, src_radius={src_radius}, lens_radius={lens_radius} ===")
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

    print(
        f"lens: t_ref={t_ref:5.3f}s "
        f"A_ref[min,max]=({A_ref.min():.4g},{A_ref.max():.4g}) "
        f"A_ref_src_mean={A_ref[src].mean():.4g} "
        f"A_ref_lens_mean={A_ref[lens].mean():.4g}"
    )

    for ds in subsamples:
        t1 = time.time()
        A_ds = compute_amplitude_characteristic_multires(
            phi,
            n_field,
            src,
            downsample=ds,
            step_size=0.5,
        )
        t_ds = time.time() - t1

        print(
            f"  ds={ds} t_ds={t_ds:5.3f}s "
            f"A_ds[min,max]=({A_ds.min():.4g},{A_ds.max():.4g}) "
            f"A_ds_src_mean={A_ds[src].mean():.4g} "
            f"A_ds_lens_mean={A_ds[lens].mean():.4g}"
        )


def main() -> None:
    scenario_homogeneous(size=256, radius=8, subsamples=(2, 3))
    scenario_lens(size=256, src_radius=8, lens_radius=20, subsamples=(2, 3))


if __name__ == "__main__":
    main()

