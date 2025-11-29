#!/usr/bin/env python
"""
Ad-hoc test harness for compute_amplitude_flux (grid-based eikonal amplitude).

Run from repo root after activating the venv:

    source venv/bin/activate
    PYTHONPATH=. python scripts/test_eikonal_amp_flux.py

This does not assert or write images; it prints symmetry and runtime metrics
for a few representative scenarios.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt

from elliptica.pde.eikonal_amp import compute_amplitude_flux


def build_phi_for_source(
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


def summarize_homogeneous(size: int = 256, radius: int = 8, n_iters: int = 80) -> None:
    print(f"\n=== Homogeneous n=1, size={size}, radius={radius}, n_iters={n_iters} ===")
    h = w = size
    centers = [
        (h // 2, w // 2),
        (h // 4, w // 4),
        (3 * h // 4, 3 * w // 4),
        (h // 2, w // 8),
        (h // 8, 3 * w // 4),
    ]
    for cy, cx in centers:
        src = make_disk((cy, cx), radius, size)
        n_field = np.ones((h, w), dtype=np.float32)
        phi = build_phi_for_source(size, src, n_field)

        t0 = time.time()
        A = compute_amplitude_flux(phi, n_field, src, n_iters=n_iters)
        dt = time.time() - t0

        y_mid = A[:, cx]
        x_mid = A[cy, :]
        axis_l1 = float(np.mean(np.abs(y_mid - x_mid)))

        print(
            f"center=({cy},{cx}) time={dt:5.3f}s "
            f"A[min,max]=({A.min():.4g},{A.max():.4g}) axis_L1={axis_l1:.3e}"
        )


def summarize_lens(size: int = 256, src_radius: int = 8, lens_radius: int = 20, n_iters: int = 80) -> None:
    print(f"\n=== Piecewise constant lens, size={size}, n_iters={n_iters} ===")
    h = w = size

    # Source in the center, lens off to the right.
    src_center = (h // 2, w // 4)
    lens_center = (h // 2, 3 * w // 4)

    src = make_disk(src_center, src_radius, size)
    lens = make_disk(lens_center, lens_radius, size)

    n_field = np.ones((h, w), dtype=np.float32)
    n_field[lens] = 1.5

    phi = build_phi_for_source(size, src, n_field)

    t0 = time.time()
    A = compute_amplitude_flux(phi, n_field, src, n_iters=n_iters)
    dt = time.time() - t0

    print(
        f"lens test: time={dt:5.3f}s "
        f"A[min,max]=({A.min():.4g},{A.max():.4g}) "
        f"A_src_mean={A[src].mean():.4g} A_lens_mean={A[lens].mean():.4g}"
    )


def summarize_multisource(size: int = 256, radius: int = 8, n_iters: int = 80) -> None:
    print(f"\n=== Multiple sources, size={size}, n_iters={n_iters} ===")
    h = w = size
    centers = [
        (h // 3, w // 3),
        (2 * h // 3, w // 3),
        (h // 2, 2 * w // 3),
    ]
    src = np.zeros((h, w), dtype=bool)
    for cy, cx in centers:
        src |= make_disk((cy, cx), radius, size)

    n_field = np.ones((h, w), dtype=np.float32)
    phi = build_phi_for_source(size, src, n_field)

    t0 = time.time()
    A = compute_amplitude_flux(phi, n_field, src, n_iters=n_iters)
    dt = time.time() - t0

    print(
        f"multisource: time={dt:5.3f}s "
        f"A[min,max]=({A.min():.4g},{A.max():.4g}) "
        f"A_src_mean={A[src].mean():.4g}"
    )


def main() -> None:
    summarize_homogeneous(size=256, radius=8, n_iters=80)
    summarize_lens(size=256, src_radius=8, lens_radius=20, n_iters=80)
    summarize_multisource(size=256, radius=8, n_iters=80)


if __name__ == "__main__":
    main()
