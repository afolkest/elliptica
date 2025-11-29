#!/usr/bin/env python
"""
Profile the torch-based characteristic amplitude solver on CPU vs MPS.

Run from repo root after activating the venv:
    source venv/bin/activate
    PYTHONPATH=. python scripts/profile_eikonal_amp_torch.py
"""

from __future__ import annotations

import time
import argparse

import numpy as np
import skfmm
import torch
from scipy.ndimage import distance_transform_edt

from elliptica.pde.eikonal_amp import compute_amplitude_characteristic_torch


def build_test_problem(size: int = 512, radius: int = 12):
    h = w = size
    cy, cx = h // 2, w // 2
    Y, X = np.mgrid[0:h, 0:w]
    r = np.hypot(Y - cy, X - cx)
    src = r <= radius

    # Homogeneous n=1
    n_field = np.ones((h, w), dtype=np.float32)

    # Eikonal solve (Skfmm) for |∇φ| = n
    phi_init = distance_transform_edt(~src).astype(float)
    phi_init[src] = -distance_transform_edt(src)[src]
    phi = skfmm.travel_time(phi_init, np.ones_like(phi_init), order=2)
    phi[src] = 0.0

    return phi.astype(np.float32), n_field, src


def profile_device(device: str, size: int, step_size: float) -> None:
    phi, n_field, src = build_test_problem(size=size)

    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("device mps not available on this system")
            return

    t0 = time.time()
    A = compute_amplitude_characteristic_torch(
        phi,
        n_field,
        src,
        step_size=step_size,
        device=device,
    )
    dt = time.time() - t0

    center = A[size // 2, size // 2]
    # Sample an annulus around the source to check symmetry
    Y, X = np.mgrid[0:size, 0:size]
    r = np.hypot(Y - size // 2, X - size // 2)
    ring_mask = (r > 20) & (r < 22)
    ring_vals = A[ring_mask]
    ring_mean = float(ring_vals.mean()) if ring_vals.size > 0 else float("nan")
    ring_std = float(ring_vals.std()) if ring_vals.size > 0 else float("nan")

    print(
        f"device={device:>3} size={size:4d} "
        f"time={dt:6.3f}s "
        f"A[min,max]=({A.min():.4g},{A.max():.4g}) "
        f"A_center={center:.4g} ring_mean={ring_mean:.4g} ring_std={ring_std:.4g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile torch characteristic amplitude solver (CPU vs MPS)."
    )
    parser.add_argument("--size", type=int, default=512, help="Grid size (NxN)")
    parser.add_argument("--step", type=float, default=0.75, help="Backtracking step size in pixels")
    args = parser.parse_args()

    for dev in ("cpu", "mps"):
        profile_device(dev, size=args.size, step_size=args.step)


if __name__ == "__main__":
    main()

