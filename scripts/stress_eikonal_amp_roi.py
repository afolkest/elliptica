#!/usr/bin/env python
"""
Stress tests for ROI and (homogeneous) multires characteristic amplitude.

Run from repo root after activating the venv:

    source venv/bin/activate
    PYTHONPATH=. python scripts/stress_eikonal_amp_roi.py

This exercises a set of scenes (homogeneous disks, lenses, multi-lens,
slab lenses) and compares:

  - Full-resolution characteristic amplitude solve
  - ROI-cropped characteristic solve (amplitude only)

Metrics printed:
  - Runtime for full vs ROI
  - ROI size vs full size
  - L2 / max error inside the ROI
  - Simple symmetry / radial diagnostics in homogeneous cases
"""

from __future__ import annotations

import time
from typing import Tuple, Sequence

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt

from elliptica.pde.eikonal_amp import compute_amplitude_characteristic


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


def solve_full_and_roi(
    phi: np.ndarray,
    n_field: np.ndarray,
    src_mask: np.ndarray,
    roi_margin: int = 64,
    step_size: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None, float, float]:
    """
    Compute full-resolution and ROI-cropped characteristic amplitude.

    ROI is defined as a bounding box around sources and lens regions
    (where n deviates from the background), dilated by roi_margin.

    Returns:
        A_full: Full amplitude field from characteristic solver.
        A_roi: Amplitude field with ROI cropping (A=1 outside ROI).
        bbox: ROI bounding box (y0, y1, x0, x1) or None if no crop.
        t_full: Runtime of full solve.
        t_roi: Runtime of ROI solve (including cropping/embedding).
    """
    phi = np.asarray(phi)
    n_field = np.asarray(n_field)
    src_mask = np.asarray(src_mask, dtype=bool)

    h, w = phi.shape

    # Full solve
    t0 = time.perf_counter()
    A_full = compute_amplitude_characteristic(phi, n_field, src_mask, step_size=step_size)
    t_full = time.perf_counter() - t0

    # Background refractive index (exclude sources to avoid picking their n if modified).
    non_src = ~src_mask
    if non_src.any():
        n_bg = float(np.median(n_field[non_src]))
    else:
        n_bg = float(np.median(n_field))
    lens_mask = np.abs(n_field - n_bg) > 1e-4

    roi_core = src_mask | lens_mask
    if not roi_core.any():
        # No lenses and no sources (should not happen in practice); ROI=full.
        return A_full, A_full.copy(), None, t_full, 0.0

    if roi_margin > 0:
        dist = distance_transform_edt(~roi_core)
        roi_mask = dist <= roi_margin
    else:
        roi_mask = roi_core

    rows = np.where(roi_mask.any(axis=1))[0]
    cols = np.where(roi_mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return A_full, A_full.copy(), None, t_full, 0.0

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    # Crop phi, n, source to ROI.
    phi_local = phi[y0:y1, x0:x1]
    n_local = n_field[y0:y1, x0:x1]
    src_local = src_mask[y0:y1, x0:x1]

    t1 = time.perf_counter()
    A_local = compute_amplitude_characteristic(phi_local, n_local, src_local, step_size=step_size)
    t_roi = time.perf_counter() - t1

    # Embed back into a full field; use A=1 outside ROI as a far-field approximation.
    A_roi = np.ones((h, w), dtype=np.float32)
    A_roi[y0:y1, x0:x1] = A_local.astype(np.float32, copy=False)
    A_roi[src_mask] = 1.0

    return A_full, A_roi, (y0, y1, x0, x1), t_full, t_roi


def scenario_homogeneous_offcenter(size: int = 256, radius: int = 8, roi_margin: int = 64) -> None:
    print(f"\n=== Homogeneous off-center disks (ROI), size={size}, radius={radius} ===")
    h = w = size
    centers = [
        (h // 2, w // 2),
        (h // 4, w // 4),
        (3 * h // 4, 3 * w // 4),
        (h // 2, w // 8),
        (h // 2, 7 * w // 8),
    ]
    for cy, cx in centers:
        src = make_disk((cy, cx), radius, size)
        n_field = np.ones((h, w), dtype=np.float32)
        phi = build_phi(size, src, n_field)

        A_full, A_roi, bbox, t_full, t_roi = solve_full_and_roi(phi, n_field, src, roi_margin=roi_margin)

        # Axis symmetry (through source center) for full and ROI.
        y_mid_full = A_full[:, cx]
        x_mid_full = A_full[cy, :]
        axis_l1_full = float(np.mean(np.abs(y_mid_full - x_mid_full)))

        y_mid_roi = A_roi[:, cx]
        x_mid_roi = A_roi[cy, :]
        axis_l1_roi = float(np.mean(np.abs(y_mid_roi - x_mid_roi)))

        # Radial profiles.
        prof_full = radial_profile(A_full, cy, cx)
        prof_roi = radial_profile(A_roi, cy, cx)
        valid = np.isfinite(prof_full) & np.isfinite(prof_roi)
        if valid.any():
            prof_err_L2 = float(np.sqrt(np.mean((prof_full[valid] - prof_roi[valid]) ** 2)))
            prof_err_max = float(np.max(np.abs(prof_full[valid] - prof_roi[valid])))
        else:
            prof_err_L2 = float("nan")
            prof_err_max = float("nan")

        roi_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) if bbox is not None else h * w
        print(
            f"center=({cy},{cx}) "
            f"t_full={t_full:5.3f}s t_roi={t_roi:5.3f}s "
            f"roi_bbox={bbox} roi_area={roi_area}/{h*w} "
            f"axis_L1_full={axis_l1_full:.3e} axis_L1_roi={axis_l1_roi:.3e} "
            f"radial_L2={prof_err_L2:.3e} radial_max={prof_err_max:.3e}"
        )


def scenario_single_lens(size: int = 256, src_radius: int = 8, lens_radius: int = 20, roi_margin: int = 64) -> None:
    print(f"\n=== Single lens (ROI), size={size}, src_radius={src_radius}, lens_radius={lens_radius} ===")
    h = w = size
    src_center = (h // 2, w // 4)
    lens_center = (h // 2, 3 * w // 4)

    src = make_disk(src_center, src_radius, size)
    lens = make_disk(lens_center, lens_radius, size)

    n_field = np.ones((h, w), dtype=np.float32)
    n_field[lens] = 1.5

    phi = build_phi(size, src, n_field)

    A_full, A_roi, bbox, t_full, t_roi = solve_full_and_roi(phi, n_field, src, roi_margin=roi_margin)

    lens_region = lens
    down_region = (np.arange(w)[None, :] > lens_center[1]) & (np.arange(h)[:, None] == lens_center[0])

    def region_stats(name: str, mask: np.ndarray, A0: np.ndarray, A1: np.ndarray) -> str:
        if not mask.any():
            return f"{name}: empty"
        mean0 = float(A0[mask].mean())
        mean1 = float(A1[mask].mean())
        l2 = float(np.sqrt(np.mean((A0[mask] - A1[mask]) ** 2)))
        mx = float(np.max(np.abs(A0[mask] - A1[mask])))
        return f"{name}: mean_full={mean0:.4g} mean_roi={mean1:.4g} L2={l2:.3e} max={mx:.3e}"

    roi_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) if bbox is not None else h * w

    print(
        f"t_full={t_full:5.3f}s t_roi={t_roi:5.3f}s "
        f"roi_bbox={bbox} roi_area={roi_area}/{h*w} "
        f"A_full[min,max]=({A_full.min():.4g},{A_full.max():.4g}) "
        f"A_roi[min,max]=({A_roi.min():.4g},{A_roi.max():.4g})"
    )
    print("  ", region_stats("source", src, A_full, A_roi))
    print("  ", region_stats("lens", lens_region, A_full, A_roi))
    print("  ", region_stats("downstream_axis", down_region, A_full, A_roi))


def scenario_multi_lens(size: int = 256, src_radius: int = 8, lens_radius: int = 16, roi_margin: int = 64) -> None:
    print(f"\n=== Multi-lens (ROI), size={size}, src_radius={src_radius}, lens_radius={lens_radius} ===")
    h = w = size
    src_center = (h // 2, w // 8)
    lens_centers = [
        (h // 3, 3 * w // 8),
        (2 * h // 3, 5 * w // 8),
        (h // 2, 7 * w // 8),
    ]

    src = make_disk(src_center, src_radius, size)
    lens_masks = [make_disk(c, lens_radius, size) for c in lens_centers]
    lens_union = np.zeros((h, w), dtype=bool)
    for lm in lens_masks:
        lens_union |= lm

    n_field = np.ones((h, w), dtype=np.float32)
    for lm in lens_masks:
        n_field[lm] = 1.5

    phi = build_phi(size, src, n_field)

    A_full, A_roi, bbox, t_full, t_roi = solve_full_and_roi(phi, n_field, src, roi_margin=roi_margin)

    roi_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) if bbox is not None else h * w
    l2_lens = float(np.sqrt(np.mean((A_full[lens_union] - A_roi[lens_union]) ** 2)))
    max_lens = float(np.max(np.abs(A_full[lens_union] - A_roi[lens_union])))

    print(
        f"t_full={t_full:5.3f}s t_roi={t_roi:5.3f}s "
        f"roi_bbox={bbox} roi_area={roi_area}/{h*w} "
        f"A_full[min,max]=({A_full.min():.4g},{A_full.max():.4g}) "
        f"A_roi[min,max]=({A_roi.min():.4g},{A_roi.max():.4g}) "
        f"L2_lens={l2_lens:.3e} max_lens={max_lens:.3e}"
    )


def scenario_slab_lens(size: int = 256, src_radius: int = 8, roi_margin: int = 64) -> None:
    print(f"\n=== Slab lens (ROI), size={size}, src_radius={src_radius} ===")
    h = w = size
    src_center = (h // 2, w // 8)
    src = make_disk(src_center, src_radius, size)

    # Vertical slab in the middle of the domain.
    slab = np.zeros((h, w), dtype=bool)
    slab[:, w // 2 - w // 16 : w // 2 + w // 16] = True

    n_field = np.ones((h, w), dtype=np.float32)
    n_field[slab] = 1.5

    phi = build_phi(size, src, n_field)

    A_full, A_roi, bbox, t_full, t_roi = solve_full_and_roi(phi, n_field, src, roi_margin=roi_margin)

    roi_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) if bbox is not None else h * w
    slab_region = slab

    l2_slab = float(np.sqrt(np.mean((A_full[slab_region] - A_roi[slab_region]) ** 2)))
    max_slab = float(np.max(np.abs(A_full[slab_region] - A_roi[slab_region])))

    print(
        f"t_full={t_full:5.3f}s t_roi={t_roi:5.3f}s "
        f"roi_bbox={bbox} roi_area={roi_area}/{h*w} "
        f"A_full[min,max]=({A_full.min():.4g},{A_full.max():.4g}) "
        f"A_roi[min,max]=({A_roi.min():.4g},{A_roi.max():.4g}) "
        f"L2_slab={l2_slab:.3e} max_slab={max_slab:.3e}"
    )


def main() -> None:
    size = 256
    roi_margin = 64
    scenario_homogeneous_offcenter(size=size, radius=8, roi_margin=roi_margin)
    scenario_single_lens(size=size, src_radius=8, lens_radius=20, roi_margin=roi_margin)
    scenario_multi_lens(size=size, src_radius=8, lens_radius=16, roi_margin=roi_margin)
    scenario_slab_lens(size=size, src_radius=8, roi_margin=roi_margin)


if __name__ == "__main__":
    main()

