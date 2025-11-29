#!/usr/bin/env python
"""
Sanity check: 1D piecewise-constant slowness against analytic solution.

We embed a 1D profile s(x) in a 2D grid (no y-variation) and compare the
factored solver tau to the exact 1D integral of s(x) from the source.
"""

from __future__ import annotations

import numpy as np

from elliptica.go_luo.factored_eikonal import solve_factored_eikonal


def analytic_phi_1d(s_profile: np.ndarray, h: float, src_idx: int) -> np.ndarray:
    """Exact 1D travel time phi[i] = ∫ s dx from source to i."""
    n = s_profile.size
    phi = np.zeros(n, dtype=float)
    # integrate left of source
    for i in range(src_idx - 1, -1, -1):
        phi[i] = phi[i + 1] + 0.5 * (s_profile[i] + s_profile[i + 1]) * h
    # integrate right of source
    for i in range(src_idx + 1, n):
        phi[i] = phi[i - 1] + 0.5 * (s_profile[i] + s_profile[i - 1]) * h
    return phi


def run_case(n: int = 201, s_left: float = 1.0, s_right: float = 1.5) -> None:
    h = 1.0
    src_idx = n // 2
    s_profile = np.ones(n, dtype=float) * s_left
    s_profile[src_idx + 1 :] = s_right

    # Embed in 2D
    s_field = np.tile(s_profile[:, None], (1, n))
    res = solve_factored_eikonal(
        s_field,
        h,
        (src_idx, n // 2),
        v_ref=1.0 / s_left,
        max_sweeps=80,
        smooth_sigma=0.5,
    )
    tau = res.tau[:, n // 2]  # take center column

    phi_exact = analytic_phi_1d(s_profile, h, src_idx)

    err = tau - phi_exact
    rms = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))
    print(
        f"1D slab n={n}, s_left={s_left}, s_right={s_right} "
        f"tau[min,max]=({tau.min():.3g},{tau.max():.3g}) "
        f"phi_exact[max]={phi_exact.max():.3g} "
        f"err_rms={rms:.3e} err_max={max_abs:.3e}"
    )


def main() -> None:
    run_case()


if __name__ == "__main__":
    main()
