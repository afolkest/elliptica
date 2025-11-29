from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .numba_utils import sweep_factored_eikonal


@dataclass
class FactoredEikonalResult:
    tau: np.ndarray
    u: np.ndarray
    tau0: np.ndarray
    s0: np.ndarray


def _analytic_tau0_s0(
    x: np.ndarray,
    z: np.ndarray,
    source: Tuple[float, float],
    v0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytic reference traveltime and slowness for constant velocity v0.

    This corresponds to τ0 = r / v0 with r the Euclidean distance to the
    point source, and s0 = 1 / v0.
    """
    x0, z0 = source
    dx = x - x0
    dz = z - z0
    r = np.sqrt(dx * dx + dz * dz)
    # Avoid division by zero at the source; τ0→0 there.
    tau0 = r / v0
    s0 = np.full_like(tau0, 1.0 / v0)
    return tau0, s0


def solve_factored_eikonal(
    s: np.ndarray,
    h: float,
    source_idx: Tuple[int, int],
    v_ref: float | None = None,
    max_sweeps: int = 40,
    tol: float = 1e-4,
    gauss_seidel: bool = True,
    u_min: float | None = None,
    u_max: float | None = None,
    smooth_sigma: float = 0.0,
    interface_minmod: bool = False,
) -> FactoredEikonalResult:
    """
    Solve the factored eikonal equation τ = τ0 * u on a regular grid.

    Parameters
    ----------
    s:
        Slowness field s(x,z) = 1/v(x,z), shape (nx, nz).
    h:
        Grid spacing (assumed equal in x and z for now).
    source_idx:
        Grid index (ix, iz) of the point source.
    v_ref:
        Reference constant velocity v0 used for factorisation. If None,
        uses the velocity at the source.
    max_sweeps:
        Maximum number of four‑direction sweep cycles.
    tol:
        Convergence tolerance on u (max‑norm of change per sweep cycle).
    """
    nx, nz = s.shape
    ix0, iz0 = source_idx

    # Optional smoothing of slowness to soften sharp jumps (helps interface artefacts).
    if smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter

        s_sm = gaussian_filter(s, smooth_sigma, mode="nearest")
    else:
        s_sm = s

    v = 1.0 / (s_sm + 1e-12)
    if v_ref is None:
        v_ref = float(v[ix0, iz0])

    xs = (np.arange(nx) - ix0) * h
    zs = (np.arange(nz) - iz0) * h
    X, Z = np.meshgrid(xs, zs, indexing="ij")

    tau0, s0 = _analytic_tau0_s0(X, Z, (0.0, 0.0), v_ref)

    # Initialize u to s/s0 (exact for homogeneous regions), then fix the source patch
    # and domain boundaries to act as outflow/Dirichlet equal to s/s0.
    u = (s_sm / s0).astype(np.float64, copy=True)

    # If not provided, set bounds for u based on slowness extremes.
    if u_min is None:
        u_min = float(np.min(s_sm / s0))
    if u_max is None:
        u_max = float(np.max(s_sm / s0))
    # Keep bounds reasonable
    u_min = max(u_min, 1e-6)
    u_max = max(u_max, 1.0)

    alpha_cap = max(np.max(s_sm), np.max(s0)) * 2.0
    # Disable interface-specific limiting by default (set to False mask).
    interface_mask = np.zeros_like(s, dtype=np.bool_)

    # Fixed mask: source patch + domain boundary (Dirichlet u=s/s0).
    patch = np.zeros_like(u, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii = ix0 + di
            jj = iz0 + dj
            if 0 <= ii < nx and 0 <= jj < nz:
                patch[ii, jj] = True
                u[ii, jj] = s[ii, jj] / s0[ii, jj]

    # Always fix the source value itself.
    u[ix0, iz0] = 1.0
    patch[ix0, iz0] = True

    # Fix domain boundaries to initial values to prevent drift at edges.
    patch[0, :] = True
    patch[-1, :] = True
    patch[:, 0] = True
    patch[:, -1] = True

    for sweep in range(max_sweeps):
        u_cycle_start = u.copy()
        for direction in range(4):
            sweep_factored_eikonal(
                u,
                s,
                tau0,
                s0,
                h,
                direction,
                ix0,
                iz0,
                patch,
                gauss_seidel=gauss_seidel,
                u_min=u_min,
                u_max=u_max,
                alpha_cap=alpha_cap,
                interface_mask=interface_mask,
                use_interface_minmod=interface_minmod,
            )
        # Clamp u globally to the assumed monotonicity range.
        np.clip(u, u_min, u_max, out=u)
        diff = np.max(np.abs(u - u_cycle_start))
        if diff < tol:
            break

    tau = tau0 * u
    return FactoredEikonalResult(tau=tau, u=u, tau0=tau0, s0=s0)
