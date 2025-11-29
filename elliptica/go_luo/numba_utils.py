from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is optional at runtime
    njit = lambda *args, **kwargs: (  # type: ignore[assignment]
        (lambda f: f) if (not args or callable(args[0])) else (lambda f: f)
    )


@njit(cache=True)
def weno3_1d_minus(u: np.ndarray, i: int, j: int, axis: int, h: float) -> float:
    """
    Third‑order upwind‑biased WENO approximation of du/dx (\"minus\" stencil).

    Parameters
    ----------
    u:
        2D array of values.
    i, j:
        Index of the evaluation point.
    axis:
        0 for derivative along the first axis, 1 for the second.
    h:
        Grid spacing.
    """
    if axis == 0:
        im2 = max(i - 2, 0)
        im1 = max(i - 1, 0)
        ip1 = min(i + 1, u.shape[0] - 1)
        ui = u[i, j]
        uim1 = u[im1, j]
        uim2 = u[im2, j]
        uip1 = u[ip1, j]
    else:
        jm2 = max(j - 2, 0)
        jm1 = max(j - 1, 0)
        jp1 = min(j + 1, u.shape[1] - 1)
        ui = u[i, j]
        ujm1 = u[i, jm1]
        ujm2 = u[i, jm2]
        ujp1 = u[i, jp1]

    eps = 1e-6

    if axis == 0:
        d0 = (ui - 2.0 * uim1 + uim2) ** 2
        d1 = (uip1 - 2.0 * ui + uim1) ** 2
        gamma = (eps + d0) / (eps + d1)
        omega = 1.0 / (1.0 + 2.0 * gamma * gamma)
        cen = (uip1 - uim1) * 0.5 / h
        upw = (3.0 * ui - 4.0 * uim1 + uim2) * 0.5 / h
    else:
        d0 = (ui - 2.0 * ujm1 + ujm2) ** 2
        d1 = (ujp1 - 2.0 * ui + ujm1) ** 2
        gamma = (eps + d0) / (eps + d1)
        omega = 1.0 / (1.0 + 2.0 * gamma * gamma)
        cen = (ujp1 - ujm1) * 0.5 / h
        upw = (3.0 * ui - 4.0 * ujm1 + ujm2) * 0.5 / h

    return (1.0 - omega) * cen + omega * upw


@njit(cache=True)
def weno3_1d_plus(u: np.ndarray, i: int, j: int, axis: int, h: float) -> float:
    """
    Third‑order upwind‑biased WENO approximation of du/dx (\"plus\" stencil).
    """
    if axis == 0:
        im1 = max(i - 1, 0)
        ip1 = min(i + 1, u.shape[0] - 1)
        ip2 = min(i + 2, u.shape[0] - 1)
        ui = u[i, j]
        uim1 = u[im1, j]
        uip1 = u[ip1, j]
        uip2 = u[ip2, j]
    else:
        jm1 = max(j - 1, 0)
        jp1 = min(j + 1, u.shape[1] - 1)
        jp2 = min(j + 2, u.shape[1] - 1)
        ui = u[i, j]
        ujm1 = u[i, jm1]
        ujp1 = u[i, jp1]
        ujp2 = u[i, jp2]

    eps = 1e-6

    if axis == 0:
        d0 = (ui - 2.0 * uip1 + uip2) ** 2
        d1 = (uip1 - 2.0 * ui + uim1) ** 2
        gamma = (eps + d0) / (eps + d1)
        omega = 1.0 / (1.0 + 2.0 * gamma * gamma)
        cen = (uip1 - uim1) * 0.5 / h
        upw = (-3.0 * ui + 4.0 * uip1 - uip2) * 0.5 / h
    else:
        d0 = (ui - 2.0 * ujp1 + ujp2) ** 2
        d1 = (ujp1 - 2.0 * ui + ujm1) ** 2
        gamma = (eps + d0) / (eps + d1)
        omega = 1.0 / (1.0 + 2.0 * gamma * gamma)
        cen = (ujp1 - ujm1) * 0.5 / h
        upw = (-3.0 * ui + 4.0 * ujp1 - ujp2) * 0.5 / h

    return (1.0 - omega) * cen + omega * upw


@njit(cache=True)
def lf_update_scalar(
    u: np.ndarray,
    s: np.ndarray,
    tau0: np.ndarray,
    s0: np.ndarray,
    h: float,
    i: int,
    j: int,
    ix_src: int,
    iz_src: int,
) -> float:
    """
    Third-order WENO + Lax–Friedrichs update for factored eikonal τ=τ0*u.

    This follows the structure in Luo–Qian (2011): use a monotone LF
    Hamiltonian with upwind-biased WENO3 approximations of ux, uz inserted
    via the virtual-neighbour construction.
    """
    # Fix source point to u=1 to avoid singularity at τ0=0.
    if i == ix_src and j == iz_src:
        return 1.0

    # WENO directional derivatives of u
    ux_minus = weno3_1d_minus(u, i, j, 0, h)
    ux_plus = weno3_1d_plus(u, i, j, 0, h)
    uz_minus = weno3_1d_minus(u, i, j, 1, h)
    uz_plus = weno3_1d_plus(u, i, j, 1, h)

    # Symmetric combination for Hamiltonian evaluation
    ux_hat = 0.5 * (ux_minus + ux_plus)
    uz_hat = 0.5 * (uz_minus + uz_plus)

    # τ0 gradients (central)
    im1 = max(i - 1, 0)
    ip1 = min(i + 1, u.shape[0] - 1)
    jm1 = max(j - 1, 0)
    jp1 = min(j + 1, u.shape[1] - 1)
    t0x = (tau0[ip1, j] - tau0[im1, j]) * 0.5 / h
    t0z = (tau0[i, jp1] - tau0[i, jm1]) * 0.5 / h

    tau0_ij = tau0[i, j]
    u_ij = u[i, j]

    # Hamiltonian H = sqrt(...) - s
    a = tau0_ij * tau0_ij * (ux_hat * ux_hat + uz_hat * uz_hat)
    b = 2.0 * tau0_ij * u_ij * (t0x * ux_hat + t0z * uz_hat)
    c = u_ij * u_ij * s0[i, j] * s0[i, j]
    rad = a + b + c
    if rad < 0.0:
        rad = 0.0
    val = np.sqrt(rad)
    H = val - s[i, j]

    # Derivatives for LF diffusion coefficients
    denom = val if val > 1e-8 else 1e-8
    Hp_x = tau0_ij * (t0x * u_ij + tau0_ij * ux_hat) / denom
    Hp_z = tau0_ij * (t0z * u_ij + tau0_ij * uz_hat) / denom
    Hu = (s0[i, j] * s0[i, j] * u_ij + tau0_ij * (t0x * ux_hat + t0z * uz_hat)) / denom

    alpha_x = 0.5 * abs(Hp_x) + abs(Hu)
    alpha_z = 0.5 * abs(Hp_z) + abs(Hu)
    # Avoid degenerate denominator
    denom_alpha = alpha_x / h + alpha_z / h + 1e-12

    # Virtual neighbours using upwind derivatives
    u_im1 = u_ij - h * ux_minus
    u_ip1 = u_ij + h * ux_plus
    u_jm1 = u_ij - h * uz_minus
    u_jp1 = u_ij + h * uz_plus

    u_new = (
        -H
        + alpha_x * (u_ip1 + u_im1) / (2.0 * h)
        + alpha_z * (u_jp1 + u_jm1) / (2.0 * h)
    ) / denom_alpha

    if not np.isfinite(u_new):
        return u_ij

    # Enforce positivity and prevent runaway growth during iteration.
    if u_new < 1e-6:
        u_new = 1e-6
    if u_new > 1e6:
        u_new = 1e6

    return u_new


@njit(cache=True)
def sweep_factored_eikonal(
    u: np.ndarray,
    s: np.ndarray,
    tau0: np.ndarray,
    s0: np.ndarray,
    h: float,
    order: int,
    ix_src: int,
    iz_src: int,
    fixed_mask: np.ndarray | None = None,
    gauss_seidel: bool = True,
    u_min: float = 1e-6,
    u_max: float = 1e6,
) -> None:
    """
    Perform one Gauss–Seidel sweep over the grid in a given order.

    Parameters
    ----------
    order:
        0..3 selecting the sweep direction (↘, ↗, ↙, ↖).
    """
    nx, nz = u.shape

    if order == 0:  # i+, j+
        irange = range(nx)
        jrange = range(nz)
    elif order == 1:  # i-, j+
        irange = range(nx - 1, -1, -1)
        jrange = range(nz)
    elif order == 2:  # i+, j-
        irange = range(nx)
        jrange = range(nz - 1, -1, -1)
    else:  # order == 3: i-, j-
        irange = range(nx - 1, -1, -1)
        jrange = range(nz - 1, -1, -1)

    if gauss_seidel:
        u_old = u  # alias; will be read from updated u
    else:
        u_old = u.copy()

    for ii in irange:
        for jj in jrange:
            # Skip physical boundary; treat as outflow to reduce spurious decay.
            if ii == 0 or jj == 0 or ii == nx - 1 or jj == nz - 1:
                continue
            if fixed_mask is not None and fixed_mask[ii, jj]:
                continue
            u[ii, jj] = lf_update_scalar(u_old, s, tau0, s0, h, ii, jj, ix_src, iz_src)
            if u[ii, jj] < u_min:
                u[ii, jj] = u_min
            if u[ii, jj] > u_max:
                u[ii, jj] = u_max
