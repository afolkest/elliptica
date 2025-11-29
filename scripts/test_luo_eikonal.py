from __future__ import annotations

import time

import numpy as np

from elliptica.go_luo.factored_eikonal import solve_factored_eikonal


def make_constant_velocity_model(nx: int, nz: int, v0: float = 1.0) -> np.ndarray:
    return np.full((nx, nz), 1.0 / v0, dtype=np.float64)


def main() -> None:
    nx = nz = 201
    h = 1.0
    v0 = 2.0

    s = make_constant_velocity_model(nx, nz, v0=v0)
    source_idx = (nx // 2, nz // 2)

    t0 = time.time()
    res = solve_factored_eikonal(s, h, source_idx, v_ref=v0, max_sweeps=40)
    elapsed = time.time() - t0

    ix0, iz0 = source_idx
    xs = (np.arange(nx) - ix0) * h
    zs = (np.arange(nz) - iz0) * h
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    r = np.sqrt(X * X + Z * Z)
    tau_analytic = r / v0

    err = res.tau - tau_analytic
    print(f"nx=nz={nx}, v0={v0}, h={h}")
    print(f"elapsed={elapsed:.3f}s")
    print(
        f"tau error: max={np.max(np.abs(err)):.3e}, "
        f"rms={np.sqrt(np.mean(err**2)):.3e}"
    )


if __name__ == "__main__":
    main()
