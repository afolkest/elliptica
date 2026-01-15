"""Compare scipy.convolve vs numba for GL at 2k."""

import time
import numpy as np
import numba
from scipy.ndimage import convolve

N = 2048
print(f"Grid: {N} x {N}\n")

# Create mask
y, x = np.ogrid[:N, :N]
mask = np.ones((N, N), dtype=bool)
np.random.seed(42)
for _ in range(5):
    cx, cy = np.random.randint(N//4, 3*N//4, 2)
    r = np.random.randint(N//10, N//5)
    mask[(x - cx)**2 + (y - cy)**2 < r**2] = False

# ============================================================
# Method 1: scipy.ndimage.convolve (current approach)
# ============================================================
lap_stencil = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=np.float64)

psi = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) * 0.1
psi[~mask] = 0

def gl_step_scipy(psi, mask, dt=0.1):
    lap_r = convolve(psi.real, lap_stencil, mode='constant')
    lap_i = convolve(psi.imag, lap_stencil, mode='constant')
    lap_psi = lap_r + 1j * lap_i
    abs_sq = np.abs(psi)**2
    psi_new = psi + dt * (lap_psi + psi - abs_sq * psi)
    psi_new[~mask] = 0
    return psi_new

print("scipy.ndimage.convolve:")
t0 = time.perf_counter()
for _ in range(20):
    psi = gl_step_scipy(psi, mask)
t = time.perf_counter() - t0
print(f"  20 steps: {t:.2f}s ({t/20*1000:.0f}ms/step)")
print(f"  Est 500 steps: {t*25:.0f}s")

# ============================================================
# Method 2: Numba fused kernel
# ============================================================
@numba.njit(cache=True, parallel=True)
def gl_step_numba(psi_r, psi_i, out_r, out_i, mask, dt):
    height, width = psi_r.shape
    for i in numba.prange(1, height - 1):
        for j in range(1, width - 1):
            if not mask[i, j]:
                out_r[i, j] = 0.0
                out_i[i, j] = 0.0
                continue

            # Laplacian
            lap_r = (psi_r[i+1, j] + psi_r[i-1, j] +
                     psi_r[i, j+1] + psi_r[i, j-1] - 4*psi_r[i, j])
            lap_i = (psi_i[i+1, j] + psi_i[i-1, j] +
                     psi_i[i, j+1] + psi_i[i, j-1] - 4*psi_i[i, j])

            # |psi|^2
            abs_sq = psi_r[i, j]**2 + psi_i[i, j]**2

            # GL update: psi + dt*(lap + psi - |psi|^2 * psi)
            out_r[i, j] = psi_r[i, j] + dt * (lap_r + psi_r[i, j] - abs_sq * psi_r[i, j])
            out_i[i, j] = psi_i[i, j] + dt * (lap_i + psi_i[i, j] - abs_sq * psi_i[i, j])

# Initialize
psi_r = np.random.randn(N, N) * 0.1
psi_i = np.random.randn(N, N) * 0.1
psi_r[~mask] = 0
psi_i[~mask] = 0
out_r = np.zeros_like(psi_r)
out_i = np.zeros_like(psi_i)

# Warmup (JIT compile)
print("\nNumba (compiling...):")
gl_step_numba(psi_r, psi_i, out_r, out_i, mask, 0.1)

# Benchmark
psi_r = np.random.randn(N, N) * 0.1
psi_i = np.random.randn(N, N) * 0.1
psi_r[~mask] = 0
psi_i[~mask] = 0

t0 = time.perf_counter()
for _ in range(20):
    gl_step_numba(psi_r, psi_i, out_r, out_i, mask, 0.1)
    psi_r, out_r = out_r, psi_r
    psi_i, out_i = out_i, psi_i
t = time.perf_counter() - t0
print(f"  20 steps: {t:.2f}s ({t/20*1000:.0f}ms/step)")
print(f"  Est 500 steps: {t*25:.0f}s")

print("\n" + "="*60)
print(f"Speedup: {(2.36/t*20):.1f}x")  # Compare to scipy baseline
