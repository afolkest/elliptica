"""CGL + LIC visualization of different vector quantities."""

import numpy as np
import numba
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# ============================================================
# CGL Solver (same as before)
# ============================================================
@numba.njit(cache=True, parallel=True)
def cgl_step(psi_r, psi_i, out_r, out_i, mask, dt, b, c):
    height, width = psi_r.shape
    for i in numba.prange(1, height - 1):
        for j in range(1, width - 1):
            if not mask[i, j]:
                out_r[i, j] = 0.0
                out_i[i, j] = 0.0
                continue
            lap_r = psi_r[i+1,j] + psi_r[i-1,j] + psi_r[i,j+1] + psi_r[i,j-1] - 4*psi_r[i,j]
            lap_i = psi_i[i+1,j] + psi_i[i-1,j] + psi_i[i,j+1] + psi_i[i,j-1] - 4*psi_i[i,j]
            diff_r = lap_r - b * lap_i
            diff_i = lap_i + b * lap_r
            abs_sq = psi_r[i,j]**2 + psi_i[i,j]**2
            nonlin_r = abs_sq * (psi_r[i,j] - c * psi_i[i,j])
            nonlin_i = abs_sq * (psi_i[i,j] + c * psi_r[i,j])
            dpsi_r = psi_r[i,j] + diff_r - nonlin_r
            dpsi_i = psi_i[i,j] + diff_i - nonlin_i
            out_r[i,j] = psi_r[i,j] + dt * dpsi_r
            out_i[i,j] = psi_i[i,j] + dt * dpsi_i

def solve_cgl(mask, n_steps=5000, dt=0.05, b=1.0, c=1.0, seed=42):
    N = mask.shape[0]
    np.random.seed(seed)
    psi_r = np.random.randn(N, N) * 0.1
    psi_i = np.random.randn(N, N) * 0.1
    psi_r[~mask] = 0
    psi_i[~mask] = 0
    out_r, out_i = np.zeros_like(psi_r), np.zeros_like(psi_i)
    for step in range(n_steps):
        cgl_step(psi_r, psi_i, out_r, out_i, mask, dt, b, c)
        psi_r, out_r = out_r, psi_r
        psi_i, out_i = out_i, psi_i
    return psi_r + 1j * psi_i

# ============================================================
# Vector field extraction
# ============================================================
def compute_gradient(field):
    """Compute gradient of a 2D field."""
    gy, gx = np.gradient(field)
    return gx, gy

def compute_supercurrent(psi):
    """Compute supercurrent j = Im(ψ* ∇ψ)."""
    psi_conj = np.conj(psi)
    dpsi_dy, dpsi_dx = np.gradient(psi)
    j_x = np.imag(psi_conj * dpsi_dx)
    j_y = np.imag(psi_conj * dpsi_dy)
    return j_x, j_y

def compute_phase_gradient(psi):
    """Compute gradient of phase (with care for branch cuts)."""
    phase = np.angle(psi)
    # Use complex gradient to handle wrapping
    gy, gx = np.gradient(np.exp(1j * phase))
    # Extract phase gradient from d(e^iθ) = i*e^iθ * dθ
    gx_phase = np.imag(gx * np.exp(-1j * phase))
    gy_phase = np.imag(gy * np.exp(-1j * phase))
    return gx_phase, gy_phase

def compute_magnitude_gradient(psi):
    """Compute gradient of |ψ|."""
    mag = np.abs(psi)
    gy, gx = np.gradient(mag)
    return gx, gy

# ============================================================
# Simple LIC implementation
# ============================================================
@numba.njit(cache=True)
def lic_kernel(vx, vy, noise, length=20):
    """Simple LIC: integrate streamlines through noise texture."""
    h, w = vx.shape
    result = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            # Forward integration
            x, y = float(j), float(i)
            total = noise[i, j]
            count = 1.0

            for _ in range(length):
                ix, iy = int(x + 0.5), int(y + 0.5)
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    break
                dx, dy = vx[iy, ix], vy[iy, ix]
                mag = np.sqrt(dx*dx + dy*dy) + 1e-10
                dx, dy = dx/mag, dy/mag
                x += dx * 0.5
                y += dy * 0.5
                ix, iy = int(x + 0.5), int(y + 0.5)
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    break
                total += noise[iy, ix]
                count += 1.0

            # Backward integration
            x, y = float(j), float(i)
            for _ in range(length):
                ix, iy = int(x + 0.5), int(y + 0.5)
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    break
                dx, dy = vx[iy, ix], vy[iy, ix]
                mag = np.sqrt(dx*dx + dy*dy) + 1e-10
                dx, dy = dx/mag, dy/mag
                x -= dx * 0.5
                y -= dy * 0.5
                ix, iy = int(x + 0.5), int(y + 0.5)
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    break
                total += noise[iy, ix]
                count += 1.0

            result[i, j] = total / count

    return result

def apply_lic(vx, vy, length=25, seed=123):
    """Apply LIC to a vector field."""
    np.random.seed(seed)
    noise = np.random.rand(*vx.shape).astype(np.float64)
    return lic_kernel(vx.astype(np.float64), vy.astype(np.float64), noise, length)

def complex_to_hsv(psi, mask=None):
    mag = np.abs(psi)
    phase = np.angle(psi)
    mag_norm = np.clip(mag / (np.percentile(mag[mag > 0], 95) + 1e-10), 0, 1)
    hue = (phase + np.pi) / (2 * np.pi)
    hsv = np.zeros((*psi.shape, 3))
    hsv[..., 0] = hue
    hsv[..., 1] = 0.85
    hsv[..., 2] = mag_norm
    rgb = hsv_to_rgb(hsv)
    if mask is not None:
        rgb[~mask] = 0.15
    return rgb

# ============================================================
# Main visualization
# ============================================================
N = 400  # Smaller for faster LIC
print(f"Grid: {N}x{N}")
y, x = np.ogrid[:N, :N]

# Create mask with obstacles
mask = np.ones((N, N), dtype=bool)
outer = ((x - N//2)**2 + (y - N//2)**2) > (N//2.2)**2
mask[outer] = False
for cx, cy, r in [(150, 150, 30), (280, 130, 25), (210, 300, 35)]:
    mask[((x - cx)**2 + (y - cy)**2) < r**2] = False

# Solve CGL
print("Solving CGL...")
psi = solve_cgl(mask, n_steps=6000, dt=0.02, b=1.0, c=-0.8, seed=42)

# Compute vector fields
print("Computing vector fields...")
jx, jy = compute_supercurrent(psi)
gx_phase, gy_phase = compute_phase_gradient(psi)
gx_mag, gy_mag = compute_magnitude_gradient(psi)

# Zero out vectors outside mask
jx[~mask], jy[~mask] = 0, 0
gx_phase[~mask], gy_phase[~mask] = 0, 0
gx_mag[~mask], gy_mag[~mask] = 0, 0

# Apply LIC
print("Computing LIC (supercurrent)...")
lic_supercurrent = apply_lic(jx, jy, length=30)

print("Computing LIC (phase gradient)...")
lic_phase = apply_lic(gx_phase, gy_phase, length=30)

print("Computing LIC (magnitude gradient)...")
lic_mag = apply_lic(gx_mag, gy_mag, length=30)

print("Computing LIC (perpendicular to supercurrent)...")
lic_perp = apply_lic(-jy, jx, length=30)  # Rotate 90 degrees

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Raw field and basic LICs
axes[0, 0].imshow(complex_to_hsv(psi, mask))
axes[0, 0].set_title("CGL solution (phase=hue)")
axes[0, 0].axis('off')

axes[0, 1].imshow(lic_supercurrent, cmap='gray')
axes[0, 1].set_title("LIC of supercurrent Im(ψ*∇ψ)")
axes[0, 1].axis('off')

axes[0, 2].imshow(lic_phase, cmap='gray')
axes[0, 2].set_title("LIC of phase gradient ∇θ")
axes[0, 2].axis('off')

# Row 2: More LICs and composites
axes[1, 0].imshow(lic_mag, cmap='gray')
axes[1, 0].set_title("LIC of magnitude gradient ∇|ψ|")
axes[1, 0].axis('off')

axes[1, 1].imshow(lic_perp, cmap='gray')
axes[1, 1].set_title("LIC perpendicular to supercurrent")
axes[1, 1].axis('off')

# Composite: LIC + phase coloring
hsv_img = complex_to_hsv(psi, mask)
lic_norm = (lic_supercurrent - lic_supercurrent.min()) / (lic_supercurrent.max() - lic_supercurrent.min() + 1e-10)
composite = hsv_img * lic_norm[:, :, np.newaxis]
composite[~mask] = 0.15
axes[1, 2].imshow(composite)
axes[1, 2].set_title("Composite: supercurrent LIC × phase color")
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('gl_solutions.png', dpi=150)
print("\nSaved to gl_solutions.png")
