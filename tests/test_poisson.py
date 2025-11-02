#!/usr/bin/env python3
"""
Test suite for the Poisson solver.
Tests multiple resolutions, boundary conditions, and nontrivial conductor shapes.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import flowcol
sys.path.insert(0, str(Path(__file__).parent.parent))
from flowcol.poisson import solve_poisson_system, DIRICHLET, NEUMANN
from flowcol.field import compute_field, _build_relaxation_mask, _relax_potential_band
from flowcol.types import Project, Conductor


def create_annulus_conductor(size, outer_radius=0.4, inner_radius=0.2, center=(0.5, 0.5)):
    """Create an annular (ring) conductor mask."""
    y, x = np.ogrid[0:size, 0:size]
    y = y / size - center[1]
    x = x / size - center[0]
    r = np.sqrt(x**2 + y**2)
    return (r <= outer_radius) & (r >= inner_radius)


def create_l_shape_conductor(size, thickness=0.1):
    """Create an L-shaped conductor mask."""
    mask = np.zeros((size, size), dtype=bool)
    # Vertical part
    x1 = int(size * 0.3)
    x2 = int(size * (0.3 + thickness))
    y1 = int(size * 0.2)
    y2 = int(size * 0.7)
    mask[y1:y2, x1:x2] = True

    # Horizontal part
    x3 = int(size * 0.3)
    x4 = int(size * 0.7)
    y3 = int(size * 0.6)
    y4 = int(size * (0.6 + thickness))
    mask[y3:y4, x3:x4] = True

    return mask


def create_parallel_plates(size, gap=0.3):
    """Create parallel plate conductors for validation."""
    mask = np.zeros((size, size), dtype=bool)
    left = int(size * 0.2)
    right = int(size * (0.2 + gap))
    mask[:, left] = True
    mask[:, right] = True
    return mask, left, right


def compute_electric_field(phi):
    """Compute E = -grad(phi) using central differences."""
    ey, ex = np.gradient(phi)
    return -ex, -ey


def test_resolution_convergence(resolutions=[64, 128, 256, 512], boundary_condition=DIRICHLET):
    """Test solver at multiple resolutions and measure convergence."""

    times = []
    solutions = []

    print(f"\nTesting resolution convergence with {['Dirichlet', 'Neumann'][boundary_condition]} BCs:")
    print("-" * 50)

    for res in resolutions:
        # Create annulus conductor
        conductor_mask = create_annulus_conductor(res)
        values = np.zeros((res, res))
        values[conductor_mask] = 1.0  # Set conductor to 1V

        # Time the solve
        start = time.time()
        phi = solve_poisson_system(
            conductor_mask,
            values,
            tol=1e-6,
            boundary_top=boundary_condition,
            boundary_bottom=boundary_condition,
            boundary_left=boundary_condition,
            boundary_right=boundary_condition
        )
        elapsed = time.time() - start
        times.append(elapsed)
        solutions.append(phi)

        # Print statistics
        print(f"Resolution: {res:4d}×{res:<4d} | Time: {elapsed:6.3f}s | "
              f"φ_max: {phi.max():6.4f} | φ_min: {phi.min():6.4f}")

    # Check convergence by comparing center point values
    if len(solutions) > 1:
        print("\nCenter point convergence:")
        for i, res in enumerate(resolutions):
            center_val = solutions[i][res//2, res//2]
            print(f"  {res:4d}: φ_center = {center_val:.6f}")

    return solutions, times




def test_l_shape():
    """Test with L-shaped conductor and create detailed visualization."""

    res = 256  # Reduced from 2048 for faster testing
    print(f"\nTesting L-shaped conductor at {res}×{res}:")
    print("-" * 50)

    # Create L-shape
    conductor_mask = create_l_shape_conductor(res)
    values = np.zeros((res, res))
    values[conductor_mask] = 1.0

    # Solve with both BCs
    start = time.time()
    phi_d = solve_poisson_system(conductor_mask, values,
                                  boundary_top=DIRICHLET, boundary_bottom=DIRICHLET,
                                  boundary_left=DIRICHLET, boundary_right=DIRICHLET)
    time_d = time.time() - start

    start = time.time()
    phi_n = solve_poisson_system(conductor_mask, values,
                                  boundary_top=NEUMANN, boundary_bottom=NEUMANN,
                                  boundary_left=NEUMANN, boundary_right=NEUMANN)
    time_n = time.time() - start

    print(f"Dirichlet: {time_d:.3f}s | φ ∈ [{phi_d.min():.4f}, {phi_d.max():.4f}]")
    print(f"Neumann:   {time_n:.3f}s | φ ∈ [{phi_n.min():.4f}, {phi_n.max():.4f}]")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, phi, title in zip(axes, [phi_d, phi_n], ['Dirichlet BC', 'Neumann BC']):
        im = ax.imshow(phi, cmap='plasma', origin='lower')

        # Add field lines using streamplot
        ex, ey = compute_electric_field(phi)
        # Downsample for streamplot to avoid issues
        step = 8  # Adjusted for 256 res
        y, x = np.mgrid[0:res:step, 0:res:step]
        ax.streamplot(x, y, ex[::step, ::step], ey[::step, ::step],
                     color='white', density=1.0,
                     linewidth=0.5, arrowsize=0.8)

        # Conductor outline
        ax.contour(conductor_mask, levels=[0.5], colors='cyan', linewidths=2)

        ax.set_title(f'L-Shaped Conductor: {title}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='Potential φ [V]')

    plt.tight_layout()
    return fig


def validate_parallel_plates():
    """Validate solver with parallel plates (known analytical solution)."""

    res = 128
    print(f"\nValidating with parallel plates at {res}×{res}:")
    print("-" * 50)

    mask, left, right = create_parallel_plates(res)
    values = np.zeros((res, res))
    values[:, left] = 0.0   # Left plate at 0V
    values[:, right] = 1.0  # Right plate at 1V

    phi = solve_poisson_system(mask, values,
                                boundary_top=DIRICHLET, boundary_bottom=DIRICHLET,
                                boundary_left=DIRICHLET, boundary_right=DIRICHLET)

    # Check if solution is approximately linear between plates
    mid_row = phi[res//2, left:right+1]
    expected = np.linspace(0, 1, right-left+1)
    error = np.abs(mid_row - expected).max()

    print(f"Maximum deviation from linear: {error:.6f}")
    print(f"✓ Parallel plates test {'PASSED' if error < 0.01 else 'FAILED'}")

    return error < 0.01


def test_compute_field_preview_scale_accuracy():
    """Ensure coarse Poisson preview stays close to full-resolution field."""

    project = Project(canvas_resolution=(64, 64))
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    conductor = Conductor(mask=mask, voltage=1.0, position=(24, 24))
    project.conductors.append(conductor)

    ex_full, ey_full = compute_field(project, poisson_scale=1.0)
    ex_preview, ey_preview = compute_field(project, poisson_scale=0.5)

    rms_error = np.sqrt(
        np.mean((ex_full - ex_preview) ** 2 + (ey_full - ey_preview) ** 2)
    )
    assert rms_error < 0.1, f"Preview field deviates too much (RMS={rms_error:.3f})"


def test_relaxation_preserves_dirichlet_and_modifies_band():
    """Relaxation should leave conductor values intact while smoothing the band."""

    size = 32
    phi = np.linspace(0, 1, size, dtype=np.float64)
    phi = np.tile(phi, (size, 1))
    dirichlet_mask = np.zeros((size, size), dtype=bool)
    dirichlet_mask[12:20, 12:20] = True

    phi[dirichlet_mask] = 1.0
    relax_mask = _build_relaxation_mask(dirichlet_mask, band_width=2)
    phi_before = phi.copy()

    if relax_mask.any():
        _relax_potential_band(phi, relax_mask, iterations=6, omega=0.8)

    # Dirichlet region should remain fixed
    assert np.allclose(phi[dirichlet_mask], phi_before[dirichlet_mask])
    # Smoothing should change at least some band pixels
    diff = np.abs(phi[relax_mask] - phi_before[relax_mask])
    assert np.any(diff > 1e-6)

def main():
    """Run all tests and generate visualizations."""

    print("=" * 60)
    print("POISSON SOLVER TEST SUITE")
    print("=" * 60)

    # Test resolutions for performance (using smaller sizes for quick testing)
    resolutions = [64, 128, 256]

    # Test with Dirichlet BCs
    dirichlet_sols, d_times = test_resolution_convergence(resolutions, DIRICHLET)

    # Test with Neumann BCs
    neumann_sols, n_times = test_resolution_convergence(resolutions, NEUMANN)

    # Validation test
    validate_parallel_plates()

    # L-shape test
    fig_l = test_l_shape()

    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Resolution':<12} {'Dirichlet [s]':<15} {'Neumann [s]':<15}")
    print("-" * 42)
    for i, res in enumerate(resolutions):
        print(f"{res}×{res:<6} {d_times[i]:<15.4f} {n_times[i]:<15.4f}")

    # Check if solutions are physically reasonable
    print("\n" + "=" * 60)
    print("PHYSICS CHECKS")
    print("=" * 60)

    all_passed = True
    for i, res in enumerate(resolutions):
        # Check that potential is bounded [0, 1] for unit conductor
        phi_d = dirichlet_sols[i]
        phi_n = neumann_sols[i]

        # Dirichlet should be strictly bounded [0, 1]
        d_bounds_ok = (-0.01 <= phi_d.min() <= phi_d.max() <= 1.01)
        # Neumann can float slightly outside [0, 1] due to floating boundary
        n_bounds_ok = (-0.1 <= phi_n.min() <= phi_n.max() <= 1.1)

        if not (d_bounds_ok and n_bounds_ok):
            all_passed = False
            print(f"✗ {res}×{res}: Potential out of bounds")
        else:
            print(f"✓ {res}×{res}: Potential bounds OK")

    if all_passed:
        print("\n✓ ALL PHYSICS CHECKS PASSED")
    else:
        print("\n✗ SOME PHYSICS CHECKS FAILED")

    plt.show()

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
