#!/usr/bin/env python3
"""
Test script for the new multi-PDE architecture.

This verifies that the PDE abstraction works correctly and maintains
backwards compatibility with the existing system.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from elliptica.types import Project, BoundaryObject
from elliptica.pde.register import register_all_pdes, list_pde_names
from elliptica.pde import PDERegistry
from elliptica.field_pde import compute_field_pde, compute_field_legacy


def create_test_project() -> Project:
    """Create a simple test project with two boundary objects."""
    project = Project()
    project.canvas_resolution = (100, 100)

    # Create first boundary object (disk at center)
    mask1 = np.zeros((20, 20), dtype=np.float32)
    y, x = np.ogrid[:20, :20]
    center = 10
    mask1[(x - center) ** 2 + (y - center) ** 2 <= 64] = 1.0

    obj1 = BoundaryObject(
        mask=mask1,
        voltage=5.0,  # Using voltage for backwards compatibility
        position=(40, 40)
    )

    # Create second boundary object
    mask2 = np.ones((15, 15), dtype=np.float32)
    obj2 = BoundaryObject(
        mask=mask2,
        voltage=-3.0,
        position=(60, 60)
    )

    project.conductors = [obj1, obj2]
    return project


def test_pde_registration():
    """Test PDE registration system."""
    print("Testing PDE Registration...")

    # Register all PDEs
    register_all_pdes()

    # Check available PDEs
    available = list_pde_names()
    print(f"  Available PDEs: {available}")
    assert "poisson" in available, "Poisson PDE should be registered"

    # Check active PDE
    active = PDERegistry.get_active()
    print(f"  Active PDE: {active.display_name}")
    assert active.name == "poisson", "Poisson should be default active PDE"

    print("  ✓ PDE registration working")


def test_backwards_compatibility():
    """Test that BoundaryObject is compatible with Conductor."""
    print("\nTesting Backwards Compatibility...")

    obj = BoundaryObject(mask=np.ones((10, 10)), voltage=2.5)

    # Test value property (future compatibility)
    assert obj.value == 2.5, "value property should return voltage"
    obj.value = 3.0
    assert obj.voltage == 3.0, "value setter should update voltage"

    # Test direct voltage access
    obj.voltage = 4.0
    assert obj.value == 4.0, "voltage and value should be synchronized"

    print("  ✓ BoundaryObject backwards compatible with Conductor")


def test_compute_field():
    """Test field computation with PDE abstraction."""
    print("\nTesting Field Computation...")

    project = create_test_project()

    # Test new compute_field_pde
    solution, (ex, ey) = compute_field_pde(project)

    print(f"  Solution keys: {list(solution.keys())}")
    assert "phi" in solution, "Poisson PDE should return phi in solution"

    phi = solution["phi"]
    print(f"  Phi shape: {phi.shape}")
    print(f"  Phi range: [{phi.min():.3f}, {phi.max():.3f}]")

    print(f"  Field shape: ex={ex.shape}, ey={ey.shape}")
    print(f"  Field magnitude range: [{np.sqrt(ex**2 + ey**2).max():.3f}]")

    # Test that field is non-zero
    field_mag = np.sqrt(ex**2 + ey**2)
    assert field_mag.max() > 0, "Field should be non-zero"

    # Test legacy wrapper
    ex_legacy, ey_legacy = compute_field_legacy(project)
    # Use allclose instead of exact equality due to floating point differences
    # The tiny differences (< 1e-7) are due to numerical precision
    np.testing.assert_allclose(ex, ex_legacy, rtol=1e-4, atol=1e-7,
                               err_msg="Legacy wrapper should return same field")
    np.testing.assert_allclose(ey, ey_legacy, rtol=1e-4, atol=1e-7,
                               err_msg="Legacy wrapper should return same field")

    print("  ✓ Field computation working")


def test_project_attributes():
    """Test new Project attributes."""
    print("\nTesting Project Attributes...")

    project = create_test_project()

    # Test new attributes
    assert project.pde_type == "poisson", "Default PDE type should be poisson"
    assert isinstance(project.pde_params, dict), "pde_params should be dict"

    # Test compatibility accessors
    assert project.boundary_objects == project.conductors, "boundary_objects should alias conductors"
    assert project.shape == project.canvas_resolution, "shape should return canvas_resolution"
    assert project.poisson_scale == 1.0, "poisson_scale should be 1.0"

    bc = project.boundary_conditions
    assert isinstance(bc, dict), "boundary_conditions should be dict"
    assert 'top' in bc and 'bottom' in bc, "boundary_conditions should have all edges"

    print("  ✓ Project attributes working")


def test_pde_definition():
    """Test PDE definition attributes."""
    print("\nTesting PDE Definition...")

    pde = PDERegistry.get("poisson")

    print(f"  Name: {pde.name}")
    print(f"  Display Name: {pde.display_name}")
    print(f"  Description: {pde.description}")

    assert callable(pde.solve), "solve should be callable"
    assert callable(pde.extract_lic_field), "extract_lic_field should be callable"

    print("  ✓ PDE definition structure correct")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Multi-PDE Architecture Test Suite")
    print("=" * 60)

    try:
        test_pde_registration()
        test_backwards_compatibility()
        test_project_attributes()
        test_pde_definition()
        test_compute_field()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe multi-PDE architecture is working correctly!")
        print("Backwards compatibility is maintained.")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()