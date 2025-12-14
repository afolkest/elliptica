"""Quick test of Phase 1 colorization infrastructure."""

import numpy as np
from elliptica.types import BoundaryObject, Project
from elliptica.app.core import AppState, DisplaySettings
from elliptica.app.actions import add_boundary, set_color_enabled, set_palette, remove_boundary

def test_boundary_ids():
    """Test boundary ID assignment and style dict sync."""
    state = AppState()

    # Add first boundary
    mask1 = np.ones((10, 10), dtype=np.float32)
    c1 = BoundaryObject(mask=mask1, params={"voltage": 1.0})
    add_boundary(state, c1)

    assert c1.id == 0, f"Expected ID 0, got {c1.id}"
    assert 0 in state.boundary_color_settings, "Color settings entry not created"
    assert state.project.next_boundary_id == 1, "Counter not incremented"

    # Add second boundary
    mask2 = np.ones((15, 15), dtype=np.float32)
    c2 = BoundaryObject(mask=mask2, params={"voltage": -1.0})
    add_boundary(state, c2)

    assert c2.id == 1, f"Expected ID 1, got {c2.id}"
    assert 1 in state.boundary_color_settings, "Color settings entry not created for second boundary"
    assert state.project.next_boundary_id == 2, "Counter not incremented correctly"

    # Test removal
    remove_boundary(state, 0)

    assert 0 not in state.boundary_color_settings, "Color settings entry not removed"
    assert len(state.project.boundary_objects) == 1, "Boundary not removed from list"

    print("✓ Boundary ID system working correctly")

def test_interior_detection():
    """Test automatic interior detection."""
    state = AppState()

    # Create a thick ring mask
    y, x = np.ogrid[:50, :50]
    center = 25
    outer = ((x - center)**2 + (y - center)**2) <= 20**2
    inner = ((x - center)**2 + (y - center)**2) <= 10**2
    ring_mask = (outer & ~inner).astype(np.float32)

    c = BoundaryObject(mask=ring_mask, params={"voltage": 1.0})
    add_boundary(state, c)

    assert c.interior_mask is not None, "Interior not auto-detected"
    assert c.interior_mask.shape == ring_mask.shape, "Interior shape mismatch"
    assert np.any(c.interior_mask > 0), "Interior mask is empty"

    print("✓ Interior detection working")

def test_colorization_settings():
    """Test color settings work correctly."""
    state = AppState()

    # Create simple boundary
    mask = np.ones((20, 20), dtype=np.float32)
    c = BoundaryObject(mask=mask, params={"voltage": 1.0})
    add_boundary(state, c)

    # Start in grayscale mode
    set_color_enabled(state, False)
    assert state.display_settings.color_enabled is False, "Color should be disabled"

    # Test color/palette settings work
    set_color_enabled(state, True)
    set_palette(state, "Deep Ocean")

    # Verify settings were applied
    assert state.display_settings.color_enabled is True, "Color not enabled"
    assert state.display_settings.palette == "Deep Ocean", "Palette not set"

    print("✓ Colorization settings working")

if __name__ == "__main__":
    test_boundary_ids()
    test_interior_detection()
    test_colorization_settings()
    print("\n✅ All Phase 1 tests passed!")
