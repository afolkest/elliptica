"""Quick test of Phase 1 colorization infrastructure."""

import numpy as np
from flowcol.types import Conductor, Project
from flowcol.app.core import AppState, DisplaySettings
from flowcol.app.actions import add_conductor, set_color_enabled, set_palette, ensure_render, ensure_base_rgb

def test_conductor_ids():
    """Test conductor ID assignment and style dict sync."""
    state = AppState()

    # Add first conductor
    mask1 = np.ones((10, 10), dtype=np.float32)
    c1 = Conductor(mask=mask1, voltage=1.0)
    add_conductor(state, c1)

    assert c1.id == 0, f"Expected ID 0, got {c1.id}"
    assert 0 in state.conductor_styles, "Style entry not created"
    assert state.project.next_conductor_id == 1, "Counter not incremented"

    # Add second conductor
    mask2 = np.ones((15, 15), dtype=np.float32)
    c2 = Conductor(mask=mask2, voltage=-1.0)
    add_conductor(state, c2)

    assert c2.id == 1, f"Expected ID 1, got {c2.id}"
    assert 1 in state.conductor_styles, "Style entry not created for second conductor"
    assert state.project.next_conductor_id == 2, "Counter not incremented correctly"

    # Test removal
    from flowcol.app.actions import remove_conductor
    remove_conductor(state, 0)

    assert 0 not in state.conductor_styles, "Style entry not removed"
    assert len(state.project.conductors) == 1, "Conductor not removed from list"

    print("✓ Conductor ID system working correctly")

def test_interior_detection():
    """Test automatic interior detection."""
    state = AppState()

    # Create a thick ring mask
    y, x = np.ogrid[:50, :50]
    center = 25
    outer = ((x - center)**2 + (y - center)**2) <= 20**2
    inner = ((x - center)**2 + (y - center)**2) <= 10**2
    ring_mask = (outer & ~inner).astype(np.float32)

    c = Conductor(mask=ring_mask, voltage=1.0)
    add_conductor(state, c)

    assert c.interior_mask is not None, "Interior not auto-detected"
    assert c.interior_mask.shape == ring_mask.shape, "Interior shape mismatch"
    assert np.any(c.interior_mask > 0), "Interior mask is empty"

    print("✓ Interior detection working")

def test_colorization_settings():
    """Test color settings and cache invalidation."""
    state = AppState()

    # Create simple conductor
    mask = np.ones((20, 20), dtype=np.float32)
    c = Conductor(mask=mask, voltage=1.0)
    add_conductor(state, c)

    # Do a render
    success = ensure_render(state)
    assert success, "Render failed"

    cache = state.render_cache
    assert cache is not None, "No render cache created"
    assert cache.display_array is not None, "No display array"
    assert cache.base_rgb is None, "base_rgb should start None"
    assert cache.conductor_masks is not None, "No conductor masks generated"
    assert cache.interior_masks is not None, "No interior masks generated"

    # Build base RGB in grayscale mode
    success = ensure_base_rgb(state)
    assert success, "Failed to build base RGB"
    assert cache.base_rgb is not None, "base_rgb not created"
    assert cache.base_rgb.dtype == np.uint8, "base_rgb should be uint8"
    assert cache.base_rgb.ndim == 3, "base_rgb should be 3D (H,W,C)"

    grayscale_rgb = cache.base_rgb.copy()

    # Enable color and change palette
    set_color_enabled(state, True)
    assert cache.base_rgb is None, "base_rgb not invalidated on color enable"

    success = ensure_base_rgb(state)
    assert success, "Failed to rebuild base RGB"

    colored_rgb = cache.base_rgb.copy()

    # Should be different from grayscale
    assert not np.array_equal(grayscale_rgb, colored_rgb), "Color mode didn't change output"

    # Change palette
    set_palette(state, "Deep Ocean")
    assert cache.base_rgb is None, "base_rgb not invalidated on palette change"

    success = ensure_base_rgb(state)
    assert success, "Failed to rebuild with new palette"

    ocean_rgb = cache.base_rgb.copy()
    assert not np.array_equal(colored_rgb, ocean_rgb), "Palette change didn't affect output"

    print("✓ Colorization settings and invalidation working")

if __name__ == "__main__":
    test_conductor_ids()
    test_interior_detection()
    test_colorization_settings()
    print("\n✅ All Phase 1 tests passed!")
