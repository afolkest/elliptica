"""Test per-region colorization features."""

import numpy as np
from flowcol.types import Conductor
from flowcol.app.core import AppState
from flowcol.app.actions import (
    add_conductor,
    set_region_style_enabled,
    set_region_palette,
    set_region_solid_color,
    ensure_render,
    ensure_base_rgb,
)
from flowcol.postprocess.color import apply_region_overlays


def test_region_style_settings():
    """Test region style enable/disable and palette/color settings."""
    state = AppState()

    # Create conductor with interior
    y, x = np.ogrid[:50, :50]
    center = 25
    outer = ((x - center)**2 + (y - center)**2) <= 20**2
    inner = ((x - center)**2 + (y - center)**2) <= 10**2
    ring_mask = (outer & ~inner).astype(np.float32)

    c = Conductor(mask=ring_mask, voltage=1.0)
    add_conductor(state, c)

    assert c.id == 0
    settings = state.conductor_color_settings[c.id]

    # Check defaults
    assert not settings.surface.enabled, "Surface should start disabled"
    assert not settings.interior.enabled, "Interior should start disabled"
    assert settings.surface.use_palette, "Surface should default to palette mode"
    assert settings.interior.use_palette, "Interior should default to palette mode"

    # Enable surface with custom palette
    set_region_style_enabled(state, c.id, "surface", True)
    assert settings.surface.enabled, "Surface not enabled"

    set_region_palette(state, c.id, "surface", "Deep Ocean")
    assert settings.surface.use_palette, "Surface should be in palette mode"
    assert settings.surface.palette == "Deep Ocean", "Palette not set correctly"

    # Enable interior with solid color
    set_region_style_enabled(state, c.id, "interior", True)
    assert settings.interior.enabled, "Interior not enabled"

    set_region_solid_color(state, c.id, "interior", (1.0, 0.0, 0.0))
    assert not settings.interior.use_palette, "Interior should be in solid color mode"
    assert settings.interior.solid_color == (1.0, 0.0, 0.0), "Color not set correctly"

    print("✓ Region style settings working correctly")


def test_multi_conductor_regions():
    """Test multiple conductors with different region styles."""
    state = AppState()

    # Conductor 1: Small solid circle
    y, x = np.ogrid[:30, :30]
    mask1 = ((x - 15)**2 + (y - 15)**2 <= 10**2).astype(np.float32)
    c1 = Conductor(mask=mask1, voltage=1.0, position=(10, 10))
    add_conductor(state, c1)

    # Conductor 2: Ring with interior
    y, x = np.ogrid[:60, :60]
    outer = ((x - 30)**2 + (y - 30)**2) <= 25**2
    inner = ((x - 30)**2 + (y - 30)**2) <= 15**2
    mask2 = (outer & ~inner).astype(np.float32)
    c2 = Conductor(mask=mask2, voltage=-1.0, position=(100, 100))
    add_conductor(state, c2)

    # Set different styles for each conductor
    set_region_style_enabled(state, c1.id, "surface", True)
    set_region_palette(state, c1.id, "surface", "Fire")

    set_region_style_enabled(state, c2.id, "surface", True)
    set_region_palette(state, c2.id, "surface", "Ice")

    set_region_style_enabled(state, c2.id, "interior", True)
    set_region_solid_color(state, c2.id, "interior", (0.0, 1.0, 0.0))

    # Verify settings
    s1 = state.conductor_color_settings[c1.id]
    s2 = state.conductor_color_settings[c2.id]

    assert s1.surface.enabled and s1.surface.palette == "Fire"
    assert not s1.interior.enabled  # c1 has no interior

    assert s2.surface.enabled and s2.surface.palette == "Ice"
    assert s2.interior.enabled and s2.interior.solid_color == (0.0, 1.0, 0.0)

    print("✓ Multiple conductor region styles working")


def test_compositor_integration():
    """Test that compositor runs without errors."""
    state = AppState()

    # Create two conductors
    mask1 = np.ones((20, 20), dtype=np.float32)
    c1 = Conductor(mask=mask1, voltage=1.0)
    add_conductor(state, c1)

    y, x = np.ogrid[:40, :40]
    outer = ((x - 20)**2 + (y - 20)**2) <= 15**2
    inner = ((x - 20)**2 + (y - 20)**2) <= 8**2
    mask2 = (outer & ~inner).astype(np.float32)
    c2 = Conductor(mask=mask2, voltage=-1.0, position=(50, 50))
    add_conductor(state, c2)

    # Do render
    success = ensure_render(state)
    assert success, "Render failed"

    cache = state.render_cache
    assert cache is not None
    assert cache.result is not None
    assert cache.result.array is not None
    assert cache.conductor_masks is not None
    assert cache.interior_masks is not None

    # Build base RGB
    success = ensure_base_rgb(state)
    assert success, "Failed to build base RGB"
    assert cache.base_rgb is not None

    # Enable some region styles
    set_region_style_enabled(state, c1.id, "surface", True)
    set_region_palette(state, c1.id, "surface", "Fire")

    set_region_style_enabled(state, c2.id, "interior", True)
    set_region_solid_color(state, c2.id, "interior", (0.5, 0.5, 1.0))

    # Apply compositor (use full-res result.array)
    final_rgb = apply_region_overlays(
        cache.base_rgb,
        cache.result.array,
        cache.conductor_masks,
        cache.interior_masks,
        state.conductor_color_settings,
        state.project.conductors,
        state.display_settings.to_color_params(),
    )

    assert final_rgb is not None
    assert final_rgb.shape == cache.base_rgb.shape
    assert final_rgb.dtype == np.uint8

    # Final RGB should be different from base RGB where regions are customized
    assert not np.array_equal(final_rgb, cache.base_rgb), \
        "Compositor should change output when regions are styled"

    print("✓ Compositor integration working")


def test_region_style_isolation():
    """Test that region styles are independent per conductor."""
    state = AppState()

    # Add three conductors
    for i in range(3):
        mask = np.ones((15, 15), dtype=np.float32) * 0.8
        c = Conductor(mask=mask, voltage=float(i))
        add_conductor(state, c)

    conductors = state.project.conductors

    # Set different styles for each
    set_region_style_enabled(state, conductors[0].id, "surface", True)
    set_region_palette(state, conductors[0].id, "surface", "Fire")

    set_region_style_enabled(state, conductors[1].id, "surface", True)
    set_region_palette(state, conductors[1].id, "surface", "Ice")

    # Leave conductor 2 unstyled

    # Verify isolation
    s0 = state.conductor_color_settings[conductors[0].id]
    s1 = state.conductor_color_settings[conductors[1].id]
    s2 = state.conductor_color_settings[conductors[2].id]

    assert s0.surface.enabled and s0.surface.palette == "Fire"
    assert s1.surface.enabled and s1.surface.palette == "Ice"
    assert not s2.surface.enabled  # Should remain disabled

    # Change conductor 0 should not affect conductor 1
    set_region_solid_color(state, conductors[0].id, "surface", (1.0, 0.0, 0.0))
    assert not s0.surface.use_palette  # Changed to solid
    assert s1.surface.use_palette  # Should remain palette mode

    print("✓ Region style isolation working")


if __name__ == "__main__":
    test_region_style_settings()
    test_multi_conductor_regions()
    test_compositor_integration()
    test_region_style_isolation()
    print("\n✅ All per-region colorization tests passed!")
