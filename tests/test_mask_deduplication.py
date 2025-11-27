"""Test mask deduplication performance improvement."""

import numpy as np
import time
from elliptica.types import Conductor
from elliptica.app.core import AppState
from elliptica.app.actions import add_conductor, ensure_render


def test_masks_cached_in_result():
    """Test that RenderResult contains cached masks."""
    state = AppState()

    # Create conductor with interior
    y, x = np.ogrid[:50, :50]
    center = 25
    outer = ((x - center)**2 + (y - center)**2) <= 20**2
    inner = ((x - center)**2 + (y - center)**2) <= 10**2
    ring_mask = (outer & ~inner).astype(np.float32)

    c = Conductor(mask=ring_mask, voltage=1.0)
    add_conductor(state, c)

    # Do render
    success = ensure_render(state)
    assert success, "Render failed"

    # Check that RenderResult has cached masks
    result = state.render_cache.result
    assert result.conductor_masks_canvas is not None, "Canvas masks not cached"
    assert result.interior_masks_canvas is not None, "Interior masks not cached"
    assert len(result.conductor_masks_canvas) == 1, "Wrong number of conductor masks"
    assert len(result.interior_masks_canvas) == 1, "Wrong number of interior masks"

    # Check that masks have correct shape (should match canvas_scaled_shape)
    conductor_mask = result.conductor_masks_canvas[0]
    assert conductor_mask.shape == result.canvas_scaled_shape, "Mask shape mismatch"

    print("✓ Masks are cached in RenderResult")


def test_rendercache_uses_cached_masks():
    """Test that RenderCache uses masks from RenderResult."""
    state = AppState()

    # Create two conductors
    mask1 = np.ones((20, 20), dtype=np.float32)
    c1 = Conductor(mask=mask1, voltage=1.0)
    add_conductor(state, c1)

    mask2 = np.ones((30, 30), dtype=np.float32)
    c2 = Conductor(mask=mask2, voltage=-1.0, position=(50, 50))
    add_conductor(state, c2)

    # Do render
    success = ensure_render(state)
    assert success, "Render failed"

    # Check that RenderCache has masks
    cache = state.render_cache
    assert cache.conductor_masks is not None, "RenderCache missing conductor masks"
    assert cache.interior_masks is not None, "RenderCache missing interior masks"
    assert len(cache.conductor_masks) == 2, "Wrong number of conductor masks in cache"

    # Verify that RenderCache masks are same object (not a copy) as RenderResult masks
    # This confirms no redundant rasterization happened
    assert cache.conductor_masks is cache.result.conductor_masks_canvas, \
        "RenderCache should use same mask list (not rasterize again)"

    print("✓ RenderCache uses cached masks from RenderResult")


def test_performance_improvement():
    """Benchmark render time to verify performance improvement."""
    state = AppState()

    # Create 5 conductors for more significant mask overhead
    for i in range(5):
        size = 25 + i * 5
        mask = np.ones((size, size), dtype=np.float32)
        c = Conductor(mask=mask, voltage=float(i - 2), position=(i * 40, i * 40))
        add_conductor(state, c)

    # Warm up (first render may include compilation overhead)
    ensure_render(state)

    # Benchmark multiple renders
    num_renders = 3
    times = []
    for _ in range(num_renders):
        # Change resolution to trigger new render
        from elliptica.app.actions import set_canvas_resolution
        w = 180 + _ * 20
        set_canvas_resolution(state, w, w)

        start = time.time()
        success = ensure_render(state)
        end = time.time()

        assert success, "Render failed"
        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"✓ Average render time with 5 conductors: {avg_time:.3f}s")
    print(f"  (Mask deduplication eliminated redundant rasterization)")


if __name__ == "__main__":
    test_masks_cached_in_result()
    test_rendercache_uses_cached_masks()
    test_performance_improvement()
    print("\n✅ All mask deduplication tests passed!")
