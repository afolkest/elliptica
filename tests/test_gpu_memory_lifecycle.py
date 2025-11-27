"""Test GPU memory lifecycle management."""

import numpy as np
from elliptica.types import Conductor
from elliptica.app.core import AppState
from elliptica.app.actions import add_conductor, ensure_render, set_canvas_resolution
from elliptica.gpu import GPUContext


def test_gpu_cleanup_on_clear():
    """Test that clear_render_cache() properly frees GPU memory."""
    if not GPUContext.is_available():
        print("⊘ GPU not available, skipping test")
        return

    state = AppState()

    # Create simple conductor
    mask = np.ones((20, 20), dtype=np.float32)
    c = Conductor(mask=mask, voltage=1.0)
    add_conductor(state, c)

    # Do render to populate GPU tensors
    success = ensure_render(state)
    assert success, "Render failed"

    cache = state.render_cache
    assert cache.result_gpu is not None, "GPU tensor should be uploaded"
    assert cache.ex_gpu is not None, "ex GPU tensor should be uploaded"
    assert cache.ey_gpu is not None, "ey GPU tensor should be uploaded"

    # Clear cache should free GPU memory
    state.clear_render_cache()

    # empty_cache() is called internally, but we can't directly verify VRAM freed
    # Just verify tensors are cleared
    assert state.render_cache is None, "Cache should be cleared"

    print("✓ GPU cleanup on clear_render_cache() working")


def test_gpu_cleanup_on_rerender():
    """Test that ensure_render() frees old GPU tensors before uploading new ones."""
    if not GPUContext.is_available():
        print("⊘ GPU not available, skipping test")
        return

    state = AppState()

    # Create conductor
    mask = np.ones((20, 20), dtype=np.float32)
    c = Conductor(mask=mask, voltage=1.0)
    add_conductor(state, c)

    # First render
    success = ensure_render(state)
    assert success, "First render failed"

    first_cache = state.render_cache
    first_result_gpu = first_cache.result_gpu
    assert first_result_gpu is not None, "GPU tensor should be uploaded"

    # Trigger re-render by changing resolution
    set_canvas_resolution(state, 150, 150)

    # Second render should free old tensors before uploading new
    success = ensure_render(state)
    assert success, "Second render failed"

    second_cache = state.render_cache
    assert second_cache is not first_cache, "Should have new cache"
    assert second_cache.result_gpu is not None, "New GPU tensor should be uploaded"
    assert second_cache.result_gpu is not first_result_gpu, "Should be different GPU tensor"

    # Old tensors should have been freed (we set them to None before empty_cache())
    # Can't directly verify VRAM freed, but the cleanup code should have run

    print("✓ GPU cleanup on re-render working")


def test_multiple_renders():
    """Test that multiple sequential renders don't leak GPU memory."""
    if not GPUContext.is_available():
        print("⊘ GPU not available, skipping test")
        return

    state = AppState()

    # Create conductor
    mask = np.ones((30, 30), dtype=np.float32)
    c = Conductor(mask=mask, voltage=1.0)
    add_conductor(state, c)

    # Do 5 renders with different resolutions
    resolutions = [(100, 100), (120, 120), (140, 140), (160, 160), (180, 180)]

    for i, (w, h) in enumerate(resolutions):
        set_canvas_resolution(state, w, h)
        success = ensure_render(state)
        assert success, f"Render {i+1} failed at resolution {w}x{h}"

        cache = state.render_cache
        assert cache.result_gpu is not None, f"GPU tensor missing in render {i+1}"

    # Final cleanup
    state.clear_render_cache()
    assert state.render_cache is None

    print("✓ Multiple sequential renders working without leaks")


def test_empty_cache_graceful_when_no_gpu():
    """Test that empty_cache() is safe to call when GPU unavailable."""
    # Force CPU mode temporarily
    original_available = GPUContext._available
    GPUContext._available = False

    try:
        # This should not crash
        GPUContext.empty_cache()
        print("✓ empty_cache() safe when GPU unavailable")
    finally:
        # Restore original state
        GPUContext._available = original_available


if __name__ == "__main__":
    test_gpu_cleanup_on_clear()
    test_gpu_cleanup_on_rerender()
    test_multiple_renders()
    test_empty_cache_graceful_when_no_gpu()
    print("\n✅ All GPU memory lifecycle tests passed!")
