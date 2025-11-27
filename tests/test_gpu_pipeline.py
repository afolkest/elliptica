#!/usr/bin/env python3
"""Test the full GPU postprocessing pipeline."""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from elliptica.types import Project, Conductor


def test_gpu_pipeline_imports():
    """Test that GPU pipeline modules can be imported."""
    try:
        from elliptica.gpu.ops import apply_highpass_gpu
        from elliptica.gpu.smear import apply_conductor_smear_gpu
        from elliptica.gpu.overlay import apply_region_overlays_gpu
        from elliptica.gpu.postprocess import apply_full_postprocess_hybrid
        print("✓ All GPU modules import successfully")
        return True
    except ImportError as e:
        print(f"⚠ GPU modules require torch: {e}")
        return False


def test_hybrid_cpu_fallback():
    """Test that hybrid pipeline falls back to CPU gracefully."""
    from elliptica.gpu.postprocess import apply_full_postprocess_hybrid

    # Create simple test data
    lic_array = np.random.rand(100, 100).astype(np.float32)
    project = Project(canvas_resolution=(100, 100))

    # Simple conductor
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:15, 5:15] = 1.0
    conductor = Conductor(
        id=0,
        mask=mask,
        voltage=1.0,
        position=(40.0, 40.0),
        smear_enabled=False,
    )
    project.conductors.append(conductor)

    # Run hybrid pipeline (will use GPU if available, CPU otherwise)
    try:
        final_rgb, used_percentiles = apply_full_postprocess_hybrid(
            scalar_array=lic_array,
            conductor_masks=None,
            interior_masks=None,
            conductor_color_settings={},
            conductors=project.conductors,
            render_shape=lic_array.shape,
            canvas_resolution=project.canvas_resolution,
            clip_percent=0.5,
            brightness=0.0,
            contrast=1.0,
            gamma=1.0,
            color_enabled=True,
            palette="Ink & Gold",
            lic_percentiles=None,
            use_gpu=True,
            scalar_tensor=None,
        )

        assert final_rgb.shape == (100, 100, 3), f"Expected (100, 100, 3), got {final_rgb.shape}"
        assert final_rgb.dtype == np.uint8, f"Expected uint8, got {final_rgb.dtype}"
        assert np.all(final_rgb >= 0) and np.all(final_rgb <= 255), "RGB values out of range"
        assert isinstance(used_percentiles, tuple) and len(used_percentiles) == 2, "Expected percentile tuple"

        print("✓ Hybrid pipeline works (GPU or CPU fallback)")
        return True
    except Exception as e:
        print(f"✗ Hybrid pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing GPU postprocessing pipeline...\n")

    passed = 0
    total = 2

    # Test 1: Imports
    if test_gpu_pipeline_imports():
        passed += 1

    # Test 2: Hybrid CPU fallback
    if test_hybrid_cpu_fallback():
        passed += 1

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
