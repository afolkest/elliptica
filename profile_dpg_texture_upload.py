"""Profile DearPyGUI texture upload performance.

This script measures the actual DPG texture operations to see if they're a bottleneck.
"""

import time
import numpy as np
import dearpygui.dearpygui as dpg


def profile_step(name: str, func, *args, **kwargs):
    """Profile a single step and print timing."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{name:40s}: {elapsed:8.4f}s")
    return result, elapsed


def profile_dpg_texture_upload(resolution: int = 7000):
    """Profile DPG texture upload operations.

    Args:
        resolution: Image resolution (default 7000x7000)
    """
    print(f"\n{'='*60}")
    print(f"Profiling DPG Texture Upload at {resolution}x{resolution}")
    print(f"{'='*60}\n")

    # Initialize DPG (required for texture operations)
    dpg.create_context()

    # Create registry
    registry_id = dpg.add_texture_registry()

    # Prepare texture data (simulate the RGBA float32 data from texture_manager.py)
    h, w = resolution, resolution
    print(f"Creating {w}x{h} RGBA texture data...")

    # Simulate the actual data format used by DPG
    # RGBA float32 array in range [0, 1]
    rgba_data = np.random.rand(h, w, 4).astype(np.float32)
    rgba_flat = rgba_data.reshape(-1)

    print(f"  Shape: {rgba_flat.shape}")
    print(f"  Dtype: {rgba_flat.dtype}")
    print(f"  Memory: {rgba_flat.nbytes / 1024**2:.1f} MB")

    # Test 1: add_dynamic_texture (initial upload)
    print("\nTest 1: Initial texture upload with add_dynamic_texture...")

    def create_texture():
        return dpg.add_dynamic_texture(w, h, rgba_flat, parent=registry_id)

    tex_id, t_create = profile_step("  dpg.add_dynamic_texture()", create_texture)

    # Test 2: set_value (update existing texture)
    print("\nTest 2: Update texture with set_value...")

    # Modify the data slightly
    rgba_flat_modified = rgba_flat * 0.9

    def update_texture():
        dpg.set_value(tex_id, rgba_flat_modified)

    _, t_update = profile_step("  dpg.set_value()", update_texture)

    # Test 3: Multiple updates to see if it's consistent
    print("\nTest 3: Multiple consecutive updates...")
    update_times = []
    for i in range(5):
        rgba_flat_modified = rgba_flat * (0.9 + i * 0.02)
        def update_tex():
            dpg.set_value(tex_id, rgba_flat_modified)
        _, t = profile_step(f"  Update #{i+1}", update_tex)
        update_times.append(t)

    avg_update = np.mean(update_times)
    print(f"\n  Average update time: {avg_update:.4f}s")

    # Cleanup
    dpg.destroy_context()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    print(f"  Initial upload (add_dynamic_texture): {t_create:.4f}s")
    print(f"  Update (set_value):                    {t_update:.4f}s")
    print(f"  Average update time:                   {avg_update:.4f}s")
    print(f"{'='*60}\n")

    return {
        "create": t_create,
        "update": t_update,
        "avg_update": avg_update,
    }


def profile_full_pipeline(resolution: int = 7000):
    """Profile the complete pipeline from uint8 RGB to DPG texture."""
    print(f"\n{'='*60}")
    print(f"Complete Pipeline: uint8 RGB -> DPG RGBA Texture")
    print(f"{'='*60}\n")

    # Initialize DPG
    dpg.create_context()
    registry_id = dpg.add_texture_registry()

    h, w = resolution, resolution

    # Step 1: Simulate having uint8 RGB data from GPU
    print("Step 1: Creating simulated uint8 RGB data (from GPU)...")
    rgb_uint8 = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    print(f"  Shape: {rgb_uint8.shape}, dtype: {rgb_uint8.dtype}")
    print(f"  Memory: {rgb_uint8.nbytes / 1024**2:.1f} MB")

    # Step 2: Current approach via PIL
    print("\nCurrent approach: uint8 RGB -> PIL -> RGBA -> float32...")
    from PIL import Image

    def current_pipeline():
        # This is what texture_manager.py does
        pil_img = Image.fromarray(rgb_uint8, mode='RGB')
        pil_rgba = pil_img.convert('RGBA')
        width, height = pil_rgba.size
        rgba = np.asarray(pil_rgba, dtype=np.float32) / 255.0
        return width, height, rgba.reshape(-1)

    (tex_w, tex_h, tex_data), t_current = profile_step("  Current pipeline", current_pipeline)
    print(f"  Result: {tex_w}x{tex_h}, {tex_data.nbytes / 1024**2:.1f} MB")

    # Create texture with current approach
    def create_tex_current():
        return dpg.add_dynamic_texture(tex_w, tex_h, tex_data, parent=registry_id)

    tex_id_1, t_upload_current = profile_step("  DPG texture upload", create_tex_current)

    # Step 3: Optimized approach (direct conversion)
    print("\nOptimized approach: uint8 RGB -> float32 RGBA directly...")

    def optimized_pipeline():
        # Direct NumPy conversion without PIL
        rgba_float = np.zeros((h, w, 4), dtype=np.float32)
        rgba_float[..., :3] = rgb_uint8.astype(np.float32) / 255.0
        rgba_float[..., 3] = 1.0  # Full alpha
        return w, h, rgba_float.reshape(-1)

    (opt_w, opt_h, opt_data), t_optimized = profile_step("  Optimized pipeline", optimized_pipeline)
    print(f"  Result: {opt_w}x{opt_h}, {opt_data.nbytes / 1024**2:.1f} MB")

    # Create texture with optimized approach
    def create_tex_opt():
        return dpg.add_dynamic_texture(opt_w, opt_h, opt_data, parent=registry_id)

    tex_id_2, t_upload_opt = profile_step("  DPG texture upload", create_tex_opt)

    # Cleanup
    dpg.destroy_context()

    # Summary
    total_current = t_current + t_upload_current
    total_optimized = t_optimized + t_upload_opt
    speedup = total_current / total_optimized if total_optimized > 0 else 0

    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"{'='*60}")
    print(f"  Current approach:")
    print(f"    Conversion:        {t_current:.4f}s")
    print(f"    DPG upload:        {t_upload_current:.4f}s")
    print(f"    Total:             {total_current:.4f}s")
    print()
    print(f"  Optimized approach:")
    print(f"    Conversion:        {t_optimized:.4f}s")
    print(f"    DPG upload:        {t_upload_opt:.4f}s")
    print(f"    Total:             {total_optimized:.4f}s")
    print()
    print(f"  Speedup:             {speedup:.2f}x")
    print(f"  Time saved:          {total_current - total_optimized:.4f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test basic DPG texture operations
    profile_dpg_texture_upload(7000)

    # Test full pipeline comparison
    profile_full_pipeline(7000)
