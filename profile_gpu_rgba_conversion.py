"""Profile GPU-based RGBA conversion to eliminate CPU bottlenecks.

This tests converting RGB to RGBA on the GPU before downloading to CPU.
"""

import time
import numpy as np
import torch
from PIL import Image
from flowcol.gpu import GPUContext
import dearpygui.dearpygui as dpg


def profile_step(name: str, func, *args, **kwargs):
    """Profile a single step and print timing."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{name:40s}: {elapsed:8.4f}s")
    return result, elapsed


def test_gpu_rgba_conversion(resolution: int = 7000):
    """Test converting RGB to RGBA on GPU before downloading."""
    print(f"\n{'='*60}")
    print(f"GPU RGBA Conversion Test at {resolution}x{resolution}")
    print(f"{'='*60}\n")

    h, w = resolution, resolution

    # Start with GPU uint8 RGB tensor (simulating postprocess output)
    print("Step 1: Creating simulated GPU uint8 RGB tensor...")
    rgb_uint8_gpu = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, device=GPUContext.device())
    print(f"  Tensor: {rgb_uint8_gpu.shape}, {rgb_uint8_gpu.dtype}, device: {rgb_uint8_gpu.device}")

    # Approach 1: Current (download RGB, then convert on CPU via PIL)
    print("\nApproach 1: Current pipeline (download RGB, PIL conversion)...")

    def current_approach():
        # Download RGB uint8
        rgb_cpu = GPUContext.to_cpu(rgb_uint8_gpu)

        # PIL conversion
        pil_img = Image.fromarray(rgb_cpu, mode='RGB')
        pil_rgba = pil_img.convert('RGBA')

        # Convert to float32 RGBA
        rgba_float = np.asarray(pil_rgba, dtype=np.float32) / 255.0
        return rgba_float.reshape(-1)

    rgba_current, t_current = profile_step("  Current approach (total)", current_approach)
    print(f"  Result shape: {rgba_current.shape}")

    # Approach 2: GPU RGBA conversion (convert to RGBA on GPU, then download)
    print("\nApproach 2: GPU RGBA conversion (convert on GPU, download RGBA)...")

    def gpu_rgba_approach():
        # Convert to float32 on GPU
        rgb_float_gpu = rgb_uint8_gpu.float() / 255.0

        # Add alpha channel on GPU
        alpha = torch.ones((h, w, 1), dtype=torch.float32, device=GPUContext.device())
        rgba_float_gpu = torch.cat([rgb_float_gpu, alpha], dim=2)

        # Synchronize GPU
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        # Download RGBA float32
        rgba_cpu = GPUContext.to_cpu(rgba_float_gpu)
        return rgba_cpu.reshape(-1)

    rgba_gpu, t_gpu = profile_step("  GPU approach (total)", gpu_rgba_approach)
    print(f"  Result shape: {rgba_gpu.shape}")

    # Approach 3: Detailed breakdown of GPU approach
    print("\nApproach 3: GPU approach with detailed timing...")

    def step_float_conversion():
        return rgb_uint8_gpu.float() / 255.0

    rgb_float_gpu, t_float = profile_step("  GPU: uint8 -> float32", step_float_conversion)

    def step_add_alpha():
        alpha = torch.ones((h, w, 1), dtype=torch.float32, device=GPUContext.device())
        return torch.cat([rgb_float_gpu, alpha], dim=2)

    rgba_float_gpu, t_alpha = profile_step("  GPU: add alpha channel", step_add_alpha)

    def step_sync():
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    _, t_sync = profile_step("  GPU: synchronize", step_sync)

    def step_download():
        return GPUContext.to_cpu(rgba_float_gpu)

    rgba_cpu, t_download = profile_step("  GPU -> CPU download", step_download)

    def step_reshape():
        return rgba_cpu.reshape(-1)

    rgba_flat, t_reshape = profile_step("  Reshape to flat", step_reshape)

    t_gpu_detailed = t_float + t_alpha + t_sync + t_download + t_reshape

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"{'='*60}")
    print(f"  Approach 1 (Current - CPU PIL):")
    print(f"    Total time:                {t_current:.4f}s")
    print()
    print(f"  Approach 2 (GPU RGBA):")
    print(f"    Total time:                {t_gpu:.4f}s")
    print()
    print(f"  Approach 3 (GPU detailed):")
    print(f"    uint8->float32:            {t_float:.4f}s")
    print(f"    Add alpha:                 {t_alpha:.4f}s")
    print(f"    GPU sync:                  {t_sync:.4f}s")
    print(f"    Download:                  {t_download:.4f}s")
    print(f"    Reshape:                   {t_reshape:.4f}s")
    print(f"    Total:                     {t_gpu_detailed:.4f}s")
    print()
    speedup = t_current / t_gpu if t_gpu > 0 else 0
    print(f"  Speedup (GPU vs Current):    {speedup:.2f}x")
    print(f"  Time saved:                  {t_current - t_gpu:.4f}s")
    print(f"{'='*60}\n")

    return {
        "current": t_current,
        "gpu": t_gpu,
        "speedup": speedup,
    }


def test_complete_pipeline_with_dpg(resolution: int = 7000):
    """Test the complete pipeline from GPU tensor to DPG texture."""
    print(f"\n{'='*60}")
    print(f"Complete Pipeline: GPU Tensor -> DPG Texture")
    print(f"{'='*60}\n")

    h, w = resolution, resolution

    # Initialize DPG
    dpg.create_context()
    registry_id = dpg.add_texture_registry()

    # Start with GPU uint8 RGB (simulating postprocess output)
    print("Creating GPU uint8 RGB tensor...")
    rgb_uint8_gpu = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, device=GPUContext.device())

    # Pipeline 1: Current approach (what texture_manager.py does)
    print("\nPipeline 1: Current (GPU RGB -> CPU RGB -> PIL -> RGBA -> DPG)...")

    def current_full_pipeline():
        # Download RGB
        rgb_cpu = GPUContext.to_cpu(rgb_uint8_gpu)

        # PIL conversion
        pil_img = Image.fromarray(rgb_cpu, mode='RGB')
        pil_rgba = pil_img.convert('RGBA')

        # Float conversion
        rgba_float = np.asarray(pil_rgba, dtype=np.float32) / 255.0
        rgba_flat = rgba_float.reshape(-1)

        # DPG upload
        tex_id = dpg.add_dynamic_texture(w, h, rgba_flat, parent=registry_id)
        return tex_id

    tex_id_1, t_current_full = profile_step("  Total time", current_full_pipeline)

    # Pipeline 2: Optimized (GPU RGBA conversion)
    print("\nPipeline 2: Optimized (GPU RGB -> GPU RGBA -> CPU RGBA -> DPG)...")

    def optimized_full_pipeline():
        # Convert to float and add alpha on GPU
        rgb_float_gpu = rgb_uint8_gpu.float() / 255.0
        alpha = torch.ones((h, w, 1), dtype=torch.float32, device=GPUContext.device())
        rgba_float_gpu = torch.cat([rgb_float_gpu, alpha], dim=2)

        # Sync and download
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        rgba_cpu = GPUContext.to_cpu(rgba_float_gpu)
        rgba_flat = rgba_cpu.reshape(-1)

        # DPG upload
        tex_id = dpg.add_dynamic_texture(w, h, rgba_flat, parent=registry_id)
        return tex_id

    tex_id_2, t_optimized_full = profile_step("  Total time", optimized_full_pipeline)

    # Cleanup
    dpg.destroy_context()

    # Summary
    speedup = t_current_full / t_optimized_full if t_optimized_full > 0 else 0
    print(f"\n{'='*60}")
    print("COMPLETE PIPELINE COMPARISON:")
    print(f"{'='*60}")
    print(f"  Current pipeline:            {t_current_full:.4f}s")
    print(f"  Optimized pipeline:          {t_optimized_full:.4f}s")
    print(f"  Speedup:                     {speedup:.2f}x")
    print(f"  Time saved:                  {t_current_full - t_optimized_full:.4f}s")
    print(f"{'='*60}\n")

    return {
        "current": t_current_full,
        "optimized": t_optimized_full,
        "speedup": speedup,
    }


if __name__ == "__main__":
    print(f"\nGPU Available: {GPUContext.is_available()}")
    print(f"GPU Device: {GPUContext.device()}")

    # Warmup
    print("\nWarming up GPU...")
    GPUContext.warmup()

    # Test GPU RGBA conversion
    test_gpu_rgba_conversion(7000)

    # Test complete pipeline with DPG
    test_complete_pipeline_with_dpg(7000)
