"""Profile the texture update pipeline to identify bottlenecks.

This script simulates the texture update path with profiling at each step.
"""

import time
import numpy as np
import torch
from PIL import Image
from flowcol.gpu import GPUContext


def profile_step(name: str, func, *args, **kwargs):
    """Profile a single step and print timing."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{name:40s}: {elapsed:8.4f}s")
    return result, elapsed


def simulate_texture_pipeline(resolution: int = 7000):
    """Simulate the full texture update pipeline with profiling.

    Args:
        resolution: Image resolution (default 7000x7000)
    """
    print(f"\n{'='*60}")
    print(f"Profiling Texture Pipeline at {resolution}x{resolution}")
    print(f"{'='*60}\n")

    # Simulate having GPU tensor (from smear computation)
    print("Step 1: Creating simulated GPU tensor...")
    h, w = resolution, resolution
    scalar_tensor = torch.rand(h, w, dtype=torch.float32, device=GPUContext.device())
    print(f"  Tensor shape: {scalar_tensor.shape}, device: {scalar_tensor.device}")

    # Simulate GPU postprocessing (converts to RGB)
    print("\nStep 2: Simulating GPU postprocessing...")
    def gpu_postprocess():
        # Simulate colorization: grayscale -> RGB
        rgb_tensor = torch.stack([scalar_tensor, scalar_tensor, scalar_tensor], dim=2)
        # Simulate some operations
        rgb_tensor = torch.clamp(rgb_tensor * 1.2 + 0.1, 0.0, 1.0)
        return rgb_tensor

    rgb_tensor, t_gpu = profile_step("  GPU postprocess", gpu_postprocess)
    print(f"  RGB tensor shape: {rgb_tensor.shape}, dtype: {rgb_tensor.dtype}")

    # Step 3: Convert to uint8 on GPU
    print("\nStep 3: Converting to uint8 on GPU...")
    def to_uint8():
        return (rgb_tensor * 255.0).clamp(0, 255).to(torch.uint8)

    rgb_uint8_tensor, t_uint8 = profile_step("  Convert to uint8 (GPU)", to_uint8)

    # Step 4: Synchronize GPU
    print("\nStep 4: Synchronizing GPU...")
    def sync_gpu():
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    _, t_sync = profile_step("  GPU synchronize", sync_gpu)

    # Step 5: Download to CPU (GPUContext.to_cpu)
    print("\nStep 5: Downloading to CPU...")
    def download_to_cpu():
        return GPUContext.to_cpu(rgb_uint8_tensor)

    final_rgb, t_download = profile_step("  GPU -> CPU download", download_to_cpu)
    print(f"  CPU array shape: {final_rgb.shape}, dtype: {final_rgb.dtype}")
    print(f"  Memory size: {final_rgb.nbytes / 1024**2:.1f} MB")

    # Step 6: PIL Image.fromarray
    print("\nStep 6: Converting to PIL Image...")
    def create_pil():
        return Image.fromarray(final_rgb, mode='RGB')

    pil_img, t_pil = profile_step("  Image.fromarray()", create_pil)
    print(f"  PIL Image size: {pil_img.size}, mode: {pil_img.mode}")

    # Step 7: Convert PIL to RGBA
    print("\nStep 7: Converting PIL RGB -> RGBA...")
    def convert_rgba():
        return pil_img.convert("RGBA")

    pil_rgba, t_rgba = profile_step("  PIL convert('RGBA')", convert_rgba)

    # Step 8: Convert RGBA to float texture data
    print("\nStep 8: Converting RGBA to float texture data...")
    def rgba_to_texture():
        width, height = pil_rgba.size
        rgba = np.asarray(pil_rgba, dtype=np.float32) / 255.0
        return width, height, rgba.reshape(-1)

    (tex_w, tex_h, tex_data), t_texture = profile_step("  RGBA -> float texture data", rgba_to_texture)
    print(f"  Texture data shape: {tex_data.shape}, dtype: {tex_data.dtype}")
    print(f"  Texture memory: {tex_data.nbytes / 1024**2:.1f} MB")

    # Summary
    print(f"\n{'='*60}")
    print("TIMING BREAKDOWN:")
    print(f"{'='*60}")
    total = t_gpu + t_uint8 + t_sync + t_download + t_pil + t_rgba + t_texture

    breakdown = [
        ("GPU postprocess (RGB conversion)", t_gpu),
        ("Convert to uint8 (GPU)", t_uint8),
        ("GPU synchronize", t_sync),
        ("GPU -> CPU download", t_download),
        ("Image.fromarray()", t_pil),
        ("PIL RGB -> RGBA", t_rgba),
        ("RGBA -> float texture data", t_texture),
    ]

    for name, elapsed in breakdown:
        pct = (elapsed / total * 100) if total > 0 else 0
        print(f"  {name:40s}: {elapsed:8.4f}s ({pct:5.1f}%)")

    print(f"  {'-'*58}")
    print(f"  {'TOTAL':40s}: {total:8.4f}s")
    print(f"{'='*60}\n")

    # Check for potential optimizations
    print("POTENTIAL OPTIMIZATIONS:")
    print("-" * 60)

    # The GPU -> CPU download is necessary, but...
    if t_download > 1.0:
        print(f"⚠️  GPU->CPU download is slow ({t_download:.2f}s)")
        print("   This is expected for large textures (196 MB)")

    # PIL operations
    pil_total = t_pil + t_rgba
    if pil_total > 0.5:
        print(f"⚠️  PIL operations total {pil_total:.2f}s")
        print("   Could bypass PIL entirely:")
        print("   - Skip Image.fromarray() + convert('RGBA')")
        print("   - Directly convert uint8 RGB -> float32 RGBA on GPU")
        print("   - Download RGBA float32 directly")

    # RGBA conversion
    if t_texture > 0.5:
        print(f"⚠️  RGBA conversion is slow ({t_texture:.2f}s)")
        print("   This is NumPy array operations + division")
        print("   Could be done on GPU before download")

    # GPU sync
    if t_sync > 0.1:
        print(f"⚠️  GPU synchronize is taking {t_sync:.2f}s")
        print("   This means GPU operations are still running")

    print("\nRECOMMENDED OPTIMIZATION:")
    print("-" * 60)
    print("Instead of:")
    print("  GPU uint8 RGB -> CPU uint8 RGB -> PIL RGB -> PIL RGBA -> float32 RGBA")
    print("\nDo:")
    print("  GPU uint8 RGB -> GPU float32 RGBA -> CPU float32 RGBA")
    print("\nThis eliminates:")
    print("  - Image.fromarray() call")
    print("  - PIL convert('RGBA') call")
    print("  - NumPy float32 conversion on CPU")
    print("  - Saves one full copy of the image")
    print()


if __name__ == "__main__":
    # Test with different resolutions
    print(f"\nGPU Available: {GPUContext.is_available()}")
    print(f"GPU Device: {GPUContext.device()}")

    # Warmup
    print("\nWarming up GPU...")
    GPUContext.warmup()

    # Profile at 7k resolution (actual use case)
    simulate_texture_pipeline(7000)

    # Optionally test smaller size for comparison
    # simulate_texture_pipeline(2000)
