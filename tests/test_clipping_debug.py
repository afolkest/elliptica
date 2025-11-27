"""Debug script to test percentile clipping behavior."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from elliptica.serialization import load_project, load_render_cache
from elliptica.render import _get_palette_lut, _build_palette_lut
from elliptica.lospec import fetch_random_palette
from elliptica import defaults

# Load test project
project_path = Path("projects/test.flowcol")
cache_path = project_path.with_suffix('.flowcol.cache')

print("Loading project...")
state = load_project(str(project_path))
state.render_cache = load_render_cache(str(cache_path), state.project)
lic_array = state.render_cache.result.array

print(f"\n=== RAW LIC ARRAY STATS ===")
print(f"Shape: {lic_array.shape}")
print(f"Min: {lic_array.min():.6f}")
print(f"Max: {lic_array.max():.6f}")
print(f"Mean: {lic_array.mean():.6f}")
print(f"Std: {lic_array.std():.6f}")

# Check percentiles
print(f"\n=== PERCENTILE ANALYSIS ===")
for p in [0.5, 1.0, 2.0, 5.0, 10.0]:
    low = np.percentile(lic_array, p)
    high = np.percentile(lic_array, 100 - p)
    print(f"{p:5.1f}% - {100-p:5.1f}%: [{low:.6f}, {high:.6f}]  (range: {high-low:.6f})")

# Apply clipping with DEFAULT_CLIP_PERCENT
print(f"\n=== APPLYING CLIPPING (clip_percent={defaults.DEFAULT_CLIP_PERCENT}) ===")
clip_percent = defaults.DEFAULT_CLIP_PERCENT
lower = clip_percent / 100.0
upper = 1.0 - lower
vmin, vmax = np.percentile(lic_array, [lower * 100, upper * 100])
print(f"Percentile bounds: [{vmin:.6f}, {vmax:.6f}]")

if vmax > vmin:
    lic_clipped = np.clip((lic_array - vmin) / (vmax - vmin), 0.0, 1.0)
else:
    lic_clipped = np.clip(lic_array, 0.0, 1.0)

print(f"After clipping:")
print(f"  Min: {lic_clipped.min():.6f}")
print(f"  Max: {lic_clipped.max():.6f}")
print(f"  Mean: {lic_clipped.mean():.6f}")
print(f"  Std: {lic_clipped.std():.6f}")

# Test with library palette
print(f"\n=== LIBRARY PALETTE TEST ===")
lib_palette_name = "Ink & Gold"
lib_lut = _get_palette_lut(lib_palette_name)
print(f"Palette: {lib_palette_name}")
print(f"LUT shape: {lib_lut.shape}")
print(f"LUT min: {lib_lut.min():.3f}, max: {lib_lut.max():.3f}")
lum = np.mean(lib_lut, axis=1)
print(f"Luminance range: [{lum.min():.3f}, {lum.max():.3f}]")

# Apply color LUT
indices = (lic_clipped * (len(lib_lut) - 1)).astype(np.int32)
indices = np.clip(indices, 0, len(lib_lut) - 1)
rgb_lib = lib_lut[indices]
print(f"RGB output min: {rgb_lib.min():.3f}, max: {rgb_lib.max():.3f}, mean: {rgb_lib.mean():.3f}")

# Test with Lospec palette (3 colors)
print(f"\n=== LOSPEC PALETTE TEST (3 colors) ===")
print("Fetching random palette...")
palette_data = fetch_random_palette(timeout=5.0)
print(f"Palette: {palette_data['name']} by {palette_data['author']}")
print(f"Original colors: {len(palette_data['colors'])}")

# Sample 3 colors
colors_array = np.array(palette_data['colors'], dtype=np.float32)
if len(colors_array) > 3:
    indices_sample = np.random.choice(len(colors_array), size=3, replace=False)
    colors_array = colors_array[indices_sample]
    print(f"Sampled 3 colors")
else:
    colors_array = colors_array[:3].copy()

print(f"Before sorting:")
for i, color in enumerate(colors_array):
    lum = np.mean(color)
    print(f"  Color {i+1}: RGB=({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})  luminance={lum:.3f}")

# Sort by luminance
luminances = np.mean(colors_array, axis=1)
sort_indices = np.argsort(luminances)
colors_array = colors_array[sort_indices]

print(f"\nAfter sorting by luminance:")
for i, color in enumerate(colors_array):
    lum = np.mean(color)
    print(f"  Color {i+1}: RGB=({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})  luminance={lum:.3f}")

# Extend with black and white
black = np.array([0.0, 0.0, 0.0], dtype=np.float32)
white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
colors_array = np.vstack([black, colors_array, white])

print(f"\nAfter adding black/white:")
for i, color in enumerate(colors_array):
    lum = np.mean(color)
    print(f"  Color {i+1}: RGB=({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})  luminance={lum:.3f}")

# Build LUT
lospec_lut = _build_palette_lut(colors_array)
print(f"\nLospec LUT shape: {lospec_lut.shape}")
print(f"LUT min: {lospec_lut.min():.3f}, max: {lospec_lut.max():.3f}")
lum_lospec = np.mean(lospec_lut, axis=1)
print(f"Luminance range: [{lum_lospec.min():.3f}, {lum_lospec.max():.3f}]")

# Apply color LUT
indices = (lic_clipped * (len(lospec_lut) - 1)).astype(np.int32)
indices = np.clip(indices, 0, len(lospec_lut) - 1)
rgb_lospec = lospec_lut[indices]
print(f"RGB output min: {rgb_lospec.min():.3f}, max: {rgb_lospec.max():.3f}, mean: {rgb_lospec.mean():.3f}")

print("\n=== ANALYSIS ===")
print(f"Library palette luminance span: {lum.max() - lum.min():.3f}")
print(f"Lospec palette luminance span: {lum_lospec.max() - lum_lospec.min():.3f}")
if lum_lospec.max() - lum_lospec.min() < 0.5:
    print("⚠️  WARNING: Lospec palette has narrow luminance range - will look gray!")
