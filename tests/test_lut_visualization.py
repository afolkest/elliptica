"""Visualize what the LUT actually looks like."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))

from flowcol.lospec import fetch_random_palette
from flowcol.render import _build_palette_lut

# Fetch a random palette
print("Fetching random Lospec palette...")
palette_data = fetch_random_palette(timeout=5.0)
colors_orig = np.array(palette_data['colors'], dtype=np.float32)

print(f"\nPalette: {palette_data['name']} by {palette_data['author']}")
print(f"Original: {len(colors_orig)} colors\n")

# Sample 3 colors
if len(colors_orig) > 3:
    indices = np.random.choice(len(colors_orig), size=3, replace=False)
    colors_sampled = colors_orig[indices]
else:
    colors_sampled = colors_orig[:3].copy()

# Sort by luminance
luminances = np.mean(colors_sampled, axis=1)
sort_indices = np.argsort(luminances)
colors_sorted = colors_sampled[sort_indices]

# Add black and white
black = np.array([0.0, 0.0, 0.0], dtype=np.float32)
white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
colors_extended = np.vstack([black, colors_sorted, white])

print("Extended palette colors:")
for i, color in enumerate(colors_extended):
    lum = np.mean(color)
    sat = max(color) - min(color) if max(color) > 0 else 0
    print(f"  {i}: RGB=({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})  lum={lum:.2f}  sat={sat:.2f}")

# Build LUT
lut = _build_palette_lut(colors_extended, size=256)

print(f"\nLUT stats:")
print(f"  Luminance range: [{np.mean(lut, axis=1).min():.2f}, {np.mean(lut, axis=1).max():.2f}]")
lut_sats = np.max(lut, axis=1) - np.min(lut, axis=1)
print(f"  Saturation range: [{lut_sats.min():.2f}, {lut_sats.max():.2f}]")
print(f"  Saturation mean: {lut_sats.mean():.2f}")
print(f"  Low saturation entries (<0.3): {np.sum(lut_sats < 0.3)} / 256 ({100*np.sum(lut_sats < 0.3)/256:.1f}%)")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(12, 6))

# Plot 1: LUT as gradient
ax = axes[0]
ax.imshow(lut[np.newaxis, :, :], aspect='auto', interpolation='nearest')
ax.set_title(f"LUT Gradient ({len(colors_extended)} color stops)")
ax.set_xlabel("LUT Index (0-255)")
ax.set_yticks([])

# Add markers for color stops
positions = np.linspace(0, 255, len(colors_extended))
for i, pos in enumerate(positions):
    ax.axvline(pos, color='white', linewidth=2, alpha=0.7)
    ax.text(pos, 0.5, f"{i}", ha='center', va='center', color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

# Plot 2: Luminance
ax = axes[1]
lum = np.mean(lut, axis=1)
ax.plot(lum, 'k-', linewidth=2)
ax.set_ylabel("Luminance")
ax.set_xlabel("LUT Index")
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Mid-gray')
ax.legend()

# Plot 3: Saturation
ax = axes[2]
sat = np.max(lut, axis=1) - np.min(lut, axis=1)
ax.plot(sat, 'b-', linewidth=2)
ax.set_ylabel("Saturation")
ax.set_xlabel("LUT Index")
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.axhline(0.3, color='red', linestyle='--', alpha=0.3, label='Dull threshold')
ax.legend()

plt.tight_layout()
plt.savefig('/tmp/lut_analysis.png', dpi=150)
print("\n✓ Saved visualization to /tmp/lut_analysis.png")
plt.close()

# Show the problem zones
problem_indices = np.where(lut_sats < 0.3)[0]
if len(problem_indices) > 0:
    print(f"\n⚠️  PROBLEM: {len(problem_indices)} LUT entries have low saturation (<0.3)")
    print(f"   These will appear gray/muddy in the visualization")
    print(f"   Affected range: indices {problem_indices[0]} - {problem_indices[-1]}")
