"""Analyze saturation distribution in Lospec palettes."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from elliptica.lospec import fetch_random_palette

def rgb_to_hsv(rgb):
    """Convert RGB to HSV to measure saturation."""
    r, g, b = rgb
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if maxc == 0:
        s = 0
    else:
        s = (maxc - minc) / maxc
    return s, v

# Fetch a few random palettes and analyze
print("Analyzing saturation in random Lospec palettes...\n")

for i in range(5):
    palette_data = fetch_random_palette(timeout=5.0)
    colors = np.array(palette_data['colors'], dtype=np.float32)

    print(f"{i+1}. {palette_data['name']} ({len(colors)} colors)")

    saturations = []
    luminances = []
    for color in colors:
        sat, val = rgb_to_hsv(color)
        saturations.append(sat)
        luminances.append(np.mean(color))

    sat_array = np.array(saturations)
    lum_array = np.array(luminances)

    print(f"   Saturation: min={sat_array.min():.2f}, max={sat_array.max():.2f}, mean={sat_array.mean():.2f}")
    print(f"   Luminance:  min={lum_array.min():.2f}, max={lum_array.max():.2f}, mean={lum_array.mean():.2f}")

    # Check if 3 random samples would be dull
    if len(colors) >= 3:
        indices = np.random.choice(len(colors), size=3, replace=False)
        sample_sats = sat_array[indices]
        sample_lums = lum_array[indices]

        print(f"   Random 3-color sample:")
        print(f"     Saturation: {sample_sats.mean():.2f} (min={sample_sats.min():.2f})")
        print(f"     Luminance range: {sample_lums.max() - sample_lums.min():.2f}")

        if sample_sats.mean() < 0.4:
            print(f"     ⚠️  LOW SATURATION - will look dull!")
        if sample_lums.max() - sample_lums.min() < 0.6:
            print(f"     ⚠️  NARROW LUMINANCE - will look gray!")

    print()
