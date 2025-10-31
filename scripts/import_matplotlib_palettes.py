"""Import matplotlib colormaps into FlowCol user palette library.

Run this script once to bulk-add matplotlib's high-quality colormaps to your library.
You can then curate them in the FlowCol UI (right-click to delete).
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import flowcol
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from flowcol.render import add_palette

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib not installed. Install with: pip install matplotlib")
    exit(1)

# High-quality perceptually-uniform colormaps
RECOMMENDED_CMAPS = [
    # Perceptually uniform sequential
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",

    # Sequential
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",

    # Diverging
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",

    # Cyclic
    "twilight",
    "twilight_shifted",
    "hsv",

    # Miscellaneous
    "turbo",
    "rainbow",
    "jet",
    "nipy_spectral",
    "gist_rainbow",
    "ocean",
    "terrain",
    "copper",
]

def import_colormaps(cmap_names: list[str], num_stops: int = 256):
    """Import matplotlib colormaps into FlowCol user library.

    Args:
        cmap_names: List of matplotlib colormap names to import
        num_stops: Number of color stops to sample (default 256 for smooth gradients)
    """
    imported = 0
    skipped = 0

    for name in cmap_names:
        try:
            cmap = plt.get_cmap(name)

            # Sample the colormap at evenly-spaced points
            colors = []
            for i in np.linspace(0, 1, num_stops):
                rgba = cmap(i)
                # Take only RGB, drop alpha
                colors.append((rgba[0], rgba[1], rgba[2]))

            # Add to user library
            add_palette(name, colors)
            print(f"✓ Imported: {name}")
            imported += 1

        except Exception as e:
            print(f"✗ Skipped {name}: {e}")
            skipped += 1

    print(f"\nImported {imported} palettes, skipped {skipped}")
    print(f"User library saved to: flowcol/palettes_user.json")
    print("\nOpen FlowCol and curate your library (right-click palettes to delete)")

if __name__ == "__main__":
    print("Importing matplotlib colormaps into FlowCol...")
    print(f"Adding {len(RECOMMENDED_CMAPS)} palettes\n")
    import_colormaps(RECOMMENDED_CMAPS)
