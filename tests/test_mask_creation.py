#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from elliptica.mask_utils import load_alpha, create_masks, save_mask

source_path = "assets/source/disciples.png"
masks_dir = Path("assets/masks")
masks_dir.mkdir(parents=True, exist_ok=True)

mask = load_alpha(source_path)
# Binarize to avoid fractional alpha causing partition mismatch
mask = (mask > 0.5).astype(np.float32)

thickness = 3
shell, interior = create_masks(mask, thickness)

assert (shell + interior == mask).all(), "Shell + interior should equal original mask"
print(f"✓ Partition verified: shell + interior = original")

shell_path = masks_dir / "disciples_shell.png"
interior_path = masks_dir / "disciples_interior.png"

save_mask(shell, str(shell_path))
save_mask(interior, str(interior_path))
