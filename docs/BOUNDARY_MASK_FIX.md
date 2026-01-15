# Boundary Mask Mismatch Fix

## Problem

2-3 pixel offset between where the LIC shows white noise (conductor region where field=0) and where region coloring can be applied. Users cannot color the white noise region because the overlay mask doesn't cover it.

## Root Cause

Two independent mask computations that diverge:

1. **PDE side** (`biharmonic_pde.py`): Applies band extension via `_extend_dirichlet_band(steps=2)` for clamped boundary conditions. The actual field is zeroed in this extended region.

2. **Overlay side** (`masks.py`): Uses original `boundary.mask` from geometry without any band extension. Masks are 2-3 pixels smaller than the actual conductor region.

## Solution Overview

**Single source of truth**: PDE returns the actual per-boundary masks that were used during solving. Interior masks are derived from these using `binary_fill_holes`.

### Data Flow

```
PDE Solve
    │
    ├─► per_boundary_masks[boundary_id] = actual Dirichlet region (with band extension)
    │
    ▼
RenderResult / RenderCache
    │
    ▼
Overlay (at display time)
    │
    ├─► boundary_mask = per_boundary_masks[id]  (direct use)
    │
    └─► interior_mask = fill_holes(boundary_mask) - boundary_mask  (derived)
```

### Benefits

- Masks guaranteed to match LIC (same data, single source of truth)
- No hardcoded extension values to maintain
- Automatically correct for any BC type or PDE variant
- Interior masks derived on-demand, not stored separately
- Future-proof: any mask manipulation in PDE automatically reflected

---

## Implementation Plan

### Phase 1: PDE Returns Per-Boundary Masks

**File**: `elliptica/pde/biharmonic_pde.py`

During solve, after band extension is applied, store each boundary's actual mask:

```python
def solve(...):
    per_boundary_masks = {}

    for boundary in project.boundaries:
        # Get the mask AFTER any band extension
        if boundary.bc_type == "clamped":
            # This boundary had band extension applied
            extended_mask = _extend_dirichlet_band(boundary.mask, steps=2)
            per_boundary_masks[boundary.id] = extended_mask
        else:
            # Neumann or other BC - use original mask
            per_boundary_masks[boundary.id] = boundary.mask

    # Include in solution dict
    solution["per_boundary_masks"] = per_boundary_masks

    return solution
```

**Note**: Need to track which mask corresponds to which boundary through the extension process. May require refactoring how `_extend_dirichlet_band` is called.

**Estimated changes**: ~40 lines

---

### Phase 2: Propagate Through Pipeline

**File**: `elliptica/pipeline.py`

Extract per-boundary masks from solution and attach to RenderResult:

```python
def perform_render(...):
    solution = pde.solve(...)

    # Extract per-boundary masks
    per_boundary_masks = solution.get("per_boundary_masks", {})

    result = RenderResult(
        array=lic_result,
        # ... existing fields ...
        per_boundary_masks=per_boundary_masks,
    )

    return result
```

**Estimated changes**: ~15 lines

---

### Phase 3: Update Data Structures

**File**: `elliptica/app/core.py`

Add field to RenderResult:

```python
@dataclass
class RenderResult:
    array: np.ndarray
    # ... existing fields ...
    per_boundary_masks: Optional[Dict[int, np.ndarray]] = None
```

Add field to RenderCache:

```python
@dataclass
class RenderCache:
    result: RenderResult
    # ... existing fields ...
    per_boundary_masks: Optional[Dict[int, np.ndarray]] = None
    per_boundary_masks_gpu: Optional[Dict[int, Any]] = None  # GPU versions
```

**Estimated changes**: ~10 lines

---

### Phase 4: Cache Population

**File**: `elliptica/ui/dpg/render_orchestrator.py`

Copy masks from result to cache and upload to GPU:

```python
def job():
    # ... after render completes ...

    cache = RenderCache(
        result=result,
        # ... existing fields ...
        per_boundary_masks=result.per_boundary_masks,
    )

    # Upload to GPU for fast overlay
    if result.per_boundary_masks and GPUContext.is_available():
        cache.per_boundary_masks_gpu = {}
        for bid, mask in result.per_boundary_masks.items():
            cache.per_boundary_masks_gpu[bid] = GPUContext.to_gpu(mask)
```

**Estimated changes**: ~20 lines

---

### Phase 5: Derive Interior Masks

**File**: `elliptica/gpu/overlay.py` (or new utility)

Add function to derive interior from boundary mask:

```python
from scipy.ndimage import binary_fill_holes

def derive_interior_from_boundary(boundary_mask: np.ndarray) -> np.ndarray:
    """Derive interior mask from boundary mask using flood fill.

    Interior = region enclosed by boundary but not part of boundary itself.
    Uses binary_fill_holes to find enclosed regions.

    Args:
        boundary_mask: Binary mask of the boundary (where field = 0)

    Returns:
        Interior mask (float32), or zeros if no interior exists
    """
    binary = boundary_mask > 0.5

    if not np.any(binary):
        return np.zeros_like(boundary_mask, dtype=np.float32)

    # Fill all holes (regions unreachable from image border)
    filled = binary_fill_holes(binary)

    # Interior = filled region minus original boundary
    interior = filled & ~binary

    if not np.any(interior):
        return np.zeros_like(boundary_mask, dtype=np.float32)

    return interior.astype(np.float32)
```

**Estimated changes**: ~25 lines

---

### Phase 6: Update Overlay to Use New Masks

**File**: `elliptica/gpu/overlay.py`

Modify overlay functions to use PDE masks instead of geometry masks:

```python
def apply_region_overlays_gpu(
    base_rgb: torch.Tensor,
    render_cache: RenderCache,
    color_config: ColorConfig,
    boundaries: List[Boundary],
    ...
) -> torch.Tensor:
    """Apply region color overlays using PDE-derived masks."""

    for boundary in boundaries:
        bid = boundary.id

        # Get boundary mask from PDE (single source of truth)
        if render_cache.per_boundary_masks_gpu and bid in render_cache.per_boundary_masks_gpu:
            boundary_mask = render_cache.per_boundary_masks_gpu[bid]
        elif render_cache.per_boundary_masks and bid in render_cache.per_boundary_masks:
            boundary_mask = torch.from_numpy(render_cache.per_boundary_masks[bid])
        else:
            continue  # No mask for this boundary

        # Derive interior on-demand
        interior_mask = derive_interior_from_boundary_gpu(boundary_mask)

        # Apply boundary color
        if color_config.get_boundary_color(bid):
            base_rgb = apply_mask_color(base_rgb, boundary_mask, color)

        # Apply interior color
        if color_config.get_interior_color(bid):
            base_rgb = apply_mask_color(base_rgb, interior_mask, color)

    return base_rgb
```

**Estimated changes**: ~50 lines

---

### Phase 7: Remove Old Mask Path (Cleanup)

**Files**: `elliptica/gpu/postprocess.py`, `elliptica/postprocess/masks.py`

- Remove calls to `rasterize_boundary_masks()` for overlay purposes
- Keep `rasterize_boundary_masks()` if needed elsewhere, or deprecate
- Remove `boundary_masks` and `interior_masks` from RenderCache if no longer used

**Estimated changes**: ~30 lines removed

---

## Files Changed Summary

| File | Changes | Lines |
|------|---------|-------|
| `elliptica/pde/biharmonic_pde.py` | Track & return per-boundary masks | ~40 |
| `elliptica/pipeline.py` | Extract masks, attach to result | ~15 |
| `elliptica/app/core.py` | Add fields to RenderResult, RenderCache | ~10 |
| `elliptica/ui/dpg/render_orchestrator.py` | Copy masks to cache, GPU upload | ~20 |
| `elliptica/gpu/overlay.py` | Use PDE masks, derive interiors | ~75 |
| `elliptica/gpu/postprocess.py` | Update to use new mask source | ~20 |
| `elliptica/postprocess/masks.py` | Cleanup (optional) | -30 |

**Total**: ~150-180 lines changed

---

## Backward Compatibility

**Project files (.elliptica)**: Unaffected. Masks computed at render time.

**Cache files (.elliptica.cache)**: Old caches won't have `per_boundary_masks`.

Handle gracefully:
```python
if render_cache.per_boundary_masks is None:
    # Old cache - needs re-render for correct masks
    # Either: trigger re-render, or fall back to old (buggy) behavior
```

No migration code needed. Just `Optional[...] = None` and None checks.

---

## Testing Plan

1. **Basic alignment test**:
   - Render with clamped BC boundary
   - Apply color to boundary region
   - Verify colored region exactly matches white noise in LIC (no gaps)

2. **Interior test**:
   - Create ring-shaped boundary (annulus)
   - Apply interior color
   - Verify interior color fills center hole only, not the boundary band

3. **Multiple boundaries**:
   - Create multiple boundaries with different BC types
   - Verify each boundary's mask is independently correct

4. **Complex topology**:
   - Create annulus (two concentric circles as one boundary)
   - Verify interior = center hole, boundary = annular region

5. **Edge cases**:
   - Boundary touching image border
   - Very small boundaries
   - Overlapping boundaries

---

## Estimated Time

- Implementation: 3-4 hours
- Testing: 1-2 hours
- Total: ~5 hours

---

## Open Questions

1. **GPU interior derivation**: Should `binary_fill_holes` run on CPU then upload, or implement GPU version? CPU is simpler, only runs once per render.

2. **Caching derived interiors**: Derive on-demand each frame, or cache after first derivation? Probably cache for performance.

3. **Other PDEs**: Does this approach work for all PDE types, or just biharmonic? Need to verify each PDE returns appropriate per-boundary masks.
