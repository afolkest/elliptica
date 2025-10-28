# FlowCol Architecture Audit

Based on thorough codebase review, prioritized by impact (80/20 rule).

---

## Critical Issues Worth Fixing

### **0. Poisson API Regression Blocks Tests** 🧨 Critical

**Impact**: 30% of value (correctness + CI signal)

**Location**: `poisson.py:98`, `tests/test_poisson.py:80`

**Problem**:
- Public API changed from `boundary_type=` to four directional parameters.
- Tests (and likely downstream tooling) still call the old signature.
- Result: every Poisson test raises immediately, so we have zero coverage of the field solver path.

**Impact**:
- Core physics code is effectively untested; regressions would ship unnoticed.
- Breaks developer confidence and slows iteration because the suite is red by default.

**Fix**:
1. Provide a compatibility shim or restore a keyword that maps to the new args.
2. Update tests to the current API once the shim exists (keeps suite green both ways).
3. Add a regression test that exercises both call styles until the deprecation window ends.

**Effort**: Low (50-80 lines + test updates)

---

### **1. GPU Tensor Memory Leak/Lifecycle Chaos** ✅ FIXED

**Status**: ✅ FIXED

**Impact**: 30% of value

**Location**: `app/core.py:RenderCache`, `app/actions.py:ensure_render()`, `ui/dpg/app.py`

**Problem**:
GPU tensors (`result_gpu`, `display_array_gpu`, `ex_gpu`, `ey_gpu`) had inconsistent lifecycle management:
- Created in `ensure_render()` but never explicitly freed before new uploads
- Cleared in `clear_render_cache()` but no MPS cache flush
- VRAM accumulation across multiple renders

**Solution Implemented**:
1. ✅ Added `GPUContext.empty_cache()` method that calls `torch.mps.empty_cache()` (gpu/__init__.py:66-74)
2. ✅ Updated `clear_render_cache()` to call `empty_cache()` after clearing tensors (app/core.py:208-210)
3. ✅ Updated `ensure_render()` to free ALL old GPU tensors before uploading new ones (app/actions.py:227-240):
   - Clears `result_gpu`, `ex_gpu`, `ey_gpu` (field data)
   - **Critically**: Also clears `display_array_gpu` (largest tensor - downsampled display frame)
   - Calls `invalidate_cpu_cache()` to ensure lazy CPU cache is cleared
   - Then calls `empty_cache()` to release VRAM back to system
4. ✅ Added comprehensive test suite in `test_gpu_memory_lifecycle.py`:
   - Tests cleanup on `clear_render_cache()`
   - Tests cleanup on re-render
   - Tests multiple sequential renders don't leak
   - Tests graceful fallback when GPU unavailable

**Result**: Proper MPS memory management with explicit cleanup at the right lifecycle points. VRAM is released immediately when old renders are replaced or cleared. All GPU tensors including the large display frame are freed before allocating new ones.

---

### **2. Redundant Mask Rasterization** 🔥 High Performance Win

**Impact**: 25% of value (15-20% total render time saved)

**Location**: `app/actions.py:ensure_render()`, `postprocess/masks.py:rasterize_conductor_masks()`

**Problem**:
Conductor masks are rasterized MULTIPLE times per render:
1. During field computation (field.py:37-52) - scaled and blurred for Poisson solve
2. After render for display resolution (actions.py:227-238) - for colorization overlays
3. Again at full resolution for edge blur (app.py:1789-1796)
4. Region colorization does it AGAIN per-region (postprocess/color.py:126+)

Each rasterization involves:
- `scipy.ndimage.zoom()` (expensive interpolation)
- Memory allocation for full-sized masks
- Multiple passes over same data

**Impact**:
- Significant performance hit on every render (~15-20% total render time)
- Memory waste - storing multiple versions of same masks
- Cache invalidation complexity - which masks to update when?

**Fix**:
1. Compute ALL mask resolutions ONCE during render and store in RenderResult
2. Add `RenderResult.conductor_masks_full` and `conductor_masks_display` fields (plus any overlay-ready crops)
3. Reuse these cached masks for all downstream operations, including conductor overlays
4. Consider using GPU for mask rasterization (torchvision's interpolate is very fast)

**Effort**: Low-Medium (200 lines of changes across multiple files)

---

### **3. UI God Object (2888 lines)** 🏗️ Architecture Debt

**Impact**: 20% of value (maintainability multiplier)

**Location**: `ui/dpg/app.py`

**Problem**:
The FlowColApp class is a God Object handling:
- UI construction (260+ lines just for `build()`)
- Canvas rendering and interaction
- File I/O and serialization
- Background render orchestration
- State synchronization
- Postprocessing pipeline
- Debouncing timers
- Cache management
- Conductor manipulation
- ALL widget callbacks

**Impact**:
- Extremely difficult to test individual features
- High coupling - changing one UI element can break unrelated features
- Impossible to reuse any logic in non-DPG contexts
- Merge conflicts nightmare for team collaboration
- Difficult to reason about state consistency
- Blocks alternative UI implementations (web, CLI, etc.)

**Fix** (High-level):
1. Extract `CanvasController` - handles canvas rendering, mouse interaction, conductor selection
2. Extract `RenderOrchestrator` - handles background render jobs, progress tracking
3. Extract `DisplayPipelineController` - handles postprocessing, GPU/CPU path selection
4. Extract `FileDialogManager` - handles all file dialog creation/callbacks
5. Keep `FlowColApp` as thin coordinator that wires these together

**Effort**: High (1-2 weeks, but enables future velocity)

---

### **4. Backend/UI Coupling via DisplaySettings** 🔗 Architecture Boundary Violation

**Status**: ✅ FIXED

**Location**: `postprocess/color.py:build_base_rgb()`, `render.py:colorize_array()`

**Problem**:
Pure backend colorization functions directly accept `DisplaySettings` objects, violating the functional/OOP boundary stated in CLAUDE.md.

**Solution Implemented**:
1. ✅ Created `ColorParams` dataclass in `flowcol/postprocess/color.py:9-21`
2. ✅ Updated `build_base_rgb()` to accept `ColorParams` instead of `DisplaySettings`
3. ✅ Updated `apply_region_overlays()` to accept `ColorParams` instead of `DisplaySettings`
4. ✅ Added `DisplaySettings.to_color_params()` method in `flowcol/app/core.py:60-70`
5. ✅ Updated all 5 call sites to use `.to_color_params()`
6. ✅ Verified no backend imports from `app/`

**Result**: Clean backend/UI separation enables headless usage without UI dependencies.

---

### **5. Two Sources of Truth for Display Data** 🐛 State Management Bug

**Status**: ✅ FIXED

**Impact**: 10% of value (correctness issue)

**Location**: `app/core.py:RenderCache`, `app/actions.py:ensure_render()`

**Problem**:
Critical render data existed in TWO places:
- `RenderCache.display_array` - CPU-side downsampled array
- `RenderCache.display_array_gpu` - GPU-side downsampled tensor

These could become desynchronized with no validation mechanism.

**Solution Implemented**:
1. ✅ Converted `RenderCache` from `@dataclass` to regular class (core.py:73-159)
2. ✅ Established mutual exclusion: EITHER `display_array_gpu` (GPU primary) OR `_display_array_cpu` (CPU primary)
3. ✅ Made `display_array` a `@property` with lazy cached download from GPU (core.py:106-117)
4. ✅ Added `set_display_array_gpu()` / `set_display_array_cpu()` to enforce single source (core.py:127-148)
5. ✅ Added `invalidate_cpu_cache()` for when GPU tensor is modified (core.py:150-152)
6. ✅ Updated UI layer to use new setters (app.py:893, 904)
7. ✅ Eliminated wasteful `.copy()` in ensure_render() - now uses reference (actions.py:246)

**Result**: Impossible to have desynchronized CPU/GPU state. Single source of truth with lazy cached downloads.

---

### **6. Overlay Colorization Re-renders Whole Frame** 🎨 Performance Debt

**Impact**: 15% of value (major share of postprocess cost)

**Location**: `postprocess/color.py:126`

**Problem**:
- Each conductor overlay calls `colorize_array` on the entire LIC texture—even if the mask covers a tiny area.
- With multiple conductors this becomes O(N·pixels) full-frame recolorization.
- Undercuts the GPU acceleration work because we keep reallocating and re-normalizing on CPU.

**Fix**:
1. Reuse the already-normalized base RGB and blend palette variations per mask.
2. Or crop to mask bounding boxes before recolorization.
3. Combine with Issue #2 so mask data + normalized scalar fields are shared.

**Effort**: Low-Medium (120-180 lines, mostly refactor)

---

### **7. Postprocess Phase Still CPU-Bound** ⚡ GPU Roadblock

**Impact**: 25% of value (keeps us from a true GPU-first pipeline)

**Location**: `app/actions.py:ensure_render()`, `pipeline.py:74`, `postprocess/color.py`, `postprocess/blur.py`

**Problem**:
- After LIC we immediately drop back to NumPy/CPU for downsampling, high-pass, colorization, overlays, and edge blur.
- GPU tensors are created (`RenderCache.result_gpu`) but most downstream steps still read from `display_array` (CPU).
- Extra copies between CPU↔GPU negate the benefit of the existing tensor cache and make “full GPU pipeline” impossible.

**Fix**:
1. Upload the LIC result once in `ensure_render` and treat the GPU tensor as the primary representation.
2. Implement `apply_postprocess` (Gaussian high-pass, CLAHE) with torch ops so the scalar field stays on-device.
3. Use `downsample_lic_hybrid`’s GPU path by default; only fall back to SciPy when no device is present.
4. Drive `ensure_base_rgb` through `build_base_rgb_gpu` and keep the RGB tensor on GPU until export.
5. Rework `apply_region_overlays` to blend palette/solid masks using torch tensors rather than recoloring the whole frame on CPU.
6. Port anisotropic edge blur to torch by calling the existing GPU edge-blur helpers (or reimplementing the directional blur there).
7. Only move data back to NumPy when saving PNGs or handing a frame to DearPyGui.

**Effort**: Medium-High (400-600 lines touching pipeline, postprocess, GPU modules)

---

## Recommended Fix Roadmap (80/20)

### **Phase 1 - Quick Wins** (1 day)
1. Restore Poisson API compatibility so tests pass (Issue #0)
2. Fix GPU tensor lifecycle + populate `display_array_gpu` (Issue #1)
3. Eliminate redundant mask work and overlay recolorization (Issues #2 + #6)

**Outcome**: Solid performance win + fixes correctness bug

### **Phase 2 - Architecture Cleanup** (3-5 days)
4. Extract backend/UI coupling (Issue #4) - Enable headless usage
5. Fix dual state management (Issue #5) - Single source of truth for display data
6. Drive the entire postprocess pipeline on GPU (Issue #7) to avoid CPU round-trips

**Outcome**: Clean architecture, testable code, enables batch rendering

### **Phase 3 - Long-term Maintainability** (1-2 weeks)
7. Split massive UI file (Issue #3) - Extract 4-5 smaller classes

**Outcome**: Sustainable codebase, alternative UI implementations possible

---

## Detailed Implementation Order

This section provides the optimal sequencing for implementing all fixes, with rationale for dependencies and synergies.

### **Step 1: Fix Poisson API Tests** (Issue #0) - 1 hour

**Why first**: Must have green tests before refactoring anything else.

**Implementation**:
1. Add compatibility shim in `poisson.py` that accepts old `boundary_type=` parameter
2. Map old parameter to new directional boundary parameters
3. Update tests to use new API while keeping shim for backward compatibility
4. Add test that exercises both call styles

**Validation**: `pytest tests/test_poisson.py` passes

**Blocks**: Everything else - no confident refactoring without tests

---

### **Step 2: Backend/UI Decoupling** (Issue #4) - ✅ COMPLETED

**Implementation**:
1. ✅ Created `ColorParams` dataclass in `flowcol/postprocess/color.py:9-21`
2. ✅ Added `DisplaySettings.to_color_params()` method in `flowcol/app/core.py:60-70`
3. ✅ Updated backend functions:
   - `postprocess/color.py:build_base_rgb()` - now accepts `ColorParams`
   - `postprocess/color.py:apply_region_overlays()` - now accepts `ColorParams`
4. ✅ Updated all call sites (5 locations):
   - `flowcol/app/actions.py:302`
   - `flowcol/gpu/pipeline.py:198-208`
   - `flowcol/ui/dpg/app.py:971`
   - `flowcol/ui/dpg/app.py:2219`
   - `test_per_region_colorization.py:146`

**Validation**:
- ✅ All files compile successfully
- ✅ No backend imports from `app/`
- ✅ Clean separation achieved

**Result**: Backend colorization functions are now pure and headless-ready. GPU pipeline no longer depends on UI types.

---

### **Step 3: Single Source of Truth** (Issue #5) - ✅ COMPLETED

**Why third**: Establishes where data lives before optimizing how it's computed. This clarifies ownership before GPU lifecycle management.

**Implementation**:
1. ✅ Converted `RenderCache` from `@dataclass` to regular class with explicit `__init__`
2. ✅ Established single source pattern:
   - `display_array_gpu` is primary when GPU available
   - `_display_array_cpu` is primary when GPU not available
   - Mutual exclusion enforced - can't have both
3. ✅ Added `@property display_array` with lazy cached download from GPU (core.py:106-117)
4. ✅ Added `set_display_array_gpu()` / `set_display_array_cpu()` to switch sources (core.py:127-148)
5. ✅ Added `invalidate_cpu_cache()` method (core.py:150-152)
6. ✅ Updated UI layer to use setters:
   - GPU path: `cache.set_display_array_gpu(downsampled_gpu)` (app.py:893)
   - CPU path: `cache.set_display_array_cpu(downsampled)` (app.py:904)
7. ✅ Eliminated wasteful `.copy()` in `ensure_render()` - now uses reference (actions.py:246)

**Validation**:
- ✅ Per-region colorization tests pass
- ✅ Single source enforced - no way to have CPU/GPU out of sync
- ✅ Lazy download cached - no repeated allocations
- ✅ Memory efficient - reference instead of copy

**Enables**:
- Clear contract for Issue #1 (GPU lifecycle)
- Foundation for Issue #7 (GPU pipeline)

**Files changed**: `app/core.py` (87 lines), `app/actions.py` (3 lines), `ui/dpg/app.py` (13 lines)

---

### **Step 4: GPU Tensor Lifecycle** (Issue #1) - ✅ COMPLETED

**Why fourth**: Now that Issue #5 established GPU as source of truth, manage its lifecycle properly.

**Implementation**:
1. ✅ Added `GPUContext.empty_cache()` method (gpu/__init__.py:66-74)
   - Wraps `torch.mps.empty_cache()` for MPS backend
   - Safe to call when GPU unavailable (checks availability first)
2. ✅ Updated `clear_render_cache()` to call `empty_cache()` after clearing tensors (app/core.py:208-210)
   - Ensures VRAM is released when cache is cleared
3. ✅ Updated `ensure_render()` to free ALL old GPU tensors before uploading new ones (app/actions.py:227-240)
   - Prevents VRAM accumulation across multiple renders
   - Clears `result_gpu`, `ex_gpu`, `ey_gpu`, AND `display_array_gpu` (largest tensor!)
   - Calls `invalidate_cpu_cache()` to clear lazy CPU download cache
   - Then calls `empty_cache()` to release VRAM
   - Finally uploads new tensors
4. ✅ GPU tensors are uploaded immediately after render when GPU available (actions.py:268-282)
5. ✅ Added comprehensive test suite (`test_gpu_memory_lifecycle.py`):
   - `test_gpu_cleanup_on_clear()` - Verifies cleanup on cache clear
   - `test_gpu_cleanup_on_rerender()` - Verifies old tensors freed before new upload
   - `test_multiple_renders()` - Tests 5 sequential renders at different resolutions
   - `test_empty_cache_graceful_when_no_gpu()` - Tests graceful fallback

**Validation**:
- ✅ All GPU memory lifecycle tests pass
- ✅ Multiple sequential renders work without accumulation
- ✅ GPU tensors properly freed and replaced on re-render
- ✅ `empty_cache()` is safe when GPU unavailable

**Enables**:
- Confident GPU usage in Issue #7
- Foundation for GPU mask caching in Issue #2

**Files changed**:
- `flowcol/gpu/__init__.py` (9 lines added)
- `flowcol/app/core.py` (3 lines added)
- `flowcol/app/actions.py` (14 lines added - includes display_array_gpu cleanup)
- `test_gpu_memory_lifecycle.py` (new file, 134 lines)

---

### **Step 5: Mask Deduplication** (Issue #2) - 5-6 hours

**Why fifth**: Now that GPU lifecycle is clear (Issue #1) and data ownership is established (Issue #5), can cache masks properly.

**Pre-work - Audit mask consumers** (1 hour):
1. Grep for all mask rasterization call sites
2. Check if any code relies on list ordering vs ID-based lookup
3. Document which resolution each consumer needs:
   - Canvas resolution (post-crop, no margins)
   - Compute resolution (pre-crop, with margins)
   - Display resolution (downsampled for UI)

**Implementation**:
1. Add mask storage to `RenderResult`:
   ```python
   @dataclass
   class RenderResult:
       # ... existing fields
       # Masks stored as dict (Python 3.7+ preserves insertion order)
       # Keyed by conductor ID for efficient lookup
       conductor_masks_canvas: Optional[Dict[int, np.ndarray]] = None  # Canvas resolution (cropped)
       conductor_masks_display: Optional[Dict[int, np.ndarray]] = None  # Display resolution (downsampled)
   ```
   **Note**: Use `dict` (not `OrderedDict`) - Python 3.7+ preserves insertion order. Store masks keyed by conductor ID.

   **Resolution clarity**:
   - `conductor_masks_canvas`: Same as `result.array.shape` (post-crop, no margins)
   - `conductor_masks_display`: Downsampled to display resolution
   - If any consumer needs pre-crop resolution, add `conductor_masks_compute` field

2. Compute masks ONCE in `ensure_render()`:
   ```python
   # After field computation and crop
   canvas_res = result.array.shape  # This is already cropped
   display_res = (state.display_h, state.display_w)

   # Rasterize once per resolution
   result.conductor_masks_canvas = {
       c.id: rasterize_conductor_mask(c, canvas_res, project.canvas_resolution)
       for c in project.conductors
   }
   result.conductor_masks_display = {
       c.id: rasterize_conductor_mask(c, display_res, project.canvas_resolution)
       for c in project.conductors
   }
   ```

3. Update all consumers to use cached masks:
   - `postprocess/color.py:apply_region_overlays()` - use `result.conductor_masks_display`
   - Edge blur in `ui/dpg/app.py` - use `result.conductor_masks_canvas`
   - Check if any code expects list order and convert to ID-based lookup
4. Remove duplicate rasterization calls

**Validation**:
- Profiling shows `scipy.ndimage.zoom()` called once per resolution per render (not 3-4x total)
- Renders still match pixel-for-pixel
- Overlay code works with ID-based dict lookup
- 15-20% speedup in total render time

**Enables**:
- Issue #6 can reuse cached masks for overlays
- Issue #7 can convert masks to GPU tensors once

**Files changed**: `types.py`, `app/actions.py`, `postprocess/masks.py`, `postprocess/color.py`, `ui/dpg/app.py`

---

### **Step 6: Fix Overlay Recolors** (Issue #6) - 2-3 hours

**Why sixth**: Builds on Issue #4 (clean `ColorParams`) and Issue #2 (cached masks). Quick win before big GPU refactor.

**Implementation**:
1. Change `apply_region_overlays()` to reuse base RGB:
   ```python
   def apply_region_overlays(base_rgb, masks_display, color_params, palette_overrides):
       # Instead of recolorizing entire frame per conductor:
       # 1. Start with base_rgb (already normalized)
       # 2. For each conductor, blend palette variation only in masked region
       result = base_rgb.copy()
       for cond_id, mask in masks_display.items():
           if cond_id in palette_overrides:
               # Blend alternative palette only where mask > 0
               blend_palette_in_region(result, mask, palette_overrides[cond_id])
       return result
   ```
2. Add bounding box optimization - only process masked regions:
   ```python
   bbox = get_mask_bbox(mask)
   masked_region = result[bbox]
   # ... process only masked_region
   ```
3. Use cached masks from Issue #2

**Validation**:
- Conductor overlays still look identical
- Profiling shows speedup (O(mask_pixels) instead of O(N·total_pixels))
- Performance test with 10 conductors shows significant improvement

**Enables**: Clean foundation for GPU overlay blending in Issue #7

**Files changed**: `postprocess/color.py`, call sites in `ui/dpg/app.py`

---

### **Step 7: Full GPU Pipeline** (Issue #7) - 2-3 days

**Why last major refactor**: Benefits from all previous fixes. This is where everything comes together.

**Implementation** (multi-part):

**Part A: GPU Postprocessing + CLAHE Cleanup** (Day 1)
1. **Remove CLAHE dead code** (fold in cleanup from removed Step 0):
   - Remove `apply_highpass_clahe()` from `render.py:512`
   - Remove CLAHE defaults from `defaults.py` (lines 19-23)
   - Remove `highpass_enabled` and related config from `pipeline.py`
   - Remove CLAHE imports and tests from `tests/test_postprocess_pipeline.py`
   - Update any documentation mentioning CLAHE

2. Move high-pass filter to GPU:
   ```python
   def apply_highpass_gpu(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
       blurred = torchvision.transforms.functional.gaussian_blur(tensor, kernel_size, sigma)
       return tensor - blurred
   ```
3. Keep postprocessed scalar field on GPU

**Note**: Folding CLAHE cleanup into this step avoids modifying UI/state code twice. CLAHE is wired into `PostProcessConfig`, `pipeline.apply_postprocess()`, and tests - removing it cleanly requires touching the same files we're rewriting for GPU anyway.

**Part B: GPU Downsampling** (Day 1)
1. Use existing `downsample_lic_hybrid()` GPU path by default
2. Upload LIC result once in `ensure_render()` as GPU tensor
3. Keep downsampled result on GPU (`display_array_gpu`)

**Part C: GPU Colorization** (Day 2)
1. Expand `build_base_rgb_gpu()` to handle all colorization:
   ```python
   def build_base_rgb_gpu(scalar_tensor, color_params, ctx):
       # Normalize, apply brightness/contrast/gamma
       # Apply palette lookup
       # Return RGB tensor on GPU
   ```
2. Use `ColorParams` from Issue #4
3. Keep RGB tensor on GPU until final export

**Part D: GPU Overlay Blending** (Day 2)
1. Upload conductor masks to GPU (reuse cached masks from Issue #2)
2. Implement overlay blending in torch:
   ```python
   def blend_palette_overlay_gpu(base_rgb_tensor, mask_tensor, palette_rgb):
       # Blend alternative palette in masked regions
       # All tensor operations, stays on GPU
   ```
3. Replace CPU overlay logic with GPU version

**Part E: GPU Edge Blur** (Day 3)
1. **Implement `apply_anisotropic_edge_blur_gpu()`** in `gpu/edge_blur.py`:
   - Currently only building blocks exist (gaussian_blur_gpu, etc.)
   - Need to implement full anisotropic blur using ex/ey field direction
   - Reference existing CPU implementation for algorithm
2. Keep ex/ey tensors on GPU (already done in Issue #1)
3. Apply blur directly to GPU RGB tensor

**Note**: The function referenced doesn't exist yet - needs implementation before Part E can proceed.

**Part F: Integration** (Day 3)
1. Refactor `ensure_base_rgb()` and related functions to use GPU path by default
2. Only download to NumPy for final PNG save or DearPyGui display
3. Add GPU memory profiling and optimization

**Validation**:
- Renders still match pixel-for-pixel (or within tolerance for floating point differences)
- Major speedup - profiling shows minimal CPU↔GPU transfers
- GPU memory usage is reasonable (add limits if needed)
- Can still fall back to CPU when GPU unavailable

**Enables**: True GPU-first rendering pipeline

**Files changed**: `pipeline.py`, `app/actions.py`, `postprocess/color.py`, `postprocess/blur.py`, `gpu/ops.py`, `gpu/pipeline.py`, `ui/dpg/app.py`

---

### **Step 8: Split UI God Object** (Issue #3) - 1-2 weeks

**Why last**: All backend is now clean (Issues #4, #7), so extraction is cleaner. This is long-term maintainability work.

**Implementation** (multi-part):

**Part A: Extract CanvasController**
```python
class CanvasController:
    def __init__(self, app_state: AppState):
        self.state = app_state
        self.selected_conductor_id: Optional[int] = None

    def handle_mouse_click(self, x, y):
        # Canvas interaction logic

    def render_canvas(self, texture_data):
        # Canvas rendering logic
```

**Part B: Extract RenderOrchestrator**
```python
class RenderOrchestrator:
    def __init__(self, app_state: AppState):
        self.state = app_state
        self.render_thread: Optional[threading.Thread] = None

    def start_render(self):
        # Background render job management

    def check_progress(self):
        # Progress tracking
```

**Part C: Extract DisplayPipelineController**
```python
class DisplayPipelineController:
    def __init__(self, app_state: AppState, gpu_ctx: GPUContext):
        self.state = app_state
        self.gpu_ctx = gpu_ctx

    def update_display(self):
        # Postprocessing and display pipeline
```

**Part D: Extract FileDialogManager**
```python
class FileDialogManager:
    def __init__(self, app_state: AppState):
        self.state = app_state

    def show_load_dialog(self):
        # File dialog creation and callbacks
```

**Part E: Refactor FlowColApp**
```python
class FlowColApp:
    def __init__(self):
        self.state = AppState()
        self.canvas = CanvasController(self.state)
        self.render_orchestrator = RenderOrchestrator(self.state)
        self.display_pipeline = DisplayPipelineController(self.state, self.gpu_ctx)
        self.file_dialogs = FileDialogManager(self.state)

    def build(self):
        # Thin UI construction that wires controllers together
```

**Validation**:
- UI still works identically
- Can instantiate and test controllers independently
- Can write unit tests for individual features
- Line count per file is reasonable (<500 lines per controller)

**Enables**:
- Alternative UI implementations (web, CLI)
- Comprehensive unit testing
- Team collaboration without merge conflicts
- Future feature additions are easier

**Files changed**: `ui/dpg/app.py` split into multiple files in `ui/dpg/` directory

---

## Testing Checkpoints

After each step, validate progress:

| Step | Test Command | Expected Outcome |
|------|--------------|------------------|
| #1 | `pytest tests/test_poisson.py` | All tests pass |
| #2 | Write headless script using colorization | No UI imports needed |
| #3 | Memory profiler during render | Single display array, not duplicates; no draw loop allocations |
| #4 | Load/save/load project, check VRAM | Consistent memory usage |
| #5 | Profile rasterization calls | Called once per resolution, not 3-4x total |
| #6 | Overlay performance test | Speedup vs baseline |
| #7 | Profile GPU/CPU transfers | Minimal transfers, major speedup |
| #8 | Instantiate controllers independently | Works without full UI |

---

## Why This Order Works

1. **Tests first** (#0) - Can't refactor without confidence
2. **Interfaces before implementations** (#4) - Establishes clean contracts
3. **Clarify ownership before optimization** (#5, #1) - Know where data lives, then manage it
4. **Cache before consuming** (#2) - Compute masks once, then #6 and #7 reuse them
5. **Incremental GPU adoption** (#1, #5, #7) - Establish patterns, then go all-in
6. **Small wins before big refactor** (#6 before #7) - Build confidence
7. **Big UI refactor last** (#3) - Benefits from all backend cleanup

Each step is independently valuable and leaves the system in a working state. Can stop at any point with improvements shipped.

---

## Estimated Timeline

- **Step 1** (Issue #0 - Poisson API): ✅ 1 hour - COMPLETED
- **Step 2** (Issue #4 - ColorParams): ✅ 2-3 hours - COMPLETED
- **Step 3** (Issue #5 - Single source of truth): ✅ 4 hours - COMPLETED
- **Step 4** (Issue #1 - GPU lifecycle): ✅ 4 hours - COMPLETED
- **Step 5** (Issue #2 - Mask deduplication): 5-6 hours (includes audit)
- **Step 6** (Issue #6 - Overlay recolors): 2-3 hours
- **Step 7** (Issue #7 - Full GPU pipeline + CLAHE cleanup): 2-3 days
- **Step 8** (Issue #3 - UI refactor): 1-2 weeks

**Total**: ~2 weeks for Steps 1-7 (all performance and architecture wins), +1-2 weeks for Step 8 (UI maintainability)

**Progress**: Steps 1-4 completed (11-12 hours). Remaining: ~1 week for Steps 5-7, +1-2 weeks for Step 8.

**Note**: CLAHE cleanup (originally Step 0) has been folded into Step 7 Part A to avoid modifying UI/state code twice.

---

## Medium Priority Issues (Not Blocking)

### **8. Missing GPU Tensor Validation**
**Location**: `gpu/edge_blur.py`, `gpu/pipeline.py`, `postprocess/color.py`

GPU operations assume tensors are on correct device with correct dtype but never validate. Can cause cryptic PyTorch errors.

**Fix**: Add device/dtype validation helper function.

**Effort**: Low (50 lines)

---

### **9. Serialization Doesn't Save GPU Cache State**
**Location**: `serialization.py:save_render_cache()`, `serialization.py:load_render_cache()`

GPU tensors are silently lost when saving. First operation after load is 5-10x slower (needs re-upload).

**Fix**: Document this is intentional, OR add GPU tensor reconstruction on load.

**Effort**: Low (documentation) or Medium (implementation)

---

### **10. Field Computation Uses Inconsistent Blur Modes**
**Location**: `field.py:compute_field()`, lines 44-52

Conductor blur scaling uses averages which creates uneven blur on rectangular canvases.

**Fix**: Use geometric mean or min dimension for fractional mode, exact scale per axis for pixel mode.

**Effort**: Low (50 lines + tests)

---

### **11. Missing Type Annotations**
**Location**: Various, especially `render.py`, `pipeline.py`

Many backend functions lack complete type annotations. Degrades IDE autocomplete and type checking.

**Fix**: Add complete type annotations to all public APIs.

**Effort**: Medium (across entire codebase)

---

## Non-Issues (Correctly Designed)

✅ **Functional backend separation** - field.py, poisson.py, render.py are correctly stateless
✅ **OOP UI layer** - FlowColApp appropriately uses classes for stateful UI
✅ **GPU acceleration structure** - Clean fallback from GPU→CPU with try/except
✅ **Serialization format** - ZIP-based .flowcol format is sensible and extensible
✅ **No circular dependencies** - Import graph is acyclic
✅ **Poisson solver** - Correctly uses algebraic multigrid, good boundary condition handling

---

## Summary

**If you have 1 day**: Land #0, #1, #2, #6 – unblock tests and reclaim render/overlay performance
**If you have 1 week**: Do the above plus #4, #5, #7 – architectural wins that keep things fast and testable
**If you have more time**: Finish #3 – the large UI refactor for long-term sustainability

Mask dedupe (#2) + overlay tweaks (#6) remain the quickest high-value wins. The UI God Object (#3) is still the largest investment but unlocks future velocity.
