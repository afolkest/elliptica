# Texture Pipeline Bottleneck Report

## Quick Summary

**Root Cause Found:** `refresh_render_texture()` is called on EVERY canvas draw when in render mode, not just when the texture actually needs updating.

**Location:** `/Users/afolkest/code/visual_arts/flowcol/flowcol/ui/dpg/canvas_renderer.py` line 58-59

**Impact:** 1.1-1.7 second delay every time the canvas redraws (zoom, pan, mode switch, etc.)

---

## The Problem

### Call Chain After Smear Completes

```
1. render_orchestrator.poll()  [render_orchestrator.py:216]
   └─> display_pipeline.refresh_display()  [display_pipeline_controller.py:50]
       ├─> texture_manager.refresh_render_texture()  ✓ CORRECT
       │   └─> [1.1s of postprocessing + conversion + upload]
       └─> canvas_renderer.mark_dirty()

2. Main loop  [app.py:402-404]
   └─> if canvas_renderer.canvas_dirty:
       └─> canvas_renderer.draw()  [canvas_renderer.py:39]
           └─> if view_mode == "render":
               └─> texture_manager.refresh_render_texture()  ✗ REDUNDANT!
                   └─> [ANOTHER 1.1s of postprocessing + conversion + upload]
```

### Why This Is Bad

After the render completes:
1. `refresh_display()` correctly updates the texture (1.1s)
2. Marks canvas dirty
3. **Next frame:** `draw()` updates the texture AGAIN (1.1s) ← wasteful!
4. **Every subsequent frame** where canvas is dirty: updates texture again

Canvas becomes dirty when:
- Zooming or panning the view
- Selecting conductors
- Changing any UI control
- Switching between edit/render mode
- File operations

**Each event triggers 1.1s of unnecessary reprocessing!**

---

## Timing Breakdown (7000×7000 Texture)

### Current Pipeline: GPU Tensor → DPG Texture

| Operation | File:Line | Time | % | Notes |
|-----------|-----------|------|---|-------|
| **GPU Postprocessing** |
| Colorization + overlays + smear | gpu/postprocess.py:191-208 | 0.133s | 12% | GPU operations |
| Convert to uint8 | gpu/postprocess.py:211 | 0.026s | 2% | GPU operation |
| GPU synchronize | gpu/postprocess.py:214-217 | **0.298s** | **27%** | ⚠️ Waiting for GPU |
| GPU → CPU download | gpu/__init__.py:66 | 0.030s | 3% | 140 MB transfer |
| **CPU Conversion** |
| Image.fromarray() | texture_manager.py:163 | **0.101s** | **9%** | ⚠️ PIL overhead |
| PIL RGB → RGBA | texture_manager.py:29 | 0.051s | 5% | PIL conversion |
| RGBA → float32 | texture_manager.py:31-32 | **0.482s** | **43%** | ⚠️ NumPy /255.0 |
| **DPG Upload** |
| dpg.set_value() | texture_manager.py:174 | **0.569s** | **N/A** | ⚠️ 747 MB upload |
| **TOTAL** | | **~1.7s** | | Per update |

### Where Time Is Spent

```
GPU Operations:       0.457s  (27%)
GPU → CPU Transfer:   0.030s  (2%)
CPU Conversion:       0.634s  (37%)  ← Can move to GPU!
DPG Upload:           0.569s  (34%)  ← Unavoidable
─────────────────────────────────────
TOTAL:                1.690s
```

---

## Profiling Results

### Test 1: Texture Conversion Pipeline
**Script:** `profile_texture_pipeline.py`

```
Breakdown (7000×7000 RGB → RGBA float):
  GPU postprocess:              0.133s (11.9%)
  Convert to uint8:             0.026s (2.3%)
  GPU synchronize:              0.298s (26.6%)  ← GPU still working
  GPU → CPU download:           0.030s (2.7%)
  Image.fromarray():            0.101s (9.0%)   ← PIL overhead
  PIL RGB → RGBA:               0.051s (4.6%)
  RGBA → float32:               0.482s (43.0%)  ← Biggest CPU bottleneck
  ──────────────────────────────────────────
  TOTAL:                        1.121s
```

### Test 2: DPG Texture Upload
**Script:** `profile_dpg_texture_upload.py`

```
DPG Operations (7000×7000 RGBA float32):
  add_dynamic_texture():        0.564s
  set_value():                  0.607s
  Average update:               0.569s

Data size: 747.7 MB (196M pixels × 4 channels × 4 bytes/float)
```

### Test 3: GPU RGBA Conversion
**Script:** `profile_gpu_rgba_conversion.py`

```
Comparison (uint8 RGB → RGBA float32 → DPG):
  Current (CPU):                1.036s
  Optimized (GPU):              0.727s
  ──────────────────────────────────────────
  Speedup:                      1.43x
  Time saved:                   0.309s
```

---

## Root Causes

### 1. Redundant Texture Refresh (CRITICAL)

**File:** `/Users/afolkest/code/visual_arts/flowcol/flowcol/ui/dpg/canvas_renderer.py`

**Lines 57-59:**
```python
# Only refresh render texture in render mode (avoids 11 GB/sec bandwidth waste in edit mode)
if view_mode == "render":
    self.app.display_pipeline.texture_manager.refresh_render_texture()
```

**Why it exists:** Comment suggests it's for optimization (avoiding refresh in edit mode).

**Problem:** It refreshes on EVERY draw, not just when texture changed.

**Fix:** Remove these lines. The texture is already refreshed by `display_pipeline.refresh_display()`.

### 2. CPU RGBA Conversion

**File:** `/Users/afolkest/code/visual_arts/flowcol/flowcol/ui/dpg/texture_manager.py`

**Lines 163-165:**
```python
pil_img = Image.fromarray(final_rgb, mode='RGB')
width, height, data = _image_to_texture_data(pil_img)
```

**Lines 29-32:**
```python
def _image_to_texture_data(img: Image.Image) -> Tuple[int, int, np.ndarray]:
    img = img.convert("RGBA")
    width, height = img.size
    rgba = np.asarray(img, dtype=np.float32) / 255.0  ← 196M divisions on CPU!
    return width, height, rgba.reshape(-1)
```

**Problem:**
- Creating PIL Image object (unnecessary)
- Converting RGB→RGBA on CPU
- Dividing 196M floats on CPU

**Fix:** Do RGBA conversion on GPU before download.

### 3. No Cache Invalidation

**File:** `/Users/afolkest/code/visual_arts/flowcol/flowcol/ui/dpg/texture_manager.py`

**Lines 122-175:** No fingerprinting logic

**Problem:** Can't detect if texture is already up-to-date.

**Fix:** Add fingerprint tracking (like conductor textures do at line 101-110).

---

## Answers to Your Questions

### 1. Break down texture update timing

See table above. Key findings:
- **43%** of conversion time is RGBA→float32 (line 31 of texture_manager.py)
- **27%** is GPU synchronize (waiting for smear/overlay ops to finish)
- **9%** is PIL Image.fromarray overhead
- **34%** is DPG upload (unavoidable, but should only happen when needed)

### 2. Is DPG texture upload the bottleneck?

**Yes and no.**

- DPG upload itself: 0.6s (unavoidable - it's copying 747 MB to GPU memory)
- But the real issue: **it's happening on every draw** when it should only happen when settings change
- After fixing redundant refresh, 0.6s is acceptable for when texture actually needs updating

### 3. Any image resizing/resampling happening?

**No.** Verified:
- No `Image.resize()` calls in the pipeline
- No `Image.thumbnail()` calls
- DPG handles display scaling via UV coordinates (canvas_renderer.py:97-107)
- Full 7000×7000 resolution is maintained throughout

### 4. Could we bypass PIL and upload GPU tensor directly?

**No, but we can optimize the path to DPG format.**

DPG requires: `np.ndarray, dtype=float32, shape=(W*H*4,)`

Current path:
```
GPU uint8 RGB → CPU uint8 RGB → PIL RGB → PIL RGBA → CPU float32 RGBA → DPG
     ↓               ↓              ↓           ↓             ↓
   0.03s          0.10s         0.05s       0.48s         0.57s
```

Optimized path:
```
GPU uint8 RGB → GPU float32 RGBA → CPU float32 RGBA → DPG
     ↓               ↓                    ↓             ↓
  <0.01s          0.07s                0.13s         0.57s
```

Saves: 0.31s by eliminating PIL and doing float conversion on GPU.

### 5. Are there multiple texture updates happening?

**YES - this is the main bug!**

Texture updates on:
1. ✓ Render completion (`render_orchestrator.poll` → `refresh_display`)
2. ✗ **Every canvas draw in render mode** ← BUG!
3. ✓ Display settings change (`display_pipeline.refresh_display`)
4. ✓ Smear parameter changes (triggers display refresh)

Issue #2 causes massive waste. Canvas can redraw many times per second if user is zooming/panning.

---

## Recommendations

### Immediate Fix (5 minutes)

**Remove redundant refresh from canvas draw loop.**

File: `/Users/afolkest/code/visual_arts/flowcol/flowcol/ui/dpg/canvas_renderer.py`

```diff
         view_mode = self.app.state.view_mode

-    # Only refresh render texture in render mode (avoids 11 GB/sec bandwidth waste in edit mode)
-    if view_mode == "render":
-        self.app.display_pipeline.texture_manager.refresh_render_texture()
-
     # Clear the layer (transform persists on layer)
```

**Expected result:** Texture only updates when `display_pipeline.refresh_display()` is explicitly called (render complete, settings change). No more 1.7s lag on every canvas redraw!

### Follow-up Optimization (30 minutes)

**Move RGBA conversion to GPU.**

Modify `flowcol/gpu/postprocess.py` around line 210:

```python
# Current: Return uint8 RGB
rgb_uint8_tensor = (rgb_tensor * 255.0).clamp(0, 255).to(torch.uint8)
torch.mps.synchronize()
return GPUContext.to_cpu(rgb_uint8_tensor)

# Optimized: Convert to RGBA float32 on GPU
h, w = rgb_tensor.shape[:2]
alpha = torch.ones((h, w, 1), dtype=torch.float32, device=rgb_tensor.device)
rgba_tensor = torch.cat([rgb_tensor, alpha], dim=2)  # (H, W, 4) float32

torch.mps.synchronize()
return GPUContext.to_cpu(rgba_tensor)  # Return float32 RGBA directly
```

Then update `texture_manager.py` to expect RGBA:

```python
# Current
final_rgb = apply_full_postprocess_hybrid(...)  # Returns uint8 RGB
pil_img = Image.fromarray(final_rgb, mode='RGB')
width, height, data = _image_to_texture_data(pil_img)

# Optimized
final_rgba = apply_full_postprocess_hybrid(...)  # Returns float32 RGBA
height, width = final_rgba.shape[:2]
data = final_rgba.reshape(-1)
```

**Expected result:** 0.31s faster per texture update (1.0s → 0.7s).

### Nice-to-Have (15 minutes)

**Add texture cache fingerprinting.**

```python
class TextureManager:
    def __init__(self, app):
        self.render_texture_fingerprint: Optional[tuple] = None

    def refresh_render_texture(self) -> None:
        # ... existing code to get cache and settings ...

        # Compute fingerprint
        fingerprint = (
            id(cache.result),  # Changes when render updates
            self.app.state.display_settings.to_tuple(),  # All display params
        )

        # Skip if already up-to-date
        if fingerprint == self.render_texture_fingerprint:
            return

        # ... do expensive postprocessing ...
        self.render_texture_fingerprint = fingerprint
```

**Expected result:** Eliminates updates when spamming same setting value.

---

## Conclusion

The 20-second delay mystery is solved:

1. **Main culprit:** Texture refreshes on every canvas draw (1.7s × N frames)
2. **Contributing factor:** CPU-based RGBA conversion (0.6s per update)
3. **Amplifying factor:** DPG upload blocks main thread (0.6s per update)

**After immediate fix:** Texture only updates when actually needed (~1-2 times per render)
**After GPU optimization:** Each update is 1.7s → 1.2s
**Combined benefit:** User experience goes from "frozen UI" to "smooth with occasional brief pause"

The texture upload itself (0.6s for 747 MB) is unavoidable given DPG's architecture, but it's acceptable when it only happens on actual setting changes rather than every frame.
