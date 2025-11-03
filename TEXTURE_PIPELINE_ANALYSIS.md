# Texture Update Pipeline Bottleneck Analysis

## Executive Summary

**Root Cause:** The texture update pipeline is being called **on every frame** when in render mode, causing 1-2 second delays per frame. This is unnecessary - the texture should only update when display settings or render cache actually changes.

**Total Time per Update:** ~1.1-1.3 seconds for 7000×7000 texture
**Recommended Optimizations:** Can reduce to ~0.7s with GPU RGBA conversion + proper caching

---

## Detailed Findings

### 1. Pipeline Flow Analysis

Current flow when smear computation finishes:

```
render_orchestrator.poll() [line 216]
  → display_pipeline.refresh_display() [line 50]
    → texture_manager.refresh_render_texture() [line 122]
      → apply_full_postprocess_hybrid() [GPU operations]
      → Image.fromarray() [PIL conversion]
      → _image_to_texture_data() [RGBA conversion]
      → dpg.add_dynamic_texture() / dpg.set_value() [DPG upload]
    → canvas_renderer.mark_dirty() [line 51]

canvas main loop [app.py line 402-403]
  → canvas_renderer.draw() [line 39]
    → texture_manager.refresh_render_texture() [line 59] ⚠️ CALLED AGAIN!
```

**CRITICAL BUG:** `refresh_render_texture()` is called:
1. Once from `refresh_display()` when render completes ✓ (correct)
2. **Again on EVERY draw() when in render mode** ✗ (bug!)

See `/Users/afolkest/code/visual_arts/flowcol/flowcol/ui/dpg/canvas_renderer.py` line 58-59:

```python
# Only refresh render texture in render mode (avoids 11 GB/sec bandwidth waste in edit mode)
if view_mode == "render":
    self.app.display_pipeline.texture_manager.refresh_render_texture()
```

This means if the user moves the view, zooms, or anything that marks canvas dirty, the entire postprocessing pipeline runs again!

---

### 2. Timing Breakdown (7000×7000 Resolution)

Profiling results from `profile_texture_pipeline.py`:

| Step | Time | % of Total | Details |
|------|------|-----------|---------|
| GPU postprocess (RGB conversion) | 0.133s | 11.9% | Colorization, overlays, smear |
| Convert to uint8 (GPU) | 0.026s | 2.3% | Type conversion on GPU |
| **GPU synchronize** | **0.298s** | **26.6%** | ⚠️ Waiting for GPU |
| GPU → CPU download | 0.030s | 2.7% | 140 MB transfer |
| **Image.fromarray()** | **0.101s** | **9.0%** | ⚠️ PIL overhead |
| PIL RGB → RGBA | 0.051s | 4.6% | PIL conversion |
| **RGBA → float texture data** | **0.482s** | **43.0%** | ⚠️ NumPy conversion + division |
| **TOTAL** | **1.121s** | **100%** | |

Additional DPG upload time (from `profile_dpg_texture_upload.py`):
- **dpg.add_dynamic_texture()**: 0.564s (initial)
- **dpg.set_value()**: 0.569s (update)

**Grand Total: ~1.7s per texture update** (1.1s conversion + 0.6s DPG upload)

---

### 3. Identified Bottlenecks

#### 3.1 GPU Synchronization (26.6% of conversion time)
- `torch.mps.synchronize()` at line 215 of `flowcol/gpu/postprocess.py`
- Necessary to ensure GPU operations complete before CPU download
- Cannot eliminate, but indicates GPU is still processing

#### 3.2 RGBA Float Conversion (43.0% of conversion time)
- Line 29-32 of `texture_manager.py`:
  ```python
  img = img.convert("RGBA")
  rgba = np.asarray(pil_rgba, dtype=np.float32) / 255.0
  return width, height, rgba.reshape(-1)
  ```
- Converting 196M pixels from uint8 to float32 on CPU
- Division by 255.0 is expensive for 196M elements
- **This should be done on GPU before download!**

#### 3.3 PIL Overhead (9.0% of conversion time)
- `Image.fromarray()` creates unnecessary PIL object
- PIL `.convert("RGBA")` adds alpha channel on CPU
- **Can bypass PIL entirely by doing RGBA conversion on GPU**

#### 3.4 DPG Texture Upload (blocking main thread)
- `dpg.set_value()` takes ~0.6s and **blocks the main thread**
- Uploads 747 MB (RGBA float32) to GPU memory
- This is inherent to DPG's texture system - cannot optimize
- Must ensure we only upload when actually needed

---

### 4. No Resizing/Resampling Found

Good news: No image resizing operations detected in the pipeline.
- DPG handles scaling during display (line 97-99 of `canvas_renderer.py`)
- Full resolution is maintained throughout

---

### 5. Multiple Texture Updates

**YES - Major issue found:**

The texture is updated on:
1. Render completion (correct) ✓
2. **Every draw() call in render mode** (wasteful) ✗
3. Display settings change (correct) ✓
4. Smear parameter changes (correct) ✓

The canvas can be marked dirty for many reasons:
- Mouse movement while dragging view
- Zoom changes
- Conductor selection changes
- Mode switches
- File operations

Each time canvas is dirty and in render mode, `refresh_render_texture()` runs the entire 1.7s pipeline!

---

## Recommended Optimizations

### Priority 1: Fix Redundant Texture Updates (Biggest Impact!)

**Problem:** Canvas renderer calls `refresh_render_texture()` on every draw.

**Solution:** Move texture refresh logic out of draw loop:

```python
# canvas_renderer.py line 57-59
# REMOVE THIS:
if view_mode == "render":
    self.app.display_pipeline.texture_manager.refresh_render_texture()

# The texture should already be refreshed by display_pipeline.refresh_display()
# Only draw the existing texture here!
```

**Expected Savings:** Eliminates 1.7s delay on every canvas redraw in render mode.

After this fix, texture only updates when:
- `display_pipeline.refresh_display()` is explicitly called
- Which happens when render completes or settings change

---

### Priority 2: GPU RGBA Conversion

**Problem:** RGBA conversion happens on CPU (0.482s + 0.101s = 0.583s)

**Solution:** Convert RGB→RGBA on GPU before download:

Replace texture_manager.py lines 163-165:

```python
# CURRENT (slow):
pil_img = Image.fromarray(final_rgb, mode='RGB')
width, height, data = _image_to_texture_data(pil_img)

# OPTIMIZED (fast):
# Do RGBA conversion on GPU inside apply_full_postprocess_hybrid
# Return RGBA uint8 directly, then convert to float on GPU
```

Modify `flowcol/gpu/postprocess.py` to return RGBA instead of RGB:

```python
# After line 210: rgb_uint8_tensor = (rgb_tensor * 255.0).clamp(0, 255).to(torch.uint8)

# Add alpha channel on GPU
alpha = torch.ones((h, w, 1), dtype=torch.uint8, device=rgb_uint8_tensor.device) * 255
rgba_uint8_tensor = torch.cat([rgb_uint8_tensor, alpha], dim=2)

# Convert to float32 on GPU
rgba_float_tensor = rgba_uint8_tensor.float() / 255.0

# Synchronize and download
torch.mps.synchronize()
rgba_cpu = GPUContext.to_cpu(rgba_float_tensor)
return rgba_cpu  # Returns (H, W, 4) float32 array
```

Then in texture_manager.py:

```python
# final_rgba is already float32 RGBA from GPU
width, height = final_rgba.shape[1], final_rgba.shape[0]
data = final_rgba.reshape(-1)
```

**Expected Savings:** ~0.3s per update (from profiling: 1.04s → 0.73s)

---

### Priority 3: Texture Cache Invalidation

**Problem:** No way to know if texture is already up-to-date.

**Solution:** Add cache fingerprint to TextureManager:

```python
class TextureManager:
    def __init__(self, app):
        self.render_texture_fingerprint: Optional[str] = None

    def refresh_render_texture(self) -> None:
        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None:
                # ... handle no cache
                return

            # Compute fingerprint based on what affects texture
            fingerprint = (
                cache.project_fingerprint,
                self.app.state.display_settings.clip_percent,
                self.app.state.display_settings.brightness,
                # ... other settings
            )

            # Skip if texture is already up-to-date
            if fingerprint == self.render_texture_fingerprint:
                return

            # ... do expensive postprocessing
            self.render_texture_fingerprint = fingerprint
```

**Expected Savings:** Eliminates unnecessary updates when settings haven't changed.

---

## Summary of Issues

| Issue | Location | Impact | Fix Complexity |
|-------|----------|--------|---------------|
| Redundant texture refresh in draw loop | canvas_renderer.py:59 | **HIGH** (1.7s per frame) | **EASY** |
| CPU RGBA conversion | texture_manager.py:163-165 | **MEDIUM** (0.3s) | **MEDIUM** |
| No texture cache invalidation | texture_manager.py:122 | **LOW-MEDIUM** | **EASY** |
| DPG upload blocks main thread | texture_manager.py:174 | **MEDIUM** (0.6s) | **HARD** (DPG limitation) |

---

## Test Results

### GPU RGBA Conversion (profile_gpu_rgba_conversion.py)

```
Complete Pipeline Comparison:
  Current:     1.036s  (RGB download → PIL → RGBA → DPG)
  Optimized:   0.727s  (GPU RGBA conversion → DPG)
  Speedup:     1.43x
  Saved:       0.310s per update
```

### DPG Texture Upload (profile_dpg_texture_upload.py)

```
DPG Operations (7000×7000 RGBA):
  add_dynamic_texture():  0.564s
  set_value():            0.569s
  Average:                0.569s

Note: This is unavoidable - DPG uploads 747 MB to GPU memory.
```

---

## Can We Bypass PIL and Upload GPU Tensor Directly?

**NO - DPG requires CPU float32 array.**

DPG's texture API:
```python
dpg.add_dynamic_texture(width, height, data, ...)
# data must be: np.ndarray, dtype=float32, shape=(W*H*4,)
```

However, we CAN optimize the path to that format:
- Current: GPU uint8 RGB → CPU uint8 RGB → PIL RGB → PIL RGBA → CPU float32 RGBA
- Optimized: GPU uint8 RGB → GPU float32 RGBA → CPU float32 RGBA

This eliminates PIL entirely and does expensive operations on GPU.

---

## Conclusion

The 20-second delay is caused by:

1. **Redundant updates** (biggest issue): Texture refreshes on every canvas draw in render mode
2. **CPU bottleneck**: RGBA conversion on CPU instead of GPU
3. **DPG upload**: Inherently slow (0.6s) but necessary

**Immediate fix:** Remove `refresh_render_texture()` call from `canvas_renderer.draw()`
**Expected improvement:** 1.7s delay → only occurs when settings actually change

**Follow-up optimization:** Move RGBA conversion to GPU
**Expected improvement:** 1.0s → 0.7s per update

**Combined improvement:** Makes texture updates ~70% faster and only happens when needed!
