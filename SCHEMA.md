# FlowCol Project File Schema

## Overview

FlowCol projects are saved as `.flowcol` files, which are ZIP archives containing:
- `metadata.json` - All scalar/string project data
- `conductor_N_mask.png` - Conductor mask images (uint16 PNG)
- `conductor_N_interior.png` - Optional interior masks
- `conductor_N_original_mask.png` - Optional original masks (for undo/reset)
- `conductor_N_original_interior.png` - Optional original interior masks

## Schema Version

Current version: **1.0**

Schema version is incremented only for **breaking changes** (field removals, type changes, semantic changes).

Adding new optional fields does **not** require a schema version bump.

## Metadata JSON Structure

```json
{
  "schema_version": "1.0",
  "created_at": "2025-10-26T12:34:56.789012",

  "project": { ... },
  "render_settings": { ... },
  "display_settings": { ... },
  "conductors": [ ... ],
  "conductor_color_settings": { ... }
}
```

### Project

```json
"project": {
  "canvas_resolution": [width, height],        // tuple[int, int]
  "streamlength_factor": 2.0,                  // float
  "next_conductor_id": 3,                      // int
  "boundary_top": 0,                           // int (0=Dirichlet, 1=Neumann)
  "boundary_bottom": 0,                        // int
  "boundary_left": 0,                          // int
  "boundary_right": 0                          // int
}
```

**Default values:**
- `canvas_resolution`: `[1024, 1024]`
- `streamlength_factor`: `2.0`
- `next_conductor_id`: `0`
- `boundary_*`: `0` (Dirichlet)

### Render Settings

```json
"render_settings": {
  "multiplier": 1.0,                           // float - render resolution multiplier
  "supersample": 1.0,                          // float - antialiasing supersample factor
  "num_passes": 2,                             // int - LIC integration passes
  "margin": 0.1,                               // float - padding margin (fraction)
  "noise_seed": 0,                             // int - random seed for noise texture
  "noise_sigma": 0.5                           // float - noise blur sigma
}
```

**Default values:**
- `multiplier`: `1.0`
- `supersample`: `1.0`
- `num_passes`: `2`
- `margin`: `0.1`
- `noise_seed`: `0`
- `noise_sigma`: `0.5`

### Display Settings

```json
"display_settings": {
  "downsample_sigma": 0.6,                     // float - downsample blur sigma
  "clip_percent": 0.01,                        // float - histogram clipping (0-1)
  "contrast": 1.0,                             // float - contrast multiplier
  "gamma": 1.0,                                // float - gamma correction
  "color_enabled": false,                      // bool - enable colorization
  "palette": "Viridis",                        // str - color palette name
  "edge_blur_sigma": 0.0,                      // float - anisotropic edge blur sigma (physical units)
  "edge_blur_falloff": 0.015,                  // float - edge blur falloff distance (physical units)
  "edge_blur_strength": 1.0                    // float - edge blur strength (0-1)
}
```

**Default values:**
- `downsample_sigma`: `0.6`
- `clip_percent`: `0.01`
- `contrast`: `1.0`
- `gamma`: `1.0`
- `color_enabled`: `false`
- `palette`: `"Viridis"`
- `edge_blur_sigma`: `0.0` (disabled)
- `edge_blur_falloff`: `0.015`
- `edge_blur_strength`: `1.0`

### Conductors

```json
"conductors": [
  {
    "voltage": 0.5,                            // float - conductor voltage (0-1)
    "position": [x, y],                        // tuple[float, float] - position in pixels
    "scale_factor": 1.0,                       // float - current scale relative to original
    "blur_sigma": 0.0,                         // float - Gaussian blur sigma
    "blur_is_fractional": false,               // bool - if true, blur_sigma is fraction of canvas
    "smear_enabled": false,                    // bool - enable texture smearing inside conductor
    "smear_sigma": 2.0,                        // float - smear blur sigma (pixels)
    "smear_feather": 3.0,                      // float - smear feather distance (pixels)
    "id": 0,                                   // int or null - unique conductor ID

    "masks": {
      "mask": {
        "file": "conductor_0_mask.png",        // str - filename in ZIP
        "shape": [height, width],              // tuple[int, int] - array shape
        "encoding": "uint16_png"               // str - encoding format
      },
      "interior_mask": {                       // null if not present
        "file": "conductor_0_interior.png",
        "shape": [height, width],
        "encoding": "uint16_png"
      },
      "original_mask": null,                   // null if not present
      "original_interior_mask": null           // null if not present
    }
  }
]
```

**Default values:**
- `voltage`: `0.5`
- `position`: `[0.0, 0.0]`
- `scale_factor`: `1.0`
- `blur_sigma`: `0.0`
- `blur_is_fractional`: `false`
- `smear_enabled`: `false`
- `smear_sigma`: `2.0`
- `smear_feather`: `3.0`
- `id`: `null`

**Required fields:**
- `mask` (always present)

**Optional fields:**
- `interior_mask`, `original_mask`, `original_interior_mask` (can be `null`)

### Conductor Color Settings

```json
"conductor_color_settings": {
  "0": {                                       // Keyed by conductor ID (as string in JSON)
    "surface": {
      "enabled": false,                        // bool - enable coloring for this region
      "use_palette": true,                     // bool - true=palette, false=solid color
      "palette": "Viridis",                    // str - palette name (if use_palette=true)
      "solid_color": [r, g, b]                 // tuple[float, float, float] - RGB [0,1]
    },
    "interior": {
      "enabled": false,
      "use_palette": true,
      "palette": "Viridis",
      "solid_color": [r, g, b]
    }
  }
}
```

**Default values (per region):**
- `enabled`: `false`
- `use_palette`: `true`
- `palette`: `"Viridis"`
- `solid_color`: `[0.5, 0.5, 0.5]`

## Mask Encoding

Masks are stored as **uint16 PNG** files (16-bit grayscale):
- Internal format: `float32` arrays with values in `[0, 1]`
- Saved format: `uint16` with values in `[0, 65535]`
- Conversion: `uint16_value = clip(float32_value, 0, 1) * 65535`
- Precision: ~4.8 decimal digits (vs ~7 for float32)

This provides excellent precision for masks while keeping file sizes small with PNG compression.

## Backward Compatibility

**Loading old files with new code:**
- Missing fields automatically use default values via `.get(key, default)`
- All dataclass fields have defaults (except required numpy arrays)
- Old files will load successfully and get new features disabled by default

**Adding new features:**
- Add field to dataclass with sensible default
- Add field to serialization functions with `.get(key, default)`
- No schema version bump needed
- Document new field in this file

**Breaking changes (require schema version bump):**
- Removing a field
- Renaming a field
- Changing field type or semantics
- Add migration code in `load_project()` to convert old → new format

## Example: Adding a New Feature

**Step 1:** Add field to dataclass (flowcol/types.py)
```python
@dataclass
class Conductor:
    # ... existing fields ...
    animation_enabled: bool = False  # NEW
```

**Step 2:** Add to serialization (flowcol/serialization.py)
```python
def _conductor_to_dict(conductor, index):
    return {
        # ... existing fields ...
        'animation_enabled': conductor.animation_enabled,  # NEW
    }

def _dict_to_conductor(data, masks):
    return Conductor(
        # ... existing fields ...
        animation_enabled=data.get('animation_enabled', False),  # NEW
    )
```

**Step 3:** Document in SCHEMA.md (this file)
```json
"animation_enabled": false,  // bool - enable animation
```

**Step 4:** Add test to verify backward compatibility

Old files will load with `animation_enabled=False` automatically!

## File Size

Typical project with 3 conductors at 1024×1024:
- Metadata JSON: ~5-10 KB
- Each mask PNG: ~50-500 KB (depends on complexity)
- Total: **~200 KB - 2 MB** (very reasonable)

PNG compression is effective for masks because they're typically smooth or binary.

## Performance

- **Save:** ~150-600ms for 3 conductors (dominated by PNG compression)
- **Load:** ~120-360ms for 3 conductors (dominated by PNG decompression)

Both operations are subsecond for typical projects and only happen occasionally.
