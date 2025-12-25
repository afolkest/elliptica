# OKLCH Palette Editor Integration Spec

## Purpose
Integrate the OKLCH palette editor (prototype in `tools/oklch_palette_preview.py`) into the main Elliptica UI so palettes are first-class, persistent, and re-editable, while keeping the UI responsive at high resolution (2k+).

## Decisions (must preserve)
- Palettes are non-transient. Creating a new palette immediately persists it with a default unique name.
- Editing a palette updates all regions that reference that palette (no auto-duplication).
- The editor always edits the currently active highlighted palette. If a region palette is selected, that is the target.
- Default naming scheme: `user_palette_1`, `user_palette_2`, ...
- OKLCH stop metadata is stored so palettes are truly re-editable across sessions.
- Display updates are throttled full-resolution refreshes (no low-res preview pipeline for now).

## Data model and persistence
Use a unified, versioned palette schema stored in `palettes_user.json` (replace the legacy format on load).

Example schema:
```json
{
  "version": 2,
  "palettes": {
    "Ink Wash": {
      "space": "rgb",
      "stops": [
        {"pos": 0.0, "r": 0.06, "g": 0.06, "b": 0.06},
        {"pos": 1.0, "r": 0.98, "g": 0.98, "b": 0.98}
      ]
    },
    "user_palette_1": {
      "space": "oklch",
      "relative_chroma": true,
      "interp_mix": 1.0,
      "stops": [
        {"pos": 0.0, "L": 0.20, "C": 0.08, "H": 270.0},
        {"pos": 1.0, "L": 0.92, "C": 0.06, "H": 60.0}
      ]
    }
  }
}
```

Notes:
- `space` is required and drives LUT generation.
- `relative_chroma` and `interp_mix` are required for OKLCH palettes.
- RGB palettes store explicit stop positions for future extensibility.

## Migration
When loading `palettes_user.json`:
- If the old format is detected (`{name: [[r,g,b], ...]}`), convert to the new schema.
- Use evenly spaced `pos` values for legacy RGB palettes.
- Write a one-time `.bak` before rewriting the file in the new schema.

Scope note:
- Only the palette library file (`palettes_user.json`) is migrated. Project files store palette names only and do not require migration as long as names are preserved.

## OKLCH palette model
Extract the palette logic from `tools/oklch_palette_preview.py` into a reusable module (suggested: `elliptica/colorspace/oklch_palette.py`).

Model requirements:
- Stop data: `pos`, `L`, `C`, `H`.
- Stored flags: `relative_chroma`, `interp_mix`.
- LUT generation:
  - Hue interpolation uses shortest wrap-around path.
  - Chroma interpolation supports absolute, relative, and mix.
  - Gamut mapping uses `gamut_map_to_srgb(..., method="compress")`.
  - Vectorize LUT generation (no per-sample loops).

## Runtime LUT generation
`elliptica/render.py` should build `PALETTE_LUTS` from the stored palette specs:
- `space: "rgb"` uses linear interpolation of RGB stops (current behavior).
- `space: "oklch"` uses the extracted OKLCH model to produce a LUT.

Implementation notes:
- Keep a small LUT cache keyed by (stops + relative_chroma + interp_mix) to avoid recompute during drags.
- LUT generation is fast, but do not bypass the refresh throttle because the bottleneck is GPU -> CPU texture upload.
- When mutating palette state from UI callbacks, take the `app.state_lock` to avoid racing the async display pipeline.

## UI integration
Integrate the editor UI into the existing DearPyGui context (no new viewport).
Likely integration point: `elliptica/ui/dpg/postprocessing_panel.py`.

Editor requirements:
- Small editor textures (gradient, CH slice, lightness bar, swatch) update immediately.
- The main render image is updated via the display pipeline (no direct full-res updates in the editor).
- Active palette target follows the global palette or selected region palette.

## Display update policy
DPG only accepts CPU texture data, so each refresh requires GPU -> CPU download plus `dpg.set_value`. Full-res continuous updates are too expensive.

Policy:
- Use the async postprocessing pipeline in `elliptica/ui/dpg/display_pipeline_controller.py`.
- Throttle refresh requests (recommend 80-120ms cadence).
- Coalesce updates: if a job is running, mark pending and refresh when the job completes.

Throttle note:
- Implement a simple time gate (e.g., `last_refresh_ts`) in the editor controller.
- On slider/drag events, only request a refresh if `now - last_refresh_ts >= throttle_ms`.
- If a request is skipped due to throttling, set a `pending_refresh` flag so the next allowed tick triggers an update.
- Always trigger a refresh on mouse release to ensure the final state is displayed.

## Palette naming
Default name for a new palette:
- `user_palette_1`, `user_palette_2`, etc (smallest available integer).

## Shared palette behavior
No duplication. Editing a palette updates all regions using that palette. This is intended.

## Files likely to change
- `elliptica/render.py` (schema load/save, LUT build from spec).
- `elliptica/colorspace/oklch_palette.py` (new module with palette math).
- `elliptica/ui/dpg/postprocessing_panel.py` (editor UI integration).
- `elliptica/ui/dpg/texture_manager.py` (colormap registry rebuild from new schema).
- `tools/oklch_palette_preview.py` (may remain as prototype, not used by main UI).

## Non-goals
- No GPU-native DPG texture interop.
- No low-res while dragging pipeline in this iteration.
