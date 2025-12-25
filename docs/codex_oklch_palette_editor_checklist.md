# OKLCH Palette Editor Implementation Checklist

## Data model and migration
- Add a versioned palette schema in `palettes_user.json` (space + stops).
- Implement migration for legacy palette format with a `.bak` backup.
- Ensure load/save round-trips preserve OKLCH metadata.

## OKLCH palette logic
- Move OKLCH stop + LUT logic from `tools/oklch_palette_preview.py` into a reusable module.
- Preserve hue wrap interpolation, relative/absolute chroma, and `interp_mix`.
- Use `gamut_map_to_srgb(..., method="compress")` for LUT generation.
- Add a small LUT cache keyed by stops + mode to avoid recompute during drags.

## Runtime palette registry
- Build `PALETTE_LUTS` from the new schema in `elliptica/render.py`.
- Keep RGB palette interpolation behavior unchanged.
- Ensure palette add/delete/rename updates the schema and runtime LUTs.

## UI integration
- Embed the editor UI into the existing DPG context.
- Target the active palette (global or selected region).
- Keep editor textures (gradient, slice, lightness bar, swatch) updating immediately.
- Mutate palette state under `app.state_lock` to keep async pipeline snapshots consistent.

## Display updates
- Route palette updates through `elliptica/ui/dpg/display_pipeline_controller.py`.
- Add throttling (80-120ms) to coalesce full-res refreshes.
- Ensure updates propagate to all regions using the palette.

## Colormap registry
- Rebuild DPG colormap registry from the new schema after palette changes.
- Keep palette picker entries in sync with the palette store.

## Manual validation
- Create new palette -> default name assigned and persisted.
- Edit palette -> all regions using it update.
- Restart -> OKLCH palette remains editable.
- Legacy `palettes_user.json` migrates to new schema and writes `.bak`.
