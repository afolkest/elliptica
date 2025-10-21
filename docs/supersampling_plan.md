## Supersampling & Post-processing Enhancements

### Core Goals
- Support rendering at a user-selectable supersample factor (≥1.0) per render.
- Match Poisson solve resolution to LIC supersample resolution.
- Treat streamlength and high-pass sigma as physical quantities that scale with compute resolution.
- Preserve a cached copy of the supersampled LIC to enable cheap tuning of downsample parameters post-render.
- Expose noise RNG seed in the render menu and keep renders deterministic for identical settings.

### Detailed Tasks

1. **UI Updates**
   - Extend render modal with:
     - Supersample factor selector (e.g., 1.0, 1.5, 2.0, 3.0).
     - Noise seed numeric input (default 0).
   - Show warning/disable render if `(render_multiplier × supersample × canvas_dim)` exceeds `MAX_RENDER_DIM`.
   - Persist user choices for the duration of the session; defaults reset on app start.

2. **Data Structures**
   - Augment `UIState` with supersample factor and seed fields.
   - Track cached supersampled LIC array (e.g., `state.highres_render_data`).

3. **Compute Pipeline**
   - Compute field on `canvas_resolution × supersample` grid.
     - Update `compute_field` to accept explicit resolution and return both field and potential if needed.
   - Run LIC on the supersampled grid using scaled streamlength:
     - `pixel_streamlength = base_streamlength × render_multiplier × supersample`.
     - Noise seeded using user-provided seed.
   - Apply post-processing using sigma scaled by compute resolution:
     - `sigma_pixels = hp_sigma_factor × compute_resolution`.
   - Store supersampled LIC in state for later downsampling tweaks.

4. **Downsampling**
   - Implement `downsample_lic(highres_array, supersample_factor, sigma_factor)`:
     - Gaussian blur (`sigma = 0.6 × supersample` by default).
     - Resize to canvas resolution with bilinear interpolation.
   - Allow post-processing panel to re-run downsampling using cached high-res data without recomputing LIC.

5. **Post-processing UI**
   - Add slider/input for downsample blur factor (optional range 0.3–1.0 × supersample).
   - Ensure HP+CLAHE operates on cached high-res data, not yet-downsampled array.

6. **Validation & Tests**
   - Update parity test to cover supersampled case (e.g., supersample=2×) against `compute_lic_with_postprocessing` + manual downsample.
   - Add tests for sigma/streamlength scaling.

7. **Performance Guards**
   - Estimate memory footprint for supersampled arrays and warn/log if above threshold.
   - Possibly allow user override for Poisson solve resolution (e.g., toggle to reuse base field for fast previews).

### Open Questions
- Do we expose downsample blur factor immediately or ship with a fixed default and add UI later?
- Should supersample defaults persist between app runs (config file) or reset every launch?

