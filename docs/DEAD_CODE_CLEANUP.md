# Dead Code Cleanup Plan

## Tier 1: Whole Features / Files (~1,200 lines)

| What | Files | Lines | Notes |
|------|-------|-------|-------|
| Eikonal equation ("icon equation") | `pde/eikonal_pde.py`, `pde/eikonal_amp.py`, registration in `pde/register.py` | ~955 | Also drops `scikit-fmm` dependency |
| `pde/geometry.py` | entire module | 116 | 4 functions, zero importers |
| `postprocess/fast.py` | entire module (3 Numba JIT functions) | 132 | Unreachable — PyTorch always available, GPU pipeline bypasses these |

## Tier 2: Dead Methods / Functions (~450 lines)

| What | File | Lines |
|------|------|-------|
| `export_image` + `_apply_postprocessing_at_resolution` | `image_export_controller.py:533-771` | ~238 |
| `build_base_rgb` (zero callers) | `postprocess/color.py` | ~70 |
| `apply_postprocess` + `PostProcessConfig` + `compute_reference_resolution` + `downsample_for_display` + `get_palette_name` | `pipeline.py` | ~60 |
| `set_region_solid_color` | `actions.py:294-318` | ~25 |
| `downsample_lic_hybrid` | `gpu/pipeline.py:106-143` | ~38 |
| `apply_highpass_gpu` | `gpu/ops.py:188-205` | ~18 |
| `DisplaySettings.to_color_params()` | `core.py:102-112` | ~11 |
| `smooth_mask_morphological` | `mask_utils.py:51-66` | 16 |
| `create_masks` + `save_mask` | `mask_utils.py:23-34` | ~12 |
| `list_pde_names` + `get_active_pde_name` | `pde/register.py:35-42` | 8 |
| `add_palette` | `render.py:570-575` | 6 |
| `_fit_canvas_to_window` | `app.py:594-600` | 7 |
| `_label_for_multiplier` | `render_modal.py:25-31` | 7 |
| `compute_amplitude` + `_compute_divergence_of_unit_gradient` | `eikonal_pde.py` (dead even if eikonal kept) | ~95 |

## Tier 3: Dead Fields, IDs, Constants

### Dead widget ID fields (postprocessing_panel.py)

All 15 are assigned but never read — the codebase migrated to string tags:

- `postprocess_clip_low_slider_id`
- `postprocess_clip_high_slider_id`
- `postprocess_brightness_slider_id`
- `postprocess_contrast_slider_id`
- `postprocess_gamma_slider_id`
- `smear_enabled_checkbox_id`
- `smear_sigma_slider_id`
- `expr_L_input_id`
- `expr_C_input_id`
- `expr_H_input_id`
- `expr_error_text_id`
- `expr_preset_combo_id`
- `lightness_expr_checkbox_id`
- `lightness_expr_input_id`
- `palette_editor_done_button_id`

### Dead instance fields (app.py)

- `mouse_handler_registry_id`
- `downsample_debounce_timer`
- `postprocess_debounce_timer`

### Dead constants

- `app.py:82-88` — `SUPERSAMPLE_CHOICES`, `SUPERSAMPLE_LABELS`, `SUPERSAMPLE_LOOKUP`, `RESOLUTION_CHOICES`, `RESOLUTION_LABELS`, `RESOLUTION_LOOKUP`
- `render_modal.py:22` — `RESOLUTION_LOOKUP`
- `defaults.py` — `DEFAULT_HIGHPASS_SIGMA_FACTOR`, `MAX_DOWNSAMPLE_SIGMA`

### Dead dataclass / fields (types.py)

- `RenderInfo` class (never instantiated)
- `Project.renders` list (always empty)
- `Project.solve_scale` property (always returns 1.0)

## Tier 4: Dead Imports

| File | Unused imports |
|------|---------------|
| `postprocessing_panel.py:6` | `Literal` |
| `render_orchestrator.py:5` | `Path` |
| `postprocess/color.py:10` | `colorize_array`, `array_to_pil` |
| `serialization.py:10` | `asdict` |
| `render.py:6` | `datetime` |
| `render.py:8` | `Project` |
| `postprocess/masks.py:4` | `distance_transform_edt` |

## Tier 5: Dead Project-Root Files

| What | Impact |
|------|--------|
| `gl_visualization.py` + `solver_benchmark.py` | ~330 lines of orphaned scripts |
| `gl_magnitude.png`, `gl_obstacles.png`, `gl_solutions.png` | ~10 MB of generated PNGs |
| `requirements.txt` | Duplicates `pyproject.toml`, diverges (includes pytest as top-level) |
| `MANIFEST.in` | Unnecessary with modern setuptools |
| `test_mask_creation.py` | Broken test — hardcoded path to nonexistent asset, tests dead functions |

## Bugs Found During Survey

1. **`image_export_controller.py:713`** — checks `boundary.smear_enabled` (old `BoundaryObject` field) instead of `RegionStyle.smear_enabled`. The UI sets the RegionStyle version, so this check is always false, meaning smear percentile pre-computation at export time never triggers.

2. **`serialization.py`** — `domain_edge_gain_strength` and `domain_edge_gain_power` from `RenderSettings` are not serialized/deserialized. They silently reset to defaults on project load.

## Vestigial (Removal Needs Care)

| What | Risk |
|------|------|
| `BoundaryObject.smear_enabled` / `smear_sigma` fields | Serialized in project files — removing breaks backward compat unless migration added |
| `_apply_lightness_expr_to_rgb` duplicate in `gpu/overlay.py` vs `gpu/postprocess.py` | Consolidation straightforward but touches GPU pipeline |
| `BOUNDARY_MASK_FIX.md` | Planning doc for unimplemented work — depends on whether still planned |

## Heavyweight Dependency Opportunities

| Dependency | Used for | Replacement |
|------------|----------|-------------|
| `matplotlib` (~80 MB) | Single call to find `DejaVuSans.ttf` font path | Bundle the font directly |
| `scikit-fmm` | Eikonal only — goes away if eikonal is nuked | — |
| `scikit-image` (~50 MB) | Single `skimage.measure` call for contours | `scipy.ndimage` or manual marching squares |
| `torchvision` (~15 MB) | Single `gaussian_blur` call | `torch.nn.functional.conv2d` with Gaussian kernel |

## Totals

- ~1,700+ lines of definite dead code
- ~10 MB of dead image assets
- 2 bugs
- 3 heavyweight dependencies potentially eliminable
