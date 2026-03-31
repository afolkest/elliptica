# Develop -> Main Merge Review

Review conducted 2026-03-30 across 7 parallel agents. 32 files changed, ~3500 insertions, ~3400 deletions, 14 commits.

---

## Pre-Merge Fixes

### Medium

- [x] **Stale `TECH_DEBT.md` references to `render.py`** — Updated to `palettes.py`, `lic.py`, `image_utils.py`.

- [x] **Dead symbols flagged by ruff** — Removed 8 unused imports/locals across 5 files.

- [ ] **Global enum params don't normalize stale saved values** — When a serialized enum value is no longer in `gf.choices`, the combo shows the first label but `pde_params` keeps the old value. Fix: sync the backing dict on rebuild, same as the LIC selector already does. `app.py` ~line 413-429.

### Low

- [ ] **Expression editor lost multiline inputs** — `expression_editor_mixin.py:174/189/204` uses single-line `add_input_text`; `main` used `multiline=True`. Restore if long expressions are expected.

- [ ] **Cache status UI not refreshed after new controls invalidate state** — New PDE param/LIC field handlers clear the cache (`app.py:502`, `app.py:541`) but don't refresh cache status indicators. Stale "View Postprocessing" affordances can linger in edit mode.

---

## Post-Merge Follow-Ups

### Architecture

- [ ] **Namespace `pde_params` per PDE** — Currently a single shared dict on `Project`. When multiple PDEs define `global_fields`, values leak across PDE switches and inactive keys pollute cache fingerprints. Match the pattern already used by `pde_bc`.

- [ ] **`field_pde.py` abstraction leak** — Relaxation still assumes Poisson-style boundary semantics (`_build_dirichlet_mask`/`_build_dirichlet_values`). The TODO at line 129 acknowledges this. New PDEs with different boundary semantics will need orchestration changes.

- [ ] **LIC extractor fallback logic** — When `lic_field_name` is stale/invalid, `field_pde.py:165` falls back to `pde.extract_lic_field` instead of the first named extractor. Latent until a PDE populates `lic_field_extractors`.

### Testing

- [ ] **Add unit tests for `place_mask_in_grid()`** — Shared by 5 call sites (`postprocess/masks.py`, `biharmonic_pde.py`, `boundary_utils.py`, `poisson_pde.py`, `pipeline.py`), no dedicated test coverage. Parameterize around: clipping, offsets, zero-size masks, smoothing, boundary positions.

- [ ] **Add regression tests for lightness expression bug fixes** — Commit `94b6ec2` fixed two stateful bugs (stale percentile cache, orphaned region `lightness_expr`) that depend on hidden state and are easy to reintroduce.

### Cleanup

- [x] **Residual `hasattr`/`getattr` checks in PDE code** — Removed from `poisson_pde.py` and `biharmonic_pde.py`; callers now always pass `SolveContext`.

- [x] **Mixin dependency docs slightly inaccurate** — Fixed `postprocessing_panel.py` docstring to describe actual init order. Removed phantom `palette_hist_height` dependency from `palette_editor_mixin.py`.

---

## Verified Clean

These areas were reviewed and found correct:

- Solver consolidation (`_solve_poisson_extended`) — numerically equivalent to ~1e-14
- `SolveContext` complete relative to old `SolveProject`
- All `from elliptica.render` imports eliminated
- All mask placement callers migrated to shared utility
- No circular import risks introduced
- Expression parser safety (AST-restricted, no injection vectors)
- GPU import rewires (purely Python, no kernel changes)
- Serialization backward compatible (`lic_field_name` defaults gracefully)
- `import elliptica` passes
- Mixin state sharing via `self` is coherent
- `TYPE_CHECKING`-guarded imports avoid circular deps in mixins
