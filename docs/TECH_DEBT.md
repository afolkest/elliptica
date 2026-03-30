# Elliptica Technical Debt & Architectural Issues

## P0 — High Priority

### Add locking around palette module state
`set_palette_spec()` in `render.py:532` mutates three module-level dicts (`_RUNTIME_PALETTE_SPECS`, `_RUNTIME_PALETTES`, `PALETTE_LUTS`) without locking. Callers hold `state_lock`, but that protects `AppState`, not the palette module state. Other code paths can read palette data concurrently. Either add a dedicated lock or move palette state into `AppState`.

---

## P1 — Important

### Add tests for PDE solvers and render pipeline
The most numerically sensitive code has zero test coverage. Bugs can produce plausible-looking but incorrect output. Priority targets:
- `poisson.py` (Poisson solver)
- `pde/biharmonic_pde.py` (Biharmonic solver)
- `pipeline.py` (render orchestration)
- `render.py` LIC computation path

### Extract remaining `PostprocessingPanel` responsibilities
Still 973 lines post-mixin-extraction. Contains slider callbacks, region toggle logic, lightness expression management, smear callbacks, and a 140-line `update_context_ui()` method. Candidates for further extraction into mixins or helper classes.

### DRY up duplicated code
- `_normalize_unit` is copy-pasted between `render.py:681` and `histogram_mixin.py:54`
- `_apply_display_transforms` and `colorize_array` share nearly identical percentile-clip/contrast/brightness/gamma logic
- Palette popup building is duplicated across `postprocessing_panel.py` (lines 415, 509) and `palette_editor_mixin.py:1653`

---

## P2 — Moderate

### Implement undo/redo
Noted as a TODO in `boundary_controls_panel.py:424`. The `StateManager` architecture already supports this well — state mutations flow through `actions.py` pure functions and `StateManager` notifications. An undo stack could be layered on top without major refactoring.

### Formalize mixin contracts with Protocol classes
`ExpressionEditorMixin` expects ~6 attributes/methods from `PaletteEditorMixin` (`palette_editor_active`, `palette_editor_dirty`, `palette_editor_persist_dirty`, `_apply_palette_editor_refresh()`, `_finalize_palette_editor_colormaps()`, `_set_palette_editor_state()`). These are documented only in docstrings, not enforced by any Protocol or ABC. Adding protocols would catch integration errors at type-check time.

### Move hardcoded palette data to data files
`render.py:11-236` contains ~220 lines of numpy arrays defining color palettes. This data should live in JSON or TOML files loaded at startup, not in Python source.

### Convert `BoundaryObject` to a dataclass
`BoundaryObject` (`types.py:9`) uses a manual `__init__` while every other data type (`Project`, `RenderSettings`, `DisplaySettings`, `RegionStyle`, etc.) uses `@dataclass`. No apparent reason for the inconsistency.

### Fix palette soft-delete growth
`delete_palette` (`render.py:559`) stores `{"deleted": True}` rather than removing the entry. The palettes JSON grows monotonically. Deleted palettes are filtered at load time. Should either hard-delete or compact periodically.

---

## P3 — Low Priority / Housekeeping

### Add CI/CD
No CI configuration exists. At minimum, add a GitHub Actions workflow that runs the test suite.

### Fix README inaccuracies
README references `pip install -r requirements.txt` but no `requirements.txt` exists. Should reference `pip install .` or `uv sync` instead.

### Clean up loose scripts at repo root
`gl_visualization.py` and `solver_benchmark.py` appear to be standalone dev scripts, not part of the package. Should be moved to a `scripts/` directory or removed.

### Fix assets symlink
`assets -> ../elliptica/assets` uses a relative path that breaks when the repo is cloned to unexpected locations.

### Reduce hub-and-spoke controller coupling
Every UI controller reaches into every other through `self.app`. No enforced isolation. Consider passing only the specific dependencies each controller needs rather than the full app reference.

---

## Things That Are Good (Don't Touch)

- **State management** (`StateManager`, `AppState`, `actions.py`) — well-designed, type-safe, with debouncing and pub-sub notifications
- **Colorspace module** (`colorspace/`) — clean, well-tested OKLCH implementation
- **Expression module** (`expr/`) — clean parser/compiler with good error hierarchy and thorough tests
- **PDE registry pattern** (`pde/registry.py`) — clean extensibility for new PDE types
- **GPU fallback architecture** (`gpu/`) — graceful degradation from CUDA → MPS → CPU
- **Serialization** (`serialization.py`) — atomic writes, schema versioning, ZIP-based format
- **Recent mixin extraction** — `HistogramMixin`, `PaletteEditorMixin`, `ExpressionEditorMixin` were the right call
