# StateManager Migration Plan

## Background

The StateManager (built in the previous refactor, Milestones 1-6) centralizes display-setting mutations with debounce, subscriber notifications, and per-frame refresh flags. It currently manages 12 `StateKey` values covering `DisplaySettings`, `COLOR_CONFIG`, and per-boundary `REGION_STYLE`.

However, the migration was left incomplete in two ways:

1. **Dual paths**: Several display-setting callbacks still bypass StateManager, mutating state via `actions.*` functions and calling `refresh_display()` manually.
2. **Limited scope**: Interaction state (`view_mode`, `selected_indices`, `selected_region_type`) and project/render mutations are entirely outside StateManager. Canvas redraw is triggered by 28 scattered `mark_dirty()` calls.

This plan addresses both.

---

## Current State: The Dual-Path Problem

These postprocessing_panel.py callbacks bypass StateManager:

| Callback | What it does | Lines |
|----------|-------------|-------|
| `on_global_grayscale` | `actions.set_color_enabled(state, False)` + manual `refresh_display()` | ~2693 |
| `on_global_palette_button` | `actions.set_color_enabled` + `actions.set_palette` + manual refresh | ~2703-2708 |
| `on_region_palette_button` | `actions.set_region_palette` + B/C/G setters + manual refresh | ~2726-2776 |
| `on_region_use_global` | `actions.set_region_style_enabled(False)` + manual refresh | ~2749 |
| Palette editor flush | `state.invalidate_base_rgb()` + manual `refresh_display()` | ~1202 |

These all mutate the exact same state that StateManager manages, but through a different pathway, meaning:
- No subscriber notifications fire
- No debounce/coalescing
- The caller must remember to call `refresh_display()` manually (fragile)

Additionally, `actions.py` contains 7 display-setting functions (~130 lines) that duplicate what StateManager already does:
- `set_color_enabled()`, `set_palette()`
- `set_region_style_enabled()`, `set_region_palette()`
- `set_region_brightness()`, `set_region_contrast()`, `set_region_gamma()`

---

## Phase 1: Consolidate Display-Setting Dual Paths

**Goal**: Every display-setting mutation goes through `StateManager.update()`. Delete the redundant `actions.*` display functions.

### 1a. Redirect palette button callbacks to StateManager

**File**: `postprocessing_panel.py`

Replace direct `actions.*` calls with `state_manager.update()`:

```python
# BEFORE (on_global_grayscale):
with self.app.state_lock:
    actions.set_color_enabled(self.app.state, False)
self.app.display_pipeline.refresh_display()

# AFTER:
self.app.state_manager.update(StateKey.COLOR_ENABLED, False)
```

```python
# BEFORE (on_global_palette_button):
with self.app.state_lock:
    actions.set_color_enabled(self.app.state, True)
    actions.set_palette(self.app.state, palette_name)
self.app.display_pipeline.refresh_display()

# AFTER:
self.app.state_manager.update(StateKey.COLOR_ENABLED, True)
self.app.state_manager.update(StateKey.PALETTE, palette_name)
```

```python
# BEFORE (on_region_palette_button):
with self.app.state_lock:
    actions.set_region_palette(self.app.state, boundary_id, region, palette)
    actions.set_region_brightness(self.app.state, boundary_id, region, brightness)
    actions.set_region_contrast(self.app.state, boundary_id, region, contrast)
    actions.set_region_gamma(self.app.state, boundary_id, region, gamma)
self.app.display_pipeline.refresh_display()

# AFTER:
ctx = (boundary_id, region)
self.app.state_manager.update(StateKey.REGION_STYLE, {
    "enabled": True, "use_palette": True, "palette": palette,
    "brightness": brightness, "contrast": contrast, "gamma": gamma,
}, context=ctx)
```

The manual `refresh_display()` calls are removed -- the main loop picks up `needs_refresh()` on the next frame.

### 1b. Redirect palette editor flush

The palette editor's `_flush_palette_editor()` currently calls `state.invalidate_base_rgb()` + `display_pipeline.refresh_display()` directly. This should be replaced with a StateManager update that triggers the same invalidation through the normal path.

The palette editor modifies the LUT of the *current* palette in-place and then needs a display refresh with cache invalidation. The cleanest approach: call `state_manager.update(StateKey.PALETTE, current_palette_name)` which sets the same value but triggers the invalidation flag (since PALETTE is in `_INVALIDATE_KEYS`). StateManager doesn't short-circuit same-value updates, so this works.

### 1c. Delete dead `actions.*` display functions

After 1a-1b, these become dead code (no remaining callers):
- `actions.set_color_enabled()`
- `actions.set_palette()`
- `actions.set_region_style_enabled()`
- `actions.set_region_palette()`
- `actions.set_region_brightness()`
- `actions.set_region_contrast()`
- `actions.set_region_gamma()`

Delete all 7 functions (~130 lines). The remaining `actions.py` functions (boundary/render/project mutations) stay.

### 1d. Verify: No direct `display_pipeline.refresh_display()` calls remain in postprocessing_panel.py

After this phase, the only callers of `refresh_display()` should be:
- Main loop (`app.py`) -- triggered by StateManager flags
- `render_orchestrator.py` -- after render completion
- `cache_management_panel.py` -- "View Postprocessing" button
- `file_io_controller.py` -- after loading a project (bulk state replacement)

### Verification
- All 101 tests pass
- Import check: `from elliptica.app.actions import add_boundary` still works
- Manual test: palette buttons, grayscale toggle, region palette buttons all update display correctly
- Manual test: slider drags still debounce correctly

---

## Phase 2: Extend StateManager to Interaction State

**Goal**: Route `view_mode`, `selected_indices`, and `selected_region_type` through StateManager, enabling subscriber-driven UI updates.

### 2a. Add new StateKeys

```python
class StateKey(enum.Enum):
    # ... existing keys ...

    # Interaction state
    VIEW_MODE = "view_mode"              # "edit" or "render"
    SELECTED_INDICES = "selected_indices" # set[int]
    SELECTED_REGION_TYPE = "selected_region_type"  # "surface" or "interior"
```

These are NOT display settings, so they need new handling in `_apply_update()`:
- `VIEW_MODE`: Sets `self._state.view_mode = value`
- `SELECTED_INDICES`: Sets `self._state.selected_indices = value` (always a set)
- `SELECTED_REGION_TYPE`: Sets `self._state.selected_region_type = value`

Classification:
- These should NOT trigger display refresh (`_REFRESH_KEYS` stays unchanged)
- These should NOT trigger invalidation
- These DO trigger subscriber notifications (which is the whole point)

### 2b. Wire canvas_dirty via subscriber

Register a subscriber in `CanvasRenderer.__init__()`:

```python
def __init__(self, app):
    self.app = app
    self.canvas_dirty = True
    # ...
    # Subscribe to interaction state changes that require canvas redraw
    for key in (StateKey.VIEW_MODE, StateKey.SELECTED_INDICES, StateKey.SELECTED_REGION_TYPE):
        app.state_manager.subscribe(key, self._on_interaction_changed)

def _on_interaction_changed(self, key, value, context):
    self.mark_dirty()
```

This eliminates many (not all) manual `mark_dirty()` calls. The remaining ones are for:
- Boundary drag/move/scale (from canvas_controller, still going through `actions.*`)
- Box selection visual feedback (not state changes, just drawing)
- Display pipeline completion (texture update)
- File I/O (project load)

### 2c. Migrate view_mode mutations

Currently `view_mode` is set directly in 5 places:

| File | Line | Context |
|------|------|---------|
| `app.py` | ~589 | "Back to Edit" button |
| `file_io_controller.py` | ~150 | After loading boundary |
| `file_io_controller.py` | ~196 | New project |
| `file_io_controller.py` | ~440 | Load project |
| `render_orchestrator.py` | ~245 | Render completion |
| `render_orchestrator.py` | ~283 | Render success callback |
| `cache_management_panel.py` | ~137 | Clear cache |
| `cache_management_panel.py` | ~153 | View postprocessing |
| `shape_dialog.py` | ~177 | After adding shape |

Replace each with:
```python
self.app.state_manager.update(StateKey.VIEW_MODE, "edit")
```

**Important**: The render_orchestrator sets `view_mode` from a background thread (under the lock). StateManager's `_apply_update` acquires the lock internally, so the caller should NOT hold the lock when calling `update()`. This means we need to restructure the render_orchestrator's completion handler to release the lock before calling StateManager, or have StateManager detect re-entrant lock acquisition (it's an RLock, so this is safe).

Since `state_lock` is an `RLock`, calling `state_manager.update()` while already holding the lock is safe -- the lock is reentrant. However, the subscriber notification fires outside the lock, which is the desired behavior. So no restructuring needed.

### 2d. Migrate selection mutations

Selection is modified in `canvas_controller.py` (8+ sites) via:
- `state.set_selected(idx)`
- `state.toggle_selected(idx)`
- `state.clear_selection()`
- `state.selected_indices = new_set`
- `state.selected_indices.update(hits)`

Per Decision 1, StateManager gets thin convenience methods that read current state, compute the new set, and call `update()` internally. Call sites become:

```python
# BEFORE:
with self.app.state_lock:
    self.app.state.set_selected(hit_idx)

# AFTER:
self.app.state_manager.set_selected(hit_idx)
```

```python
# BEFORE:
with self.app.state_lock:
    self.app.state.toggle_selected(hit_idx)

# AFTER:
self.app.state_manager.toggle_selected(hit_idx)
```

```python
# BEFORE:
with self.app.state_lock:
    self.app.state.clear_selection()

# AFTER:
self.app.state_manager.clear_selection()
```

For bulk updates (box select, post-duplicate index adjustment):
```python
self.app.state_manager.update(StateKey.SELECTED_INDICES, new_set)
```

### 2e. Migrate selected_region_type mutations

Currently set directly in `canvas_controller.py` and `postprocessing_panel.py`. Replace with:
```python
self.app.state_manager.update(StateKey.SELECTED_REGION_TYPE, "surface")
```

### 2f. Wire subscriber for UI panel updates

The postprocessing panel's `update_context_ui()` and `_on_boundary_selection_changed()` are currently called manually after selection changes. With subscribers:

```python
# In PostprocessingPanel.__init__():
app.state_manager.subscribe(StateKey.SELECTED_INDICES, self._on_selection_changed)
app.state_manager.subscribe(StateKey.SELECTED_REGION_TYPE, self._on_selection_changed)

def _on_selection_changed(self, key, value, context):
    self._on_boundary_selection_changed()
```

Similarly for boundary_controls_panel:
```python
app.state_manager.subscribe(StateKey.SELECTED_INDICES, self._on_selection_subscriber)

def _on_selection_subscriber(self, key, value, context):
    self.rebuild_controls()
    self.update_slider_labels()
```

And for `_update_control_visibility()` in app.py:
```python
app.state_manager.subscribe(StateKey.VIEW_MODE, self._on_view_mode_changed)

def _on_view_mode_changed(self, key, value, context):
    self._update_control_visibility()
    self.canvas_renderer.invalidate_selection_contour()
```

### Verification
- All tests pass
- Manual test: click boundary in edit mode -> selection outline appears, panel updates
- Manual test: shift-click for multi-select -> controls rebuild correctly
- Manual test: switch to render mode -> controls hide, canvas redraws
- Manual test: render completion -> view switches to render mode, controls update
- Manual test: load project -> all UI syncs correctly
- Audit: count remaining manual `mark_dirty()` calls (should be reduced from 28)

---

## Phase 3: Subscriber-Driven canvas_dirty Consolidation

**Goal**: Replace as many manual `mark_dirty()` calls as possible with subscriber-driven auto-marking.

### Current mark_dirty() call sites (28 total)

After Phase 2, the subscriber for interaction state handles ~12 of these. Remaining:

| File | Context | Can subscriber handle? |
|------|---------|----------------------|
| `canvas_controller.py` | Boundary drag (continuous) | No -- still direct `actions.move_boundary()` |
| `canvas_controller.py` | Box select visual | No -- ephemeral drawing state, not in AppState |
| `canvas_controller.py` | Scale boundary | No -- still direct `actions.scale_boundary()` |
| `display_pipeline_controller.py` | Texture update complete | Yes -- subscribe to a "texture ready" signal |
| `file_io_controller.py` | Project load | Yes -- subscriber on VIEW_MODE / SELECTED_INDICES handles it |
| `boundary_controls_panel.py` | Param slider change | No -- still direct boundary mutation |

**Assessment**: About half the mark_dirty calls are eliminated by Phase 2 subscribers. The remaining ones are for direct boundary mutations (`actions.*`) and ephemeral UI state (box select, drag visual feedback). These would only be eliminated by extending StateManager to cover project mutations, which is out of scope (see Phase 4 discussion).

### 3a. Add display refresh -> canvas dirty link

When the display pipeline finishes a postprocess job and updates the texture, the canvas needs redrawing. Currently `display_pipeline_controller.py` calls `mark_dirty()` explicitly. Instead, subscribe the canvas renderer to StateManager's display refresh signal.

Actually, this is tricky because the texture update happens in `poll()` (main loop), not through StateManager. The cleanest approach: keep the explicit `mark_dirty()` in `display_pipeline_controller.poll()` since it's a main-loop polling result, not a state change.

**Verdict**: Phase 3 is mostly handled by Phase 2's subscribers. The remaining manual `mark_dirty()` calls are legitimate (boundary drags, box select visuals, texture updates) and should stay.

---

## Implementation Order

```
Phase 1a  Redirect palette button callbacks          (~1 hour)
Phase 1b  Redirect palette editor flush              (~30 min)
Phase 1c  Delete dead actions.* display functions     (~30 min)
Phase 1d  Verify no stray refresh_display() calls     (~15 min)
     --- commit: "Consolidate display settings through StateManager" ---

Phase 2a  Add VIEW_MODE, SELECTED_*, REGION_TYPE keys (~30 min)
Phase 2b  Wire canvas_dirty subscriber                (~30 min)
Phase 2c  Migrate view_mode mutations (9 sites)       (~1 hour)
Phase 2d  Migrate selection mutations (8+ sites)      (~1 hour)
Phase 2e  Migrate selected_region_type mutations       (~30 min)
     --- commit: "Route interaction state through StateManager" ---

Phase 2f  Wire subscribers for panel/control updates   (~1-2 hours)
     --- commit: "Replace manual UI sync with StateManager subscribers" ---

Phase 3   Audit remaining mark_dirty() calls           (~30 min)
     --- commit: "Remove redundant mark_dirty calls" (if any) ---
```

---

## Decisions

### Decision 1: Selection convenience methods

**Decision**: Add thin convenience methods to StateManager (`set_selected`, `toggle_selected`, `clear_selection`).

The alternative -- callers read state under the lock, compute the new set, then pass it to `update()` -- creates a double-write problem (state is mutated directly by the AppState helper, then again by StateManager). Convenience methods that read current state, compute the new set, and call `update()` internally keep mutation in one place and make call sites clean. The cost is 3 small methods.

```python
def set_selected(self, idx: int) -> None:
    with self._lock:
        new_set = set()
        if 0 <= idx < len(self._state.project.boundary_objects):
            new_set.add(idx)
    self.update(StateKey.SELECTED_INDICES, new_set)

def toggle_selected(self, idx: int) -> None:
    with self._lock:
        if not (0 <= idx < len(self._state.project.boundary_objects)):
            return
        current = set(self._state.selected_indices)
    if idx in current:
        current.discard(idx)
    else:
        current.add(idx)
    self.update(StateKey.SELECTED_INDICES, current)

def clear_selection(self) -> None:
    self.update(StateKey.SELECTED_INDICES, set())
```

### Decision 2: Subscriber re-entrancy guard

**Decision**: Start without a guard. Rely on DPG's natural circuit-breaker.

DPG callbacks already check whether the value actually changed before doing work, and our existing callbacks follow this pattern. If a subscriber sets a slider to the value it already has, DPG won't fire the callback. This prevents loops naturally. Adding a `_notifying` flag preemptively would add complexity for a problem that likely won't materialize. If we discover a loop during testing, it's a 5-line addition.

### Decision 3: `flush_pending()` on selection/region change

**Decision**: Keep `flush_pending()` in the subscriber callbacks for selection/region changes.

The logic is: "when the user switches context, fire any pending display refreshes for the old context before processing the new one." This is inherent to the selection-change event, so it belongs in the subscriber that handles it. Concretely, the subscriber for `SELECTED_INDICES` / `SELECTED_REGION_TYPE` calls `flush_pending()` first, then proceeds with panel updates. This preserves the current behavior with no ordering surprises.

---

## Tests to Add

### Unit tests (test_state_manager.py)

1. **VIEW_MODE updates**: Verify `state.view_mode` is set correctly
2. **SELECTED_INDICES updates**: Verify `state.selected_indices` is set correctly
3. **SELECTED_REGION_TYPE updates**: Verify `state.selected_region_type` is set correctly
4. **Interaction keys don't set refresh flags**: These keys should NOT trigger `needs_refresh()`
5. **Interaction key subscribers fire**: Verify subscriber callbacks are called for new keys
6. **Convenience methods** (if Decision 1 goes with methods): Test `set_selected()`, `toggle_selected()`, `clear_selection()`

### Integration tests

7. **Palette button -> StateManager path**: Verify palette buttons change `display_settings.palette` and trigger refresh
8. **Region palette -> StateManager path**: Verify region palette updates go through REGION_STYLE
9. **No stray refresh_display() calls**: Grep test that postprocessing_panel.py has zero direct `refresh_display()` calls (after Phase 1)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Subscriber re-entrancy loop | Low | High (UI freeze) | DPG's "did value change?" natural guard; add guard flag if needed |
| render_orchestrator background thread + StateManager | Low | Medium | RLock is reentrant; subscriber notification is outside lock |
| Missed mark_dirty() causing stale canvas | Medium | Low (visual only, next interaction fixes it) | Manual testing of all view mode transitions |
| flush_pending() ordering with new subscriber flow | Medium | Medium (stale display) | Keep flush_pending() calls in subscriber callbacks |
| Palette editor in-place LUT mutation | Low | Low | Re-set PALETTE key to trigger invalidation |

---

## Out of Scope

- Project/render mutations (`actions.py` boundary/render functions stay as-is)
- UISync for DPG widget values (the skipped Milestone 5 -- revisit after this migration)
- Histogram/palette editor throttles (already marked exempt -- they are UI-side rate limiters, not state mutations)
- Breaking up postprocessing_panel.py (separate effort, recommended after this)
