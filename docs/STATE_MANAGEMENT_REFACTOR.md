# State Management Refactor Plan

Plan to fix the root cause of state synchronization bugs without a full rewrite.

---

## Current Problems

### 1. State Scattered Across Multiple Locations
```
AppState.display_settings.lightness_expr     # "Source of truth"
PostprocessingPanel._cached_global_lightness_expr  # UI cache
dpg.get_value("lightness_expr_input")        # Widget value
dpg.get_value("lightness_expr_checkbox")     # Widget state
lightness_expr_pending_update                # Pending flag
lightness_expr_pending_target                # Pending target
```

Any of these can be out of sync with the others.

### 2. Manual Debounce Patterns Repeated 5+ Times
Each has its own:
- `*_pending_update` flag
- `*_last_update_time` timestamp
- `*_pending_value` or `*_pending_target`
- Manual `check_*_debounce()` polling
- Manual `_apply_*_update()` application

Each copy is a bug waiting to happen.

### 3. Implicit Update Chains
```
User types in input
  → callback fires
  → sets pending flag
  → poll() checks flag
  → applies update
  → modifies state
  → calls refresh_display()
  → takes snapshot (might miss pending updates elsewhere!)
  → async job runs
  → completes
  → updates texture
  → marks canvas dirty
```

Any step can be interrupted or reordered.

### 4. No Clear Ownership
- Who owns the lightness expression? `AppState`? `PostprocessingPanel`? Both?
- When can state be modified? Any callback? Only certain ones?
- When is it safe to read state? Always? Only under lock?

---

## Target Architecture

### Core Principle: Single Source of Truth

```
┌─────────────────────────────────────────────────────────┐
│                      AppState                           │
│  (THE source of truth - all reads and writes go here)  │
└─────────────────────────────────────────────────────────┘
                           │
                           │ notify
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   StateManager                          │
│  - Handles all mutations                                │
│  - Notifies subscribers on change                       │
│  - Manages debouncing centrally                         │
│  - Ensures atomic updates                               │
└─────────────────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         ┌────────┐  ┌──────────┐  ┌──────────────┐
         │   UI   │  │ Display  │  │ Serialization│
         │ Panels │  │ Pipeline │  │              │
         └────────┘  └──────────┘  └──────────────┘

UI panels ONLY:
- Read from AppState (via StateManager)
- Request changes via StateManager.update()
- Subscribe to changes they care about

UI panels NEVER:
- Cache state locally
- Have "pending" flags
- Modify AppState directly
```

---

## Design Decisions

### 1. StateManager scope: display settings only, not structural mutations

**Decision:** `actions.py` continues to handle structural/project mutations (`move_boundary()`, `add_boundary()`, `set_palette()`, etc.). StateManager handles **display settings with debounced UI input** only.

**Rationale:**
- `actions.py` functions modify project topology (add/remove boundaries), geometry (move/scale), and set dirty flags (`field_dirty`, `render_dirty`). These are fundamentally different from "user drags a slider" updates.
- Forcing structural mutations through `StateManager.update(StateKey.BOUNDARY_POSITION, ...)` would bloat the StateKey enum, turn `_set_value()` into a massive switch handling unrelated operations, and add indirection with no benefit — these calls aren't debounced and don't need subscriptions.
- The dirty flag domains are independent: `field_dirty`/`render_dirty` (actions.py) trigger full recomputation, while display settings trigger postprocessing refresh. No conflict.
- The real problem this refactor solves is PostprocessingPanel's 6 parallel debounce systems and scattered UI caches — not that `move_boundary()` is a plain function.

**Boundary rule:** If it sets `field_dirty` or `render_dirty`, it goes through `actions.py`. If it triggers `refresh_display()`, it goes through StateManager.

### 2. Refresh coalescing: per-frame flag, not explicit batching

**Decision:** StateManager does not call `refresh_display()` directly from `_apply_update()`. Instead, it sets a `_needs_refresh` flag. The main loop flushes this flag once per frame, resulting in at most one `refresh_display()` call per frame regardless of how many state keys changed.

**Rationale:**
- Without this, updating brightness + contrast + gamma in quick succession (e.g., reset/load) starts a postprocess job on the first update with a partial state snapshot, then queues a second job via `_pending_refresh`. That's wasted GPU work.
- A `batch()` context manager would also solve this but adds API surface (nesting depth counter, batched key tracking) for a problem that the flag handles with zero ceremony.
- The 1-frame delay (~16ms at 60fps) is imperceptible. Current debounced paths already add 300ms.
- If expensive per-key subscriber work is added later, `batch()` can be introduced at that point with a concrete use case to design against.

**Implementation:**
```python
# In _apply_update():
if key in self._refresh_keys:
    self._needs_refresh = True
    if requires_invalidation:
        self._needs_invalidate = True

# In main loop (after poll_debounce):
if state_manager.needs_refresh():
    invalidate = state_manager.consume_refresh()
    display_pipeline.refresh_display(invalidate_cache=invalidate)
```

**Caveats to handle:**
- Carry a second `_needs_invalidate` flag for settings that must invalidate `base_rgb` (e.g., palette changes). Call `refresh_display(invalidate_cache=True)` when set.
- Flush must happen **after** `poll_debounce()` in the main loop so the snapshot sees final state for the frame.
- This does not address the subscriber threading issue (decision 3).

### 3. Threading model: main-thread only for StateManager

**Decision:** `update()`, `poll_debounce()`, `flush_pending()`, and all subscriber notifications run on the main thread only. This is a contract, not enforced at runtime.

**Rationale:**
- The existing codebase is already single-threaded for all UI and state mutation work. DearPyGui callbacks, slider handlers, and debounce poll methods all run on the main thread.
- Background threads (render orchestrator, display pipeline) do computation but post results back via `poll()` methods that check futures on the main thread. They write into `RenderCache` under the lock and set flags — they never modify display settings.
- UISync subscribers call `dpg.set_value()`, which is not thread-safe. Main-thread-only notifications make this safe without deferred queuing.
- No existing code path requires calling `state_manager.update()` from a background thread. If one arises, the right fix is to post a callback to the main thread, not to make StateManager thread-safe for writes.

**Rule:** Background threads must never call `state_manager.update()`. If a background thread needs to trigger a state change, it sets a flag/result that the main-thread `poll()` picks up and routes through StateManager.

### 4. Validation lives in callbacks, not StateManager

**Decision:** StateManager assumes all values passed to `update()` are already validated. Validation and error display are the responsibility of the UI callback that calls `update()`.

**Rationale:**
- StateManager is a generic setter/subscriber system. Embedding expression parsing or domain-specific validation would break that abstraction and make it harder to test.
- The UI panel already owns error display (showing "invalid expression" messages, highlighting fields). Validation naturally belongs alongside that.
- With debouncing, this gives better UX: validation runs immediately on keystroke (instant error feedback), but only valid values enter the debounce queue. Invalid expressions never overwrite the last valid state.

**Invariant:** Any code path that calls `state_manager.update()` must validate first. StateManager will apply whatever it receives.

**Implementation:**
- Validation functions (e.g., `validate_lightness_expr()`) live in a shared module (not in the panel) so non-UI callers can reuse them.
- The panel calls the validator and owns error display.

```python
# In callback:
def on_lightness_expr_change(self, sender=None, app_data=None):
    expr = dpg.get_value("lightness_expr_input").strip()
    if not expr:
        return

    error = validate_lightness_expr(expr)  # shared module
    if error:
        self._show_expr_error(error)       # UI concern
        return

    self._clear_expr_error()
    self.app.state_manager.update(StateKey.LIGHTNESS_EXPR, expr, debounce=0.3)
```

### 5. Scope: PostprocessingPanel display settings only

**Decision:** This refactor targets PostprocessingPanel's display settings (the 6 debounce patterns and UI caches). CanvasController, BoundaryControlsPanel, and RenderOrchestrator are out of scope.

**Rationale:**
- **CanvasController** drag throttling is input rate limiting (~60fps accumulated deltas), not debounced state updates. It doesn't have the scattered-cache problem. The pattern is appropriate for what it does.
- **BoundaryControlsPanel** modifies boundary parameters that set `field_dirty` — that's `actions.py` territory per decision 1.
- **RenderOrchestrator** has a clean async pattern with futures and polling. No debounce sprawl to fix.
- The actual pain point is PostprocessingPanel's 6 parallel debounce systems and duplicated UI caches. Expanding scope to other controllers adds risk without solving real bugs.

**Migration checklist adjustment:** BoundaryControlsPanel and CanvasController checklist items are reduced to "audit for consistency" — verify they don't bypass `actions.py` or hold stale caches. No migration to StateManager required.

### 6. Migration order: simplest-first, one feature at a time

**Decision:** Migrate immediate (non-debounced) sliders first to prove the architecture, then migrate debounced features one at a time. During migration, the old `check_*_debounce()` calls coexist alongside `state_manager.poll_debounce()` and are removed individually as each feature migrates.

**Order:**
1. Brightness, contrast, gamma, saturation (no debounce — trivial migration, proves the full flow)
2. Clip range (debounced, straightforward)
3. Smear sigma (debounced, straightforward)
4. Expression updates (debounced, moderate complexity)
5. Lightness expression (debounced, most complex — global vs per-region targeting)
6. Histogram throttle, palette editor throttle (throttles, not debounces — may need slight adaptation)

**Rationale:**
- Starting with immediate sliders is low-risk and validates the entire path: StateManager update → per-frame refresh flag → `refresh_display()`.
- Each subsequent migration deletes one `check_*` call from the main loop and one set of pending vars from PostprocessingPanel. Progress is visible and incremental.
- Lightness expression is last because it's the most complex (global vs custom mode, validation, per-region context). By the time we get there, the pattern is proven.

---

## Implementation Milestones

### Milestone 1: StateManager core + tests

Create `elliptica/app/state_manager.py` with `StateKey`, `StateManager`, per-frame refresh flag logic (decision 2). Write unit tests. No UI changes.

**Checklist:**
- [x] Create `StateKey` enum with all display setting keys
- [x] Create `StateManager` class with `update()`, `poll_debounce()`, `flush_pending()`
- [x] Implement `_set_value()` / `_get_value()` for all `StateKey` members
- [x] Implement `_set_region_style()` for per-boundary region updates
- [x] Implement `subscribe()` / `unsubscribe()` / `_notify()`
- [x] Implement `_needs_refresh` / `_needs_invalidate` flags with `needs_refresh()` / `consume_refresh()`
- [x] Define `_refresh_keys` and `_invalidate_keys` sets
- [x] Write tests: immediate update applies to state
- [x] Write tests: debounced update not applied until `poll_debounce()`
- [x] Write tests: `flush_pending()` applies all pending immediately
- [x] Write tests: subscriber notified on change
- [x] Write tests: `_needs_refresh` flag set for refresh keys
- [x] Write tests: `_needs_invalidate` flag set for invalidate keys

**Done when:** All tests pass. StateManager works in isolation with no UI dependencies.

**Reference — StateManager API:**
```python
manager = StateManager(app_state, lock)

# Immediate update
manager.update(StateKey.BRIGHTNESS, 1.2)

# Debounced update
manager.update(StateKey.LIGHTNESS_EXPR, "clipnorm(mag, 1, 99)", debounce=0.3)

# Region-specific update
manager.update(StateKey.REGION_STYLE, {"lightness_expr": expr},
               debounce=0.3, context=(boundary_id, region_type))

# Subscribe to changes
manager.subscribe(StateKey.BRIGHTNESS, my_callback)

# Per-frame polling (main loop)
manager.poll_debounce()
if manager.needs_refresh():
    invalidate = manager.consume_refresh()
    display_pipeline.refresh_display(invalidate_cache=invalidate)
```

---

### Milestone 2: Wire into main loop

Add StateManager to `EllipticaApp`. Add `poll_debounce()` and refresh flag flush to the main loop **alongside** existing `check_*` calls. Nothing changes yet — this is pure plumbing.

**Checklist:**
- [ ] Create `StateManager` in `EllipticaApp.__init__()`, passing `self.state` and `self.state_lock`
- [ ] Add `state_manager.poll_debounce()` to main loop (before existing `check_*` calls)
- [ ] Add refresh flag flush after `poll_debounce()` (before `dpg.render_dearpygui_frame()`)
- [ ] Verify app runs identically — no behavior changes

**Done when:** App starts, runs, and behaves exactly as before. StateManager exists but nothing uses it yet.

**Reference — main loop during migration:**
```python
while dpg.is_dearpygui_running():
    # New: StateManager polling
    self.state_manager.poll_debounce()
    if self.state_manager.needs_refresh():
        invalidate = self.state_manager.consume_refresh()
        self.display_pipeline.refresh_display(invalidate_cache=invalidate)

    # Old: remove these one at a time as each feature migrates
    self.postprocess_panel.check_clip_debounce()
    self.postprocess_panel.check_smear_debounce()
    self.postprocess_panel.check_expression_debounce()
    self.postprocess_panel.check_lightness_expr_debounce()
    self.postprocess_panel.check_histogram_debounce()
    self.postprocess_panel.check_palette_editor_debounce()

    # ... rest of loop ...
```

---

### Milestone 3: Migrate immediate sliders

Migrate brightness, contrast, gamma, saturation. These currently set state directly and call `refresh_display()` inline — no debounce. Simplest possible migration, proves the full StateManager flow.

**Checklist:**
- [ ] Rewrite `on_brightness_slider` to call `state_manager.update(StateKey.BRIGHTNESS, ...)`
- [ ] Rewrite `on_contrast_slider` to call `state_manager.update(StateKey.CONTRAST, ...)`
- [ ] Rewrite `on_gamma_slider` to call `state_manager.update(StateKey.GAMMA, ...)`
- [ ] Rewrite `on_saturation_slider` to call `state_manager.update(StateKey.SATURATION, ...)`
- [ ] Remove direct `self.app.state.display_settings.brightness = ...` from callbacks
- [ ] Remove direct `self.app.display_pipeline.refresh_display()` from these callbacks
- [ ] Verify sliders still work: drag slider → display updates

**Done when:** Four sliders route through StateManager. Refresh happens via per-frame flag, not inline `refresh_display()`.

**Reference — before/after:**
```python
# Before:
def on_brightness_slider(self, sender=None, app_data=None):
    with self.app.state_lock:
        self.app.state.display_settings.brightness = float(app_data)
    self.app.display_pipeline.refresh_display()

# After:
def on_brightness_slider(self, sender=None, app_data=None):
    self.app.state_manager.update(StateKey.BRIGHTNESS, float(app_data))
```

---

### Milestone 4: Migrate debounced features

Migrate each debounced feature one at a time. Each migration: rewrite callback → delete old pending vars/methods → remove `check_*` call from main loop → test.

#### 4a: Clip range
- [ ] Rewrite `on_clip_low_slider` / `on_clip_high_slider` to call `state_manager.update(..., debounce=0.3)`
- [ ] Delete `clip_pending_range`, `clip_last_update_time`, `clip_debounce_delay`
- [ ] Delete `check_clip_debounce()`, `_apply_clip_update()`
- [ ] Remove `check_clip_debounce()` call from main loop
- [ ] Verify: drag clip slider → display updates after debounce

#### 4b: Smear sigma
- [ ] Rewrite smear callback to call `state_manager.update(..., debounce=0.3)`
- [ ] Delete `smear_pending_value`, `smear_last_update_time`, `smear_debounce_delay`
- [ ] Delete `check_smear_debounce()`, `_apply_smear_update()`
- [ ] Remove `check_smear_debounce()` call from main loop
- [ ] Verify: smear slider works

#### 4c: Expression updates
- [ ] Rewrite expression callback with validation before `state_manager.update()`
- [ ] Delete `expr_pending_update`, `expr_last_update_time`, `expr_debounce_delay`
- [ ] Delete `check_expression_debounce()`, `_apply_expression_update()`
- [ ] Remove `check_expression_debounce()` call from main loop
- [ ] Verify: expression input works, invalid expressions show errors immediately

#### 4d: Lightness expression
- [ ] Rewrite `on_lightness_expr_change` with validation + global vs per-region routing
- [ ] Delete `lightness_expr_pending_update`, `lightness_expr_pending_target`, `lightness_expr_last_update_time`
- [ ] Delete `_cached_global_lightness_expr`, `_cached_custom_lightness_exprs`
- [ ] Delete `check_lightness_expr_debounce()`, `_apply_lightness_expr_update()`, `_apply_region_lightness_expr_update()`
- [ ] Remove `check_lightness_expr_debounce()` call from main loop
- [ ] Verify: global lightness expression works
- [ ] Verify: per-region custom lightness expression works
- [ ] Verify: switching between global and custom mode preserves values

#### 4e: Histogram throttle
- [ ] Adapt histogram throttle to StateManager (may need a throttle variant or remain as-is if it doesn't touch display settings)
- [ ] Delete `hist_pending_update`, `hist_last_update_time`, `hist_debounce_delay`
- [ ] Delete `check_histogram_debounce()`
- [ ] Remove `check_histogram_debounce()` call from main loop

#### 4f: Palette editor throttle
- [ ] Adapt palette editor refresh throttle to StateManager
- [ ] Delete `palette_editor_refresh_pending`, `palette_editor_last_refresh_time`, `palette_editor_refresh_throttle`
- [ ] Delete `check_palette_editor_debounce()`
- [ ] Remove `check_palette_editor_debounce()` call from main loop

**Done when:** All 6 `check_*` calls removed from main loop. Only `state_manager.poll_debounce()` remains. No `*_pending_*` or `_cached_*` variables in PostprocessingPanel.

---

### Milestone 5: UISync bindings (optional — can defer)

Create UISync system so widgets auto-update from state subscriptions. This eliminates manual `dpg.set_value()` calls scattered through `update_context_ui()` and similar methods.

**Checklist:**
- [ ] Create `UISync` class in `elliptica/ui/dpg/ui_sync.py`
- [ ] Implement `bind(key, widget_tag, transform)` and `sync_all()`
- [ ] Bind brightness/contrast/gamma/saturation sliders
- [ ] Bind clip low/high sliders
- [ ] Bind lightness expression checkbox and input
- [ ] Simplify `update_context_ui()` to visibility-only logic
- [ ] Simplify `update_region_properties_panel()` to visibility-only logic

**Done when:** Widget values are driven by state subscriptions. `update_context_ui()` only toggles visibility.

---

### Milestone 6: Cleanup + audit

Remove any remaining dead code. Audit out-of-scope controllers for consistency.

**Checklist:**
- [ ] Grep for any remaining `_cached_*` variables in PostprocessingPanel
- [ ] Grep for any remaining `*_pending_*` variables in PostprocessingPanel
- [ ] Grep for any remaining `check_*_debounce` methods
- [ ] Audit BoundaryControlsPanel: verify all mutations go through `actions.py`, no stale caches
- [ ] Audit CanvasController: verify drag throttling is self-consistent, no stale caches
- [ ] Verify `DisplayPipelineController` calls `state_manager.flush_pending()` before snapshot
- [ ] Update any existing tests that relied on old patterns

**Done when:** No old debounce/cache patterns remain. Audits pass.

---

## Success Criteria

After refactor:

1. **No UI caches** — All state lives in AppState only
2. **No manual debounce patterns** — StateManager handles all debouncing
3. **Single update path** — All display setting changes go through StateManager
4. **Per-frame refresh coalescing** — At most one `refresh_display()` per frame
5. **Testable** — StateManager can be unit tested in isolation
6. **Debuggable** — Single place to log/breakpoint all display setting changes

---

## Risks and Mitigations

### Risk: Breaking existing functionality
**Mitigation:** Migrate one feature at a time (milestones 3-4). Each step is independently testable. Old and new systems coexist during migration.

### Risk: Performance regression from notifications
**Mitigation:** Per-frame refresh coalescing (decision 2) prevents redundant postprocess jobs. Profile before/after if needed.

### Risk: Deadlocks from notification callbacks
**Mitigation:** Always notify outside the lock. Main-thread-only contract (decision 3) eliminates cross-thread lock contention.

### Risk: Histogram/palette throttles don't fit debounce model
**Mitigation:** Evaluate during milestone 4e/4f. These may remain as simple frame-based throttles if they don't touch display settings through StateManager.
