# Quick Wins for Immediate Stability

Tactical fixes that can be applied in 1-2 days to reduce bugs without architectural changes.

---

## 1. Flush All Pending Updates Before Display Refresh

**Problem:** Pending debounced updates get lost when `refresh_display()` snapshots state.

**Fix:** Add a flush method and call it before every snapshot.

```python
# In postprocessing_panel.py, add this method:

def flush_all_pending_updates(self) -> None:
    """Force-apply all pending debounced updates immediately.

    Call this before any state snapshot to ensure consistency.
    """
    # Lightness expression
    if self.lightness_expr_pending_update:
        if isinstance(self.lightness_expr_pending_target, tuple):
            boundary_id, region_type = self.lightness_expr_pending_target
            self._apply_region_lightness_expr_update(boundary_id, region_type)
        elif self.lightness_expr_pending_target == "global":
            self._apply_lightness_expr_update()
        self.lightness_expr_pending_update = False
        self.lightness_expr_pending_target = None

    # Clip range
    if self.clip_pending_range is not None:
        self._apply_clip_update()
        self.clip_pending_range = None

    # Smear
    if self.smear_pending_value is not None:
        self._apply_smear_update()
        self.smear_pending_value = None

    # Expression editor
    if self.expr_pending_update:
        self._apply_expression_update()
        self.expr_pending_update = False
```

```python
# In display_pipeline_controller.py, call it before snapshot:

def _start_postprocess_job(self) -> None:
    # Flush any pending UI updates before taking snapshot
    self.app.postprocess_panel.flush_all_pending_updates()

    with self.app.state_lock:
        # ... existing snapshot code ...
```

---

## 2. Add State Validation Assertions

**Problem:** State and UI can desync silently, causing mysterious bugs.

**Fix:** Add debug assertions that catch desync early.

```python
# In postprocessing_panel.py, add validation method:

def _validate_state_ui_sync(self) -> None:
    """Debug assertion to catch state/UI desync. Call after UI updates."""
    if not __debug__:
        return

    # Check lightness expression checkbox matches state
    if dpg.does_item_exist("lightness_expr_checkbox"):
        checkbox_value = dpg.get_value("lightness_expr_checkbox")
        state_has_expr = self.app.state.display_settings.lightness_expr is not None
        assert checkbox_value == state_has_expr, (
            f"Lightness checkbox desync: UI={checkbox_value}, state={state_has_expr}"
        )

    # Check brightness slider matches state
    if self.postprocess_brightness_slider_id is not None:
        slider_value = dpg.get_value(self.postprocess_brightness_slider_id)
        state_value = self.app.state.display_settings.brightness
        assert abs(slider_value - state_value) < 0.01, (
            f"Brightness desync: UI={slider_value}, state={state_value}"
        )

# Call at end of update_context_ui():
def update_context_ui(self) -> None:
    # ... existing code ...

    if __debug__:
        self._validate_state_ui_sync()
```

---

## 3. Log State Snapshots on Bug Occurrence

**Problem:** Hard to debug state issues after the fact.

**Fix:** Add snapshot logging that can be enabled when debugging.

```python
# In display_pipeline_controller.py:

import logging
logger = logging.getLogger(__name__)

def _start_postprocess_job(self) -> None:
    with self.app.state_lock:
        # ... build settings_snapshot ...

        # Log snapshot for debugging (enable with logging.DEBUG level)
        logger.debug(
            "Postprocess snapshot: lightness_expr=%r, brightness=%.2f, "
            "color_config=%s, clip=(%.1f, %.1f)",
            settings_snapshot['lightness_expr'],
            settings_snapshot['brightness'],
            'set' if color_config_snapshot else 'None',
            settings_snapshot['clip_low_percent'],
            settings_snapshot['clip_high_percent'],
        )
```

Enable in your main app startup:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 4. Guard Against None Expression in Postprocessing

**Problem:** If lightness_expr is unexpectedly None, image goes black.

**Fix:** Add defensive check with warning.

```python
# In gpu/postprocess.py, around line 272:

if lightness_expr is not None:
    base_rgb = apply_lightness_expr_gpu(...)
elif color_enabled and palette is not None:
    # Expression is None but we have a palette - this is fine, just skip expression
    pass
else:
    # Unexpected: no expression and no color - log warning
    import logging
    logging.getLogger(__name__).warning(
        "Postprocessing with no lightness_expr and color_enabled=%s - "
        "result may be dark/black", color_enabled
    )
```

---

## 5. Populate Cache on First Read, Not Just Toggle

**Problem:** `_cached_global_lightness_expr` can be None when user toggles, losing their expression.

**Fix:** Populate cache whenever we read state.

```python
# In postprocessing_panel.py, add helper:

def _ensure_lightness_cache_populated(self) -> None:
    """Ensure cache has a value if state has an expression."""
    if self._cached_global_lightness_expr is None:
        expr = self.app.state.display_settings.lightness_expr
        if expr is not None:
            self._cached_global_lightness_expr = expr

# Call at start of on_lightness_expr_toggle:
def on_lightness_expr_toggle(self, sender=None, app_data=None) -> None:
    self._ensure_lightness_cache_populated()
    # ... rest of method ...
```

---

## 6. Add Lock to All State Reads in Callbacks

**Problem:** Some callbacks read state without lock, risking torn reads.

**Fix:** Audit and add locks consistently.

```python
# BAD - no lock:
def some_callback(self):
    value = self.app.state.display_settings.brightness  # Race condition!

# GOOD - with lock:
def some_callback(self):
    with self.app.state_lock:
        value = self.app.state.display_settings.brightness
```

Quick audit command:
```bash
grep -n "self.app.state\." elliptica/ui/dpg/*.py | grep -v "state_lock"
```

---

## Implementation Order

1. **Flush pending updates** (30 min) - Highest impact, prevents lost edits
2. **Populate cache on first read** (15 min) - Prevents None cache bugs
3. **State validation assertions** (30 min) - Catches future bugs early
4. **Logging** (15 min) - Helps debug remaining issues
5. **Lock audit** (1-2 hours) - Prevents race conditions
6. **Defensive postprocess check** (15 min) - Graceful degradation

Total: ~3-4 hours for significant stability improvement.
