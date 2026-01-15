# State Management Refactor Plan

A 4-week plan to fix the root cause of state synchronization bugs without a full rewrite.

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

## Week 1: StateManager Foundation

### Day 1-2: Create StateManager Class

```python
# elliptica/app/state_manager.py

from __future__ import annotations
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Optional, TYPE_CHECKING
from enum import Enum, auto

if TYPE_CHECKING:
    from elliptica.app.core import AppState


class StateKey(Enum):
    """All observable state paths."""
    # Display settings
    LIGHTNESS_EXPR = auto()
    BRIGHTNESS = auto()
    CONTRAST = auto()
    GAMMA = auto()
    SATURATION = auto()
    CLIP_LOW = auto()
    CLIP_HIGH = auto()
    PALETTE = auto()
    COLOR_ENABLED = auto()

    # Region-specific
    REGION_STYLE = auto()  # Takes (boundary_id, region_type) as context

    # Render settings
    RENDER_MULTIPLIER = auto()

    # View state
    VIEW_MODE = auto()
    SELECTION = auto()


class StateManager:
    """Centralized state management with subscriptions and debouncing.

    Usage:
        manager = StateManager(app_state, lock)

        # Update state (immediate)
        manager.update(StateKey.BRIGHTNESS, 1.2)

        # Update state (debounced)
        manager.update(StateKey.LIGHTNESS_EXPR, "clipnorm(mag, 1, 99)", debounce=0.3)

        # Subscribe to changes
        manager.subscribe(StateKey.BRIGHTNESS, my_callback)

        # Flush all pending updates
        manager.flush_pending()
    """

    def __init__(self, state: "AppState", lock: threading.RLock):
        self.state = state
        self.lock = lock

        # Subscribers: key -> list of callbacks
        self._subscribers: dict[StateKey, list[Callable[[Any], None]]] = {}

        # Debounce state: key -> (value, timestamp, delay)
        self._pending: dict[StateKey, tuple[Any, float, float]] = {}

        # Context for region-specific keys: key -> context
        self._pending_context: dict[StateKey, Any] = {}

    def update(
        self,
        key: StateKey,
        value: Any,
        debounce: float = 0.0,
        context: Any = None,
    ) -> None:
        """Update a state value.

        Args:
            key: Which state to update
            value: New value
            debounce: If > 0, delay application by this many seconds
            context: Additional context (e.g., boundary_id for region styles)
        """
        if debounce > 0:
            self._pending[key] = (value, time.time(), debounce)
            if context is not None:
                self._pending_context[key] = context
        else:
            self._apply_update(key, value, context)

    def _apply_update(self, key: StateKey, value: Any, context: Any = None) -> None:
        """Actually apply the state update."""
        with self.lock:
            self._set_value(key, value, context)

        # Notify subscribers (outside lock to prevent deadlock)
        self._notify(key, value)

    def _set_value(self, key: StateKey, value: Any, context: Any = None) -> None:
        """Set the actual state value. Must be called with lock held."""
        ds = self.state.display_settings

        if key == StateKey.LIGHTNESS_EXPR:
            self.state.display_settings = replace(ds, lightness_expr=value)
        elif key == StateKey.BRIGHTNESS:
            self.state.display_settings = replace(ds, brightness=value)
        elif key == StateKey.CONTRAST:
            self.state.display_settings = replace(ds, contrast=value)
        elif key == StateKey.GAMMA:
            self.state.display_settings = replace(ds, gamma=value)
        elif key == StateKey.SATURATION:
            self.state.display_settings = replace(ds, saturation=value)
        elif key == StateKey.CLIP_LOW:
            self.state.display_settings = replace(ds, clip_low_percent=value)
        elif key == StateKey.CLIP_HIGH:
            self.state.display_settings = replace(ds, clip_high_percent=value)
        elif key == StateKey.PALETTE:
            self.state.display_settings = replace(ds, palette=value)
        elif key == StateKey.COLOR_ENABLED:
            self.state.display_settings = replace(ds, color_enabled=value)
        elif key == StateKey.REGION_STYLE:
            boundary_id, region_type = context
            # Update boundary_color_settings...
            self._set_region_style(boundary_id, region_type, value)
        elif key == StateKey.VIEW_MODE:
            self.state.view_mode = value
        elif key == StateKey.SELECTION:
            self.state.selected_indices = value
        # Add more as needed

    def _set_region_style(self, boundary_id: int, region_type: str, style_updates: dict) -> None:
        """Update region-specific style settings."""
        from elliptica.app.core import BoundaryColorSettings, RegionStyle

        if boundary_id not in self.state.boundary_color_settings:
            self.state.boundary_color_settings[boundary_id] = BoundaryColorSettings()

        bcs = self.state.boundary_color_settings[boundary_id]
        current = bcs.surface if region_type == "surface" else bcs.interior

        if current is None:
            current = RegionStyle()

        # Apply updates
        updated = replace(current, **style_updates)

        if region_type == "surface":
            self.state.boundary_color_settings[boundary_id] = replace(bcs, surface=updated)
        else:
            self.state.boundary_color_settings[boundary_id] = replace(bcs, interior=updated)

    def get(self, key: StateKey, context: Any = None) -> Any:
        """Read a state value."""
        with self.lock:
            return self._get_value(key, context)

    def _get_value(self, key: StateKey, context: Any = None) -> Any:
        """Get the actual state value. Must be called with lock held."""
        ds = self.state.display_settings

        if key == StateKey.LIGHTNESS_EXPR:
            return ds.lightness_expr
        elif key == StateKey.BRIGHTNESS:
            return ds.brightness
        elif key == StateKey.CONTRAST:
            return ds.contrast
        elif key == StateKey.GAMMA:
            return ds.gamma
        elif key == StateKey.SATURATION:
            return ds.saturation
        elif key == StateKey.CLIP_LOW:
            return ds.clip_low_percent
        elif key == StateKey.CLIP_HIGH:
            return ds.clip_high_percent
        elif key == StateKey.PALETTE:
            return ds.palette
        elif key == StateKey.COLOR_ENABLED:
            return ds.color_enabled
        elif key == StateKey.VIEW_MODE:
            return self.state.view_mode
        elif key == StateKey.SELECTION:
            return self.state.selected_indices.copy()
        # Add more as needed

        return None

    def subscribe(self, key: StateKey, callback: Callable[[Any], None]) -> None:
        """Subscribe to changes for a state key."""
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)

    def unsubscribe(self, key: StateKey, callback: Callable[[Any], None]) -> None:
        """Unsubscribe from changes."""
        if key in self._subscribers:
            self._subscribers[key] = [cb for cb in self._subscribers[key] if cb != callback]

    def _notify(self, key: StateKey, value: Any) -> None:
        """Notify subscribers of a change."""
        for callback in self._subscribers.get(key, []):
            try:
                callback(value)
            except Exception as e:
                import logging
                logging.getLogger(__name__).exception(
                    "Subscriber callback failed for %s: %s", key, e
                )

    def flush_pending(self) -> None:
        """Apply all pending debounced updates immediately."""
        pending_copy = list(self._pending.items())
        self._pending.clear()

        for key, (value, _, _) in pending_copy:
            context = self._pending_context.pop(key, None)
            self._apply_update(key, value, context)

    def poll_debounce(self) -> None:
        """Check and apply any debounced updates that are ready.

        Call this once per frame from the main loop.
        """
        now = time.time()
        ready_keys = []

        for key, (value, timestamp, delay) in self._pending.items():
            if now - timestamp >= delay:
                ready_keys.append(key)

        for key in ready_keys:
            value, _, _ = self._pending.pop(key)
            context = self._pending_context.pop(key, None)
            self._apply_update(key, value, context)

    def has_pending(self) -> bool:
        """Check if there are any pending updates."""
        return len(self._pending) > 0
```

### Day 3-4: Integrate with App

```python
# In elliptica/ui/dpg/app.py

class EllipticaApp:
    def __init__(self):
        # ... existing init ...
        self.state_manager = StateManager(self.state, self.state_lock)

    def render(self):
        # ... in main loop ...
        while dpg.is_dearpygui_running():
            # Poll debounced updates (replaces all the individual check_*_debounce calls)
            self.state_manager.poll_debounce()

            # ... rest of loop ...
```

### Day 5: Add Refresh Trigger

```python
# StateManager needs to trigger display refresh on relevant changes

class StateManager:
    def __init__(self, state, lock, display_pipeline=None):
        # ...
        self.display_pipeline = display_pipeline
        self._refresh_keys = {
            StateKey.LIGHTNESS_EXPR,
            StateKey.BRIGHTNESS,
            StateKey.CONTRAST,
            StateKey.GAMMA,
            StateKey.SATURATION,
            StateKey.CLIP_LOW,
            StateKey.CLIP_HIGH,
            StateKey.PALETTE,
            StateKey.COLOR_ENABLED,
            StateKey.REGION_STYLE,
        }

    def _apply_update(self, key, value, context=None):
        with self.lock:
            self._set_value(key, value, context)

        self._notify(key, value)

        # Auto-refresh display for visual settings
        if key in self._refresh_keys and self.display_pipeline:
            self.display_pipeline.refresh_display()
```

---

## Week 2: Migrate UI Panels

### Goal: Remove all local caches and pending flags from PostprocessingPanel

### Day 1-2: Migrate Lightness Expression

**Before:**
```python
# Old pattern in postprocessing_panel.py
def on_lightness_expr_change(self, sender=None, app_data=None):
    # Complex logic with pending flags, caches, targets...
    self.lightness_expr_pending_update = True
    self.lightness_expr_pending_target = "global"
    self.lightness_expr_last_update_time = time.time()
```

**After:**
```python
# New pattern
def on_lightness_expr_change(self, sender=None, app_data=None):
    expr = dpg.get_value("lightness_expr_input").strip()
    if not expr:
        return

    # Determine target
    if self._is_boundary_selected() and self._get_lightness_mode() == "Custom":
        boundary_id = self._get_selected_boundary_id()
        region_type = self.app.state.selected_region_type
        self.app.state_manager.update(
            StateKey.REGION_STYLE,
            {"lightness_expr": expr},
            debounce=0.3,
            context=(boundary_id, region_type),
        )
    else:
        self.app.state_manager.update(
            StateKey.LIGHTNESS_EXPR,
            expr,
            debounce=0.3,
        )
```

### Day 3-4: Migrate Other Sliders

```python
# Brightness (same pattern for contrast, gamma, saturation)
def on_brightness_slider(self, sender=None, app_data=None):
    self.app.state_manager.update(StateKey.BRIGHTNESS, float(app_data))

# Clip range (with debounce since percentile computation is expensive)
def on_clip_low_slider(self, sender=None, app_data=None):
    self.app.state_manager.update(StateKey.CLIP_LOW, float(app_data), debounce=0.3)
```

### Day 5: Remove Dead Code

Delete from PostprocessingPanel:
- `_cached_global_lightness_expr`
- `_cached_custom_lightness_exprs`
- `lightness_expr_pending_update`
- `lightness_expr_pending_target`
- `lightness_expr_last_update_time`
- `check_lightness_expr_debounce()`
- `_apply_lightness_expr_update()`
- `_apply_region_lightness_expr_update()`
- Similar for clip, smear, etc.

---

## Week 3: UI Synchronization

### Goal: UI always reflects state, never caches independently

### Day 1-2: Create UI Sync System

```python
# elliptica/ui/dpg/ui_sync.py

class UISync:
    """Keeps UI widgets synchronized with state.

    Instead of UI caching state, state notifies UI of changes.
    """

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self._bindings: dict[StateKey, list[tuple[str, Callable]]] = {}

    def bind(
        self,
        key: StateKey,
        widget_tag: str,
        transform: Callable[[Any], Any] = lambda x: x,
    ) -> None:
        """Bind a widget to a state key.

        When state changes, widget is automatically updated.

        Args:
            key: State key to watch
            widget_tag: DearPyGui widget tag
            transform: Optional transform from state value to widget value
        """
        if key not in self._bindings:
            self._bindings[key] = []
            # Subscribe to state changes
            self.state_manager.subscribe(key, lambda v: self._on_state_change(key, v))

        self._bindings[key].append((widget_tag, transform))

    def _on_state_change(self, key: StateKey, value: Any) -> None:
        """Handle state change by updating bound widgets."""
        for widget_tag, transform in self._bindings.get(key, []):
            if dpg.does_item_exist(widget_tag):
                try:
                    dpg.set_value(widget_tag, transform(value))
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Failed to update widget %s: %s", widget_tag, e
                    )

    def sync_all(self) -> None:
        """Force sync all widgets to current state values."""
        for key in self._bindings:
            value = self.state_manager.get(key)
            self._on_state_change(key, value)
```

### Day 3-4: Apply Bindings

```python
# In postprocessing_panel.py build method:

def build_postprocessing_ui(self, parent, colormaps):
    # ... create widgets ...

    # Bind widgets to state
    self.ui_sync = UISync(self.app.state_manager)

    self.ui_sync.bind(StateKey.BRIGHTNESS, "brightness_slider")
    self.ui_sync.bind(StateKey.CONTRAST, "contrast_slider")
    self.ui_sync.bind(StateKey.GAMMA, "gamma_slider")
    self.ui_sync.bind(StateKey.SATURATION, "saturation_slider")
    self.ui_sync.bind(StateKey.CLIP_LOW, "clip_low_slider")
    self.ui_sync.bind(StateKey.CLIP_HIGH, "clip_high_slider")

    # Lightness expression checkbox (transform: expr -> bool)
    self.ui_sync.bind(
        StateKey.LIGHTNESS_EXPR,
        "lightness_expr_checkbox",
        transform=lambda expr: expr is not None,
    )

    # Lightness expression input
    self.ui_sync.bind(
        StateKey.LIGHTNESS_EXPR,
        "lightness_expr_input",
        transform=lambda expr: expr or "clipnorm(mag, 1, 99)",
    )
```

### Day 5: Simplify update_context_ui

With bindings in place, `update_context_ui()` becomes much simpler:

```python
def update_context_ui(self) -> None:
    """Update UI based on current context (global vs boundary mode)."""
    is_boundary = self._is_boundary_selected()

    # Just toggle visibility - values are auto-synced
    dpg.configure_item("global_palette_group", show=not is_boundary)
    dpg.configure_item("region_palette_group", show=is_boundary)
    dpg.configure_item("lightness_expr_checkbox", show=not is_boundary)
    dpg.configure_item("lightness_expr_mode_toggle", show=is_boundary)

    # ... minimal visibility logic only ...
```

---

## Week 4: Testing and Cleanup

### Day 1-2: Add Integration Tests

```python
# tests/test_state_manager.py

def test_debounced_update_applied():
    """Debounced updates should be applied after delay."""
    state = AppState()
    lock = threading.RLock()
    manager = StateManager(state, lock)

    manager.update(StateKey.LIGHTNESS_EXPR, "test_expr", debounce=0.1)

    # Not applied yet
    assert state.display_settings.lightness_expr is None

    # Wait for debounce
    time.sleep(0.15)
    manager.poll_debounce()

    # Now applied
    assert state.display_settings.lightness_expr == "test_expr"


def test_flush_applies_pending():
    """flush_pending should apply all pending updates immediately."""
    state = AppState()
    lock = threading.RLock()
    manager = StateManager(state, lock)

    manager.update(StateKey.BRIGHTNESS, 1.5, debounce=10.0)  # Long debounce
    manager.update(StateKey.CONTRAST, 2.0, debounce=10.0)

    assert state.display_settings.brightness == 1.0  # Default
    assert state.display_settings.contrast == 1.0  # Default

    manager.flush_pending()

    assert state.display_settings.brightness == 1.5
    assert state.display_settings.contrast == 2.0


def test_subscriber_notified():
    """Subscribers should be called on state change."""
    state = AppState()
    lock = threading.RLock()
    manager = StateManager(state, lock)

    received = []
    manager.subscribe(StateKey.BRIGHTNESS, lambda v: received.append(v))

    manager.update(StateKey.BRIGHTNESS, 1.5)

    assert received == [1.5]
```

### Day 3-4: Remove Old Code

1. Delete all `check_*_debounce()` methods
2. Delete all `_apply_*_update()` methods
3. Delete all `*_pending_*` instance variables
4. Delete all `_cached_*` instance variables
5. Simplify callbacks to just call `state_manager.update()`

### Day 5: Documentation and Review

1. Document StateManager API
2. Document migration guide for any remaining old patterns
3. Code review for any remaining direct state access
4. Update any tests that relied on old patterns

---

## Migration Checklist

### PostprocessingPanel
- [ ] Remove `_cached_global_lightness_expr`
- [ ] Remove `_cached_custom_lightness_exprs`
- [ ] Remove `lightness_expr_pending_*` variables
- [ ] Remove `clip_pending_*` variables
- [ ] Remove `smear_pending_*` variables
- [ ] Remove `expr_pending_*` variables
- [ ] Remove `check_lightness_expr_debounce()`
- [ ] Remove `check_clip_debounce()`
- [ ] Remove `check_smear_debounce()`
- [ ] Simplify `update_context_ui()` to visibility-only
- [ ] Simplify `update_region_properties_panel()` to visibility-only

### BoundaryControlsPanel
- [ ] Audit for direct state access
- [ ] Migrate to state_manager.update()

### CanvasController
- [ ] Audit for direct state access
- [ ] Migrate selection changes to state_manager

### DisplayPipelineController
- [ ] Call `state_manager.flush_pending()` before snapshot
- [ ] Remove reliance on external flush calls

### Main Loop (app.py)
- [ ] Replace all `check_*_debounce()` calls with single `state_manager.poll_debounce()`

---

## Success Criteria

After refactor:

1. **No UI caches** - All state lives in AppState only
2. **No manual debounce patterns** - StateManager handles all debouncing
3. **Single update path** - All state changes go through StateManager
4. **Auto-sync UI** - UI bindings keep widgets in sync automatically
5. **Testable** - StateManager can be unit tested in isolation
6. **Debuggable** - Single place to log/breakpoint all state changes

---

## Risks and Mitigations

### Risk: Breaking existing functionality
**Mitigation:** Migrate one feature at a time, test thoroughly before moving to next

### Risk: Performance regression from notifications
**Mitigation:** Profile before/after, batch notifications if needed

### Risk: Deadlocks from notification callbacks
**Mitigation:** Always notify outside the lock, document threading model

### Risk: Month timeline too aggressive
**Mitigation:** Week 4 is buffer; core functionality in weeks 1-3
