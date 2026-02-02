"""Centralized state management for display settings.

StateManager mediates all display-setting mutations, providing:
- A single update path (no scattered caches or pending flags)
- Built-in debouncing with last-value-wins semantics
- Per-key subscriber notifications (outside the lock)
- Per-frame refresh/invalidation flags for the main loop
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

from elliptica.app.core import AppState, BoundaryColorSettings


class StateKey(enum.Enum):
    """Keys for managed display settings."""

    # DisplaySettings scalars
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    GAMMA = "gamma"
    SATURATION = "saturation"

    # DisplaySettings clip
    CLIP_LOW_PERCENT = "clip_low_percent"
    CLIP_HIGH_PERCENT = "clip_high_percent"

    # DisplaySettings misc
    LIGHTNESS_EXPR = "lightness_expr"
    COLOR_ENABLED = "color_enabled"
    PALETTE = "palette"

    # DisplaySettings sigma
    DOWNSAMPLE_SIGMA = "downsample_sigma"

    # AppState-level
    COLOR_CONFIG = "color_config"

    # Per-boundary (requires context tuple)
    REGION_STYLE = "region_style"

    # Interaction state (no display refresh, subscriber-driven)
    VIEW_MODE = "view_mode"
    SELECTED_INDICES = "selected_indices"
    SELECTED_REGION_TYPE = "selected_region_type"


@dataclass
class _PendingEntry:
    """A deferred refresh signal for a debounced update.

    State is mutated and subscribers notified immediately when
    ``update()`` is called with ``debounce > 0``.  Only the
    refresh/invalidation flags are deferred until the delay elapses.
    """

    timestamp: float
    delay: float
    requires_invalidation: bool


class StateManager:
    """Centralized mutation and notification hub for display settings.

    All display-setting changes flow through ``update()``.  State is
    always mutated immediately (so reads see the latest value).  For
    debounced updates, only the refresh/invalidation signals are
    deferred until ``poll_debounce()`` finds the delay has elapsed.

    The manager never calls ``refresh_display()`` directly.  Instead it
    sets ``_needs_refresh`` (and optionally ``_needs_invalidate``).  The
    main loop checks these once per frame via ``needs_refresh()`` /
    ``consume_refresh()``.
    """

    # Maps StateKey -> DisplaySettings attribute name (10 keys)
    _DISPLAY_SETTINGS_MAP: dict[StateKey, str] = {
        StateKey.BRIGHTNESS: "brightness",
        StateKey.CONTRAST: "contrast",
        StateKey.GAMMA: "gamma",
        StateKey.SATURATION: "saturation",
        StateKey.CLIP_LOW_PERCENT: "clip_low_percent",
        StateKey.CLIP_HIGH_PERCENT: "clip_high_percent",
        StateKey.LIGHTNESS_EXPR: "lightness_expr",
        StateKey.COLOR_ENABLED: "color_enabled",
        StateKey.PALETTE: "palette",
        StateKey.DOWNSAMPLE_SIGMA: "downsample_sigma",
    }

    # Display-setting keys that trigger a display refresh.
    # Interaction state keys (VIEW_MODE, SELECTED_*, SELECTED_REGION_TYPE)
    # are deliberately excluded — they only fire subscriber notifications.
    _REFRESH_KEYS: frozenset[StateKey] = frozenset({
        StateKey.BRIGHTNESS,
        StateKey.CONTRAST,
        StateKey.GAMMA,
        StateKey.SATURATION,
        StateKey.CLIP_LOW_PERCENT,
        StateKey.CLIP_HIGH_PERCENT,
        StateKey.LIGHTNESS_EXPR,
        StateKey.COLOR_ENABLED,
        StateKey.PALETTE,
        StateKey.DOWNSAMPLE_SIGMA,
        StateKey.COLOR_CONFIG,
        StateKey.REGION_STYLE,
    })

    # Keys that require base_rgb invalidation (recompute from scratch).
    _INVALIDATE_KEYS: frozenset[StateKey] = frozenset({
        StateKey.BRIGHTNESS,
        StateKey.CONTRAST,
        StateKey.GAMMA,
        StateKey.CLIP_LOW_PERCENT,
        StateKey.CLIP_HIGH_PERCENT,
        StateKey.LIGHTNESS_EXPR,
        StateKey.COLOR_ENABLED,
        StateKey.PALETTE,
    })

    # RegionStyle fields that trigger invalidation when changed.
    _REGION_INVALIDATE_FIELDS: frozenset[str] = frozenset({
        "enabled", "brightness", "contrast", "gamma", "lightness_expr",
        "use_palette", "palette", "solid_color",
        "smear_enabled", "smear_sigma",
    })

    _VALID_REGION_TYPES: frozenset[str] = frozenset({"surface", "interior"})

    def __init__(
        self,
        state: AppState,
        lock: threading.RLock,
        *,
        _clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state = state
        self._lock = lock
        self._clock = _clock

        self._subscribers: dict[StateKey, list[Callable]] = {}
        self._pending: dict[tuple[StateKey, Optional[tuple]], _PendingEntry] = {}

        self._needs_refresh: bool = False
        self._needs_invalidate: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        key: StateKey,
        value: Any,
        *,
        debounce: float = 0.0,
        context: Optional[tuple] = None,
    ) -> None:
        """Request a state change.

        Args:
            key: Which setting to change.
            value: New value (assumed already validated).
            debounce: Seconds to defer the refresh signal.  ``<= 0``
                means immediate.  State is always mutated right away
                regardless of debounce.
            context: Required for ``REGION_STYLE`` — a
                ``(boundary_id, region_type)`` tuple.
        """
        if debounce <= 0:
            # Clear any pending debounced entry so its deferred refresh
            # doesn't fire redundantly after this immediate update.
            self._pending.pop((key, context), None)
            self._apply_update(key, value, context)
        else:
            # Mutate state and notify immediately so reads are always
            # current.  Only the refresh/invalidation flags are deferred.
            requires_invalidation = self._apply_update(
                key, value, context, set_flags=False,
            )
            pending_key = (key, context)
            existing = self._pending.get(pending_key)
            self._pending[pending_key] = _PendingEntry(
                timestamp=self._clock(),
                delay=debounce,
                requires_invalidation=requires_invalidation or (
                    existing.requires_invalidation if existing else False
                ),
            )

    def poll_debounce(self) -> None:
        """Set refresh/invalidation flags for debounced entries whose delay has elapsed.

        State was already mutated when ``update()`` was called; this
        method only fires the deferred display-refresh signals.
        """
        if not self._pending:
            return
        now = self._clock()
        ready = [
            (pk, self._pending[pk])
            for pk in list(self._pending)
            if now - self._pending[pk].timestamp >= self._pending[pk].delay
        ]
        for pending_key, entry in ready:
            self._pending.pop(pending_key, None)
        for pending_key, entry in ready:
            key, _context = pending_key
            if key in self._REFRESH_KEYS:
                self._needs_refresh = True
                if entry.requires_invalidation:
                    self._needs_invalidate = True

    def flush_pending(self) -> None:
        """Fire all pending deferred refresh signals immediately.

        State was already mutated when each ``update()`` was called;
        this method only sets the refresh/invalidation flags.
        """
        pending = dict(self._pending)
        self._pending.clear()
        for (key, _context), entry in pending.items():
            if key in self._REFRESH_KEYS:
                self._needs_refresh = True
                if entry.requires_invalidation:
                    self._needs_invalidate = True

    def subscribe(self, key: StateKey, callback: Callable) -> None:
        """Register *callback* for notifications when *key* changes.

        Callback signature: ``(key, value, context)``.
        """
        self._subscribers.setdefault(key, []).append(callback)

    def unsubscribe(self, key: StateKey, callback: Callable) -> None:
        """Remove a previously registered callback."""
        callbacks = self._subscribers.get(key)
        if callbacks:
            try:
                callbacks.remove(callback)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Selection convenience methods
    # ------------------------------------------------------------------

    def set_selected(self, idx: int) -> None:
        """Replace selection with single boundary at index, or clear if invalid."""
        with self._lock:
            new_set = set()
            if 0 <= idx < len(self._state.project.boundary_objects):
                new_set.add(idx)
        self.update(StateKey.SELECTED_INDICES, new_set)

    def toggle_selected(self, idx: int) -> None:
        """Toggle boundary in/out of selection (for shift-click)."""
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
        """Clear all selection."""
        self.update(StateKey.SELECTED_INDICES, set())

    def needs_refresh(self) -> bool:
        """Return whether any setting changed since the last consume."""
        return self._needs_refresh

    def consume_refresh(self) -> bool:
        """Reset refresh flags and return whether invalidation is needed.

        Returns:
            ``True`` if the change requires ``invalidate_cache=True``
            (i.e. base_rgb must be recomputed), ``False`` otherwise.
        """
        invalidate = self._needs_invalidate
        self._needs_refresh = False
        self._needs_invalidate = False
        return invalidate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_update(
        self,
        key: StateKey,
        value: Any,
        context: Optional[tuple],
        *,
        set_flags: bool = True,
    ) -> bool:
        """Mutate state under lock, then notify subscribers and optionally set flags.

        Returns whether the update requires cache invalidation.
        """
        requires_invalidation = False

        with self._lock:
            if key is StateKey.REGION_STYLE:
                if context is None:
                    raise ValueError(
                        "REGION_STYLE updates require a "
                        "(boundary_id, region_type) context tuple"
                    )
                requires_invalidation = self._set_region_style(value, context)
            elif key is StateKey.COLOR_CONFIG:
                self._state.color_config = value
                requires_invalidation = True
            elif key is StateKey.VIEW_MODE:
                self._state.view_mode = value
            elif key is StateKey.SELECTED_INDICES:
                self._state.selected_indices = value
            elif key is StateKey.SELECTED_REGION_TYPE:
                self._state.selected_region_type = value
            else:
                self._set_value(key, value)
                requires_invalidation = key in self._INVALIDATE_KEYS

        # Notify subscribers (outside lock to avoid deadlocks).
        self._notify(key, value, context)

        # Set per-frame flags (skipped for debounced updates — deferred
        # to poll_debounce / flush_pending).
        if set_flags and key in self._REFRESH_KEYS:
            self._needs_refresh = True
            if requires_invalidation:
                self._needs_invalidate = True

        return requires_invalidation

    def _set_value(self, key: StateKey, value: Any) -> None:
        """Write *value* to the appropriate DisplaySettings attribute."""
        attr = self._DISPLAY_SETTINGS_MAP.get(key)
        if attr is None:
            raise ValueError(f"No DisplaySettings mapping for {key!r}")
        setattr(self._state.display_settings, attr, value)

    def _set_region_style(
        self,
        updates: dict[str, Any],
        context: tuple,
    ) -> bool:
        """Apply *updates* dict to the RegionStyle for *context*.

        Returns ``True`` if any invalidation-triggering field was touched.
        """
        boundary_id, region_type = context

        if region_type not in self._VALID_REGION_TYPES:
            raise ValueError(
                f"Invalid region_type {region_type!r}, "
                f"expected one of {sorted(self._VALID_REGION_TYPES)}"
            )

        bcs = self._state.boundary_color_settings.get(boundary_id)
        if bcs is None:
            bcs = BoundaryColorSettings()
            self._state.boundary_color_settings[boundary_id] = bcs

        region: object = getattr(bcs, region_type)

        touched_invalidate = False
        for field_name, field_value in updates.items():
            setattr(region, field_name, field_value)
            if field_name in self._REGION_INVALIDATE_FIELDS:
                touched_invalidate = True

        return touched_invalidate

    def _notify(
        self,
        key: StateKey,
        value: Any,
        context: Optional[tuple],
    ) -> None:
        """Call all subscribers registered for *key*, isolating exceptions."""
        callbacks = self._subscribers.get(key)
        if not callbacks:
            return
        for cb in list(callbacks):
            try:
                cb(key, value, context)
            except Exception:
                logger.warning(
                    "Subscriber %r raised for %s", cb, key, exc_info=True,
                )
