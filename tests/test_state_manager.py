"""Tests for elliptica.app.state_manager."""

import threading

import pytest

from elliptica.app.core import AppState, BoundaryColorSettings, RegionStyle
from elliptica.app.state_manager import StateKey, StateManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    return AppState()


@pytest.fixture
def lock():
    return threading.RLock()


@pytest.fixture
def manager(state, lock):
    return StateManager(state, lock)


class FakeClock:
    """Deterministic clock for debounce tests."""

    def __init__(self, start: float = 0.0):
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture
def fake_clock():
    return FakeClock()


@pytest.fixture
def timed_manager(state, lock, fake_clock):
    return StateManager(state, lock, _clock=fake_clock)


# ---------------------------------------------------------------------------
# TestImmediateUpdate
# ---------------------------------------------------------------------------

class TestImmediateUpdate:
    """Verify that each StateKey applies to AppState immediately."""

    def test_brightness(self, state, manager):
        manager.update(StateKey.BRIGHTNESS, 1.5)
        assert state.display_settings.brightness == 1.5

    def test_contrast(self, state, manager):
        manager.update(StateKey.CONTRAST, 2.0)
        assert state.display_settings.contrast == 2.0

    def test_gamma(self, state, manager):
        manager.update(StateKey.GAMMA, 0.5)
        assert state.display_settings.gamma == 0.5

    def test_saturation(self, state, manager):
        manager.update(StateKey.SATURATION, 0.8)
        assert state.display_settings.saturation == 0.8

    def test_clip_low_percent(self, state, manager):
        manager.update(StateKey.CLIP_LOW_PERCENT, 1.0)
        assert state.display_settings.clip_low_percent == 1.0

    def test_clip_high_percent(self, state, manager):
        manager.update(StateKey.CLIP_HIGH_PERCENT, 2.0)
        assert state.display_settings.clip_high_percent == 2.0

    def test_lightness_expr_string(self, state, manager):
        manager.update(StateKey.LIGHTNESS_EXPR, "clipnorm(mag, 1, 99)")
        assert state.display_settings.lightness_expr == "clipnorm(mag, 1, 99)"

    def test_lightness_expr_none(self, state, manager):
        state.display_settings.lightness_expr = "something"
        manager.update(StateKey.LIGHTNESS_EXPR, None)
        assert state.display_settings.lightness_expr is None

    def test_color_enabled(self, state, manager):
        manager.update(StateKey.COLOR_ENABLED, False)
        assert state.display_settings.color_enabled is False

    def test_palette(self, state, manager):
        manager.update(StateKey.PALETTE, "Viridis")
        assert state.display_settings.palette == "Viridis"

    def test_downsample_sigma(self, state, manager):
        manager.update(StateKey.DOWNSAMPLE_SIGMA, 1.2)
        assert state.display_settings.downsample_sigma == 1.2

    def test_color_config(self, state, manager):
        sentinel = object()
        manager.update(StateKey.COLOR_CONFIG, sentinel)
        assert state.color_config is sentinel


# ---------------------------------------------------------------------------
# TestDebouncedUpdate
# ---------------------------------------------------------------------------

class TestDebouncedUpdate:
    """Debounce queue behaviour with injectable clock."""

    def test_not_applied_immediately(self, state, timed_manager):
        timed_manager.update(StateKey.BRIGHTNESS, 9.9, debounce=0.3)
        assert state.display_settings.brightness != 9.9

    def test_applied_after_poll_when_delay_elapsed(self, state, timed_manager, fake_clock):
        timed_manager.update(StateKey.BRIGHTNESS, 9.9, debounce=0.3)
        fake_clock.advance(0.3)
        timed_manager.poll_debounce()
        assert state.display_settings.brightness == 9.9

    def test_not_applied_before_delay(self, state, timed_manager, fake_clock):
        timed_manager.update(StateKey.BRIGHTNESS, 9.9, debounce=0.3)
        fake_clock.advance(0.2)
        timed_manager.poll_debounce()
        assert state.display_settings.brightness != 9.9

    def test_last_value_wins(self, state, timed_manager, fake_clock):
        timed_manager.update(StateKey.BRIGHTNESS, 1.0, debounce=0.3)
        fake_clock.advance(0.1)
        timed_manager.update(StateKey.BRIGHTNESS, 2.0, debounce=0.3)
        fake_clock.advance(0.3)
        timed_manager.poll_debounce()
        assert state.display_settings.brightness == 2.0

    def test_flush_applies_all(self, state, timed_manager):
        timed_manager.update(StateKey.BRIGHTNESS, 5.0, debounce=1.0)
        timed_manager.update(StateKey.CONTRAST, 3.0, debounce=1.0)
        timed_manager.flush_pending()
        assert state.display_settings.brightness == 5.0
        assert state.display_settings.contrast == 3.0

    def test_flush_clears_queue(self, state, timed_manager, fake_clock):
        timed_manager.update(StateKey.BRIGHTNESS, 5.0, debounce=1.0)
        timed_manager.flush_pending()
        # Advance past original delay — poll should have nothing to do.
        fake_clock.advance(2.0)
        state.display_settings.brightness = 0.0
        timed_manager.poll_debounce()
        assert state.display_settings.brightness == 0.0  # unchanged

    def test_timestamp_reset_on_reupdate(self, state, timed_manager, fake_clock):
        timed_manager.update(StateKey.BRIGHTNESS, 1.0, debounce=0.5)
        fake_clock.advance(0.4)
        # Re-update resets the timestamp.
        timed_manager.update(StateKey.BRIGHTNESS, 2.0, debounce=0.5)
        fake_clock.advance(0.4)
        timed_manager.poll_debounce()
        # Only 0.4s since reset — should NOT have applied yet.
        assert state.display_settings.brightness != 2.0
        fake_clock.advance(0.1)
        timed_manager.poll_debounce()
        assert state.display_settings.brightness == 2.0


# ---------------------------------------------------------------------------
# TestRegionStyle
# ---------------------------------------------------------------------------

class TestRegionStyle:
    """REGION_STYLE updates with context tuples."""

    def test_basic_surface(self, state, manager):
        state.boundary_color_settings[1] = BoundaryColorSettings()
        manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.5},
            context=(1, "surface"),
        )
        assert state.boundary_color_settings[1].surface.brightness == 0.5

    def test_basic_interior(self, state, manager):
        state.boundary_color_settings[1] = BoundaryColorSettings()
        manager.update(
            StateKey.REGION_STYLE,
            {"contrast": 1.8},
            context=(1, "interior"),
        )
        assert state.boundary_color_settings[1].interior.contrast == 1.8

    def test_auto_creates_boundary_color_settings(self, state, manager):
        assert 42 not in state.boundary_color_settings
        manager.update(
            StateKey.REGION_STYLE,
            {"gamma": 0.7},
            context=(42, "surface"),
        )
        assert 42 in state.boundary_color_settings
        assert state.boundary_color_settings[42].surface.gamma == 0.7

    def test_multiple_fields_in_one_dict(self, state, manager):
        manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.3, "contrast": 2.0, "use_palette": False},
            context=(1, "surface"),
        )
        region = state.boundary_color_settings[1].surface
        assert region.brightness == 0.3
        assert region.contrast == 2.0
        assert region.use_palette is False

    def test_requires_context(self, manager):
        with pytest.raises(ValueError, match="context"):
            manager.update(StateKey.REGION_STYLE, {"brightness": 0.5})

    def test_independent_contexts(self, state, manager):
        manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.1},
            context=(1, "surface"),
        )
        manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.9},
            context=(1, "interior"),
        )
        assert state.boundary_color_settings[1].surface.brightness == 0.1
        assert state.boundary_color_settings[1].interior.brightness == 0.9

    def test_debounced_region_style(self, state, timed_manager, fake_clock):
        timed_manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.5},
            debounce=0.3,
            context=(1, "surface"),
        )
        assert 1 not in state.boundary_color_settings
        fake_clock.advance(0.3)
        timed_manager.poll_debounce()
        assert state.boundary_color_settings[1].surface.brightness == 0.5


# ---------------------------------------------------------------------------
# TestRefreshFlags
# ---------------------------------------------------------------------------

class TestRefreshFlags:
    """_needs_refresh and _needs_invalidate flag behaviour."""

    def test_set_on_display_key_update(self, manager):
        manager.update(StateKey.BRIGHTNESS, 1.0)
        assert manager.needs_refresh() is True

    def test_invalidate_for_invalidate_keys(self, manager):
        manager.update(StateKey.PALETTE, "Viridis")
        assert manager.consume_refresh() is True  # invalidate=True

    def test_no_invalidate_for_saturation(self, manager):
        manager.update(StateKey.SATURATION, 0.5)
        assert manager.needs_refresh() is True
        assert manager.consume_refresh() is False  # invalidate=False

    def test_invalidate_for_lightness_expr(self, manager):
        manager.update(StateKey.LIGHTNESS_EXPR, "mag")
        assert manager.needs_refresh() is True
        assert manager.consume_refresh() is True

    def test_consume_resets_flags(self, manager):
        manager.update(StateKey.BRIGHTNESS, 1.0)
        manager.consume_refresh()
        assert manager.needs_refresh() is False

    def test_debounced_flag_only_set_after_poll(self, timed_manager, fake_clock):
        timed_manager.update(StateKey.BRIGHTNESS, 1.0, debounce=0.3)
        assert timed_manager.needs_refresh() is False
        fake_clock.advance(0.3)
        timed_manager.poll_debounce()
        assert timed_manager.needs_refresh() is True

    def test_region_style_sets_refresh(self, manager):
        manager.update(
            StateKey.REGION_STYLE,
            {"smear_sigma": 0.01},
            context=(1, "surface"),
        )
        assert manager.needs_refresh() is True

    def test_region_invalidate_fields(self, manager):
        manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.5},
            context=(1, "surface"),
        )
        assert manager.consume_refresh() is True

    def test_region_empty_dict_no_invalidate(self, manager):
        manager.update(
            StateKey.REGION_STYLE,
            {},
            context=(1, "surface"),
        )
        assert manager.needs_refresh() is True
        assert manager.consume_refresh() is False

    def test_color_config_sets_invalidate(self, manager):
        manager.update(StateKey.COLOR_CONFIG, object())
        assert manager.consume_refresh() is True


# ---------------------------------------------------------------------------
# TestSubscribers
# ---------------------------------------------------------------------------

class TestSubscribers:
    """Subscriber notification system."""

    def test_called_on_update(self, manager):
        calls = []
        manager.subscribe(StateKey.BRIGHTNESS, lambda k, v, c: calls.append((k, v, c)))
        manager.update(StateKey.BRIGHTNESS, 1.5)
        assert calls == [(StateKey.BRIGHTNESS, 1.5, None)]

    def test_receives_context(self, manager):
        calls = []
        manager.subscribe(StateKey.REGION_STYLE, lambda k, v, c: calls.append(c))
        ctx = (1, "surface")
        manager.update(StateKey.REGION_STYLE, {"brightness": 0.5}, context=ctx)
        assert calls == [ctx]

    def test_unsubscribe(self, manager):
        calls = []
        cb = lambda k, v, c: calls.append(v)
        manager.subscribe(StateKey.BRIGHTNESS, cb)
        manager.unsubscribe(StateKey.BRIGHTNESS, cb)
        manager.update(StateKey.BRIGHTNESS, 1.5)
        assert calls == []

    def test_multiple_subscribers(self, manager):
        a, b = [], []
        manager.subscribe(StateKey.BRIGHTNESS, lambda k, v, c: a.append(v))
        manager.subscribe(StateKey.BRIGHTNESS, lambda k, v, c: b.append(v))
        manager.update(StateKey.BRIGHTNESS, 1.5)
        assert a == [1.5]
        assert b == [1.5]

    def test_exception_isolation(self, manager):
        calls = []

        def bad_callback(k, v, c):
            raise RuntimeError("boom")

        manager.subscribe(StateKey.BRIGHTNESS, bad_callback)
        manager.subscribe(StateKey.BRIGHTNESS, lambda k, v, c: calls.append(v))
        manager.update(StateKey.BRIGHTNESS, 1.5)
        # Second subscriber still called despite first raising.
        assert calls == [1.5]

    def test_called_after_debounce_not_before(self, timed_manager, fake_clock):
        calls = []
        timed_manager.subscribe(StateKey.BRIGHTNESS, lambda k, v, c: calls.append(v))
        timed_manager.update(StateKey.BRIGHTNESS, 1.5, debounce=0.3)
        assert calls == []
        fake_clock.advance(0.3)
        timed_manager.poll_debounce()
        assert calls == [1.5]

    def test_not_called_for_other_keys(self, manager):
        calls = []
        manager.subscribe(StateKey.BRIGHTNESS, lambda k, v, c: calls.append(v))
        manager.update(StateKey.CONTRAST, 2.0)
        assert calls == []


# ---------------------------------------------------------------------------
# TestFlushPending
# ---------------------------------------------------------------------------

class TestFlushPending:
    """flush_pending() specifics."""

    def test_applies_multiple_keys(self, state, timed_manager):
        timed_manager.update(StateKey.BRIGHTNESS, 5.0, debounce=1.0)
        timed_manager.update(StateKey.GAMMA, 0.3, debounce=1.0)
        timed_manager.flush_pending()
        assert state.display_settings.brightness == 5.0
        assert state.display_settings.gamma == 0.3

    def test_applies_region_styles(self, state, timed_manager):
        timed_manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.2},
            debounce=1.0,
            context=(1, "surface"),
        )
        timed_manager.flush_pending()
        assert state.boundary_color_settings[1].surface.brightness == 0.2

    def test_triggers_subscribers(self, timed_manager):
        calls = []
        timed_manager.subscribe(StateKey.BRIGHTNESS, lambda k, v, c: calls.append(v))
        timed_manager.update(StateKey.BRIGHTNESS, 3.0, debounce=1.0)
        assert calls == []
        timed_manager.flush_pending()
        assert calls == [3.0]

    def test_sets_refresh_flags(self, timed_manager):
        timed_manager.update(StateKey.PALETTE, "Magma", debounce=1.0)
        assert timed_manager.needs_refresh() is False
        timed_manager.flush_pending()
        assert timed_manager.needs_refresh() is True
        assert timed_manager.consume_refresh() is True  # palette -> invalidate

    def test_idempotent_after_flush(self, state, timed_manager):
        timed_manager.update(StateKey.BRIGHTNESS, 5.0, debounce=1.0)
        timed_manager.flush_pending()
        state.display_settings.brightness = 0.0
        timed_manager.flush_pending()  # No-op — queue already empty.
        assert state.display_settings.brightness == 0.0


# ---------------------------------------------------------------------------
# TestDebounceThenImmediate
# ---------------------------------------------------------------------------

class TestDebounceThenImmediate:
    """Immediate update must cancel any pending debounced entry for the same key."""

    def test_immediate_cancels_pending(self, state, timed_manager, fake_clock):
        timed_manager.update(StateKey.BRIGHTNESS, 1.0, debounce=0.5)
        timed_manager.update(StateKey.BRIGHTNESS, 2.0)  # immediate
        assert state.display_settings.brightness == 2.0
        # The debounced entry should have been cleared.
        fake_clock.advance(1.0)
        timed_manager.poll_debounce()
        assert state.display_settings.brightness == 2.0  # not reverted

    def test_immediate_cancels_pending_region_style(self, state, timed_manager, fake_clock):
        ctx = (1, "surface")
        timed_manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.1},
            debounce=0.5,
            context=ctx,
        )
        timed_manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.9},
            context=ctx,
        )
        assert state.boundary_color_settings[1].surface.brightness == 0.9
        fake_clock.advance(1.0)
        timed_manager.poll_debounce()
        assert state.boundary_color_settings[1].surface.brightness == 0.9

    def test_other_pending_keys_unaffected(self, state, timed_manager, fake_clock):
        timed_manager.update(StateKey.BRIGHTNESS, 1.0, debounce=0.5)
        timed_manager.update(StateKey.CONTRAST, 3.0, debounce=0.5)
        # Immediate update for brightness only.
        timed_manager.update(StateKey.BRIGHTNESS, 2.0)
        assert state.display_settings.brightness == 2.0
        # Contrast should still be pending.
        fake_clock.advance(0.5)
        timed_manager.poll_debounce()
        assert state.display_settings.contrast == 3.0


# ---------------------------------------------------------------------------
# TestPollDebounceExceptionHandling
# ---------------------------------------------------------------------------

class TestPollDebounceExceptionHandling:
    """poll_debounce processes all ready entries even if one raises."""

    def test_sibling_entries_still_applied(self, state, timed_manager, fake_clock):
        # Queue a REGION_STYLE with missing context (will raise ValueError)
        # and a normal BRIGHTNESS update. Both should be ready at the same time.
        # We need to manually inject the bad entry since update() validates
        # debounce path doesn't call _apply_update.
        from elliptica.app.state_manager import _PendingEntry

        timed_manager._pending[(StateKey.REGION_STYLE, None)] = _PendingEntry(
            value={"brightness": 0.5}, timestamp=0.0, delay=0.3,
        )
        timed_manager.update(StateKey.BRIGHTNESS, 5.0, debounce=0.3)
        fake_clock.advance(0.3)

        # poll_debounce should apply BRIGHTNESS even though REGION_STYLE raises.
        # The exception is logged, not propagated.
        timed_manager.poll_debounce()
        assert state.display_settings.brightness == 5.0

    def test_bad_entry_consumed_from_queue(self, state, timed_manager, fake_clock):
        from elliptica.app.state_manager import _PendingEntry

        timed_manager._pending[(StateKey.REGION_STYLE, None)] = _PendingEntry(
            value={"brightness": 0.5}, timestamp=0.0, delay=0.3,
        )
        fake_clock.advance(0.3)

        # The exception is caught and logged inside poll_debounce.
        timed_manager.poll_debounce()

        # The bad entry should have been consumed (not retried on next poll).
        assert (StateKey.REGION_STYLE, None) not in timed_manager._pending


# ---------------------------------------------------------------------------
# TestRegionTypeValidation
# ---------------------------------------------------------------------------

class TestRegionTypeValidation:
    """region_type must be 'surface' or 'interior'."""

    def test_invalid_region_type_raises(self, manager):
        with pytest.raises(ValueError, match="Invalid region_type"):
            manager.update(
                StateKey.REGION_STYLE,
                {"brightness": 0.5},
                context=(1, "__class__"),
            )

    def test_surface_accepted(self, state, manager):
        manager.update(
            StateKey.REGION_STYLE,
            {"brightness": 0.5},
            context=(1, "surface"),
        )
        assert state.boundary_color_settings[1].surface.brightness == 0.5

    def test_interior_accepted(self, state, manager):
        manager.update(
            StateKey.REGION_STYLE,
            {"contrast": 1.5},
            context=(1, "interior"),
        )
        assert state.boundary_color_settings[1].interior.contrast == 1.5


# ---------------------------------------------------------------------------
# TestInvalidateKeysParametrized
# ---------------------------------------------------------------------------

class TestInvalidateKeysParametrized:
    """All _INVALIDATE_KEYS members trigger invalidation."""

    @pytest.mark.parametrize("key,value", [
        (StateKey.BRIGHTNESS, 1.5),
        (StateKey.CONTRAST, 2.0),
        (StateKey.GAMMA, 0.5),
        (StateKey.CLIP_LOW_PERCENT, 1.0),
        (StateKey.CLIP_HIGH_PERCENT, 2.0),
        (StateKey.LIGHTNESS_EXPR, "mag"),
        (StateKey.COLOR_ENABLED, False),
        (StateKey.PALETTE, "Viridis"),
    ])
    def test_invalidate_key(self, manager, key, value):
        manager.update(key, value)
        assert manager.consume_refresh() is True

    @pytest.mark.parametrize("key,value", [
        (StateKey.SATURATION, 0.5),
        (StateKey.DOWNSAMPLE_SIGMA, 0.8),
    ])
    def test_non_invalidate_key(self, manager, key, value):
        manager.update(key, value)
        assert manager.needs_refresh() is True
        assert manager.consume_refresh() is False


# ---------------------------------------------------------------------------
# TestRegionInvalidateFieldsParametrized
# ---------------------------------------------------------------------------

class TestRegionInvalidateFieldsParametrized:
    """All _REGION_INVALIDATE_FIELDS members trigger invalidation."""

    @pytest.mark.parametrize("field,value", [
        ("enabled", True),
        ("brightness", 0.5),
        ("contrast", 2.0),
        ("gamma", 0.5),
        ("lightness_expr", "mag"),
        ("use_palette", False),
        ("palette", "Plasma"),
        ("solid_color", (1.0, 0.0, 0.0)),
        ("smear_enabled", True),
        ("smear_sigma", 0.01),
    ])
    def test_invalidate_field(self, manager, field, value):
        manager.update(
            StateKey.REGION_STYLE,
            {field: value},
            context=(1, "surface"),
        )
        assert manager.consume_refresh() is True


# ---------------------------------------------------------------------------
# TestUnsubscribeEdgeCases
# ---------------------------------------------------------------------------

class TestUnsubscribeEdgeCases:
    """Edge cases in subscribe/unsubscribe."""

    def test_unsubscribe_never_registered(self, manager):
        cb = lambda k, v, c: None
        # Should not raise.
        manager.unsubscribe(StateKey.BRIGHTNESS, cb)

    def test_unsubscribe_no_subscribers_for_key(self, manager):
        cb = lambda k, v, c: None
        # Key has no subscriber list at all.
        manager.unsubscribe(StateKey.GAMMA, cb)

    def test_double_unsubscribe(self, manager):
        cb = lambda k, v, c: None
        manager.subscribe(StateKey.BRIGHTNESS, cb)
        manager.unsubscribe(StateKey.BRIGHTNESS, cb)
        # Second unsubscribe should not raise.
        manager.unsubscribe(StateKey.BRIGHTNESS, cb)

    def test_subscribe_during_notify(self, manager):
        late_calls = []
        late_cb = lambda k, v, c: late_calls.append(v)

        def subscribing_cb(k, v, c):
            manager.subscribe(StateKey.BRIGHTNESS, late_cb)

        manager.subscribe(StateKey.BRIGHTNESS, subscribing_cb)
        manager.update(StateKey.BRIGHTNESS, 1.0)
        # late_cb was added during notification — should NOT be called this round.
        assert late_calls == []
        # But should be called on the next update.
        manager.update(StateKey.BRIGHTNESS, 2.0)
        assert late_calls == [2.0]

    def test_unsubscribe_during_notify(self, manager):
        calls_b = []
        cb_b = lambda k, v, c: calls_b.append(v)

        def unsubscribing_cb(k, v, c):
            manager.unsubscribe(StateKey.BRIGHTNESS, cb_b)

        manager.subscribe(StateKey.BRIGHTNESS, unsubscribing_cb)
        manager.subscribe(StateKey.BRIGHTNESS, cb_b)
        manager.update(StateKey.BRIGHTNESS, 1.0)
        # cb_b still fires this round (snapshot iteration).
        assert calls_b == [1.0]
        # But not on the next update.
        manager.update(StateKey.BRIGHTNESS, 2.0)
        assert calls_b == [1.0]


# ---------------------------------------------------------------------------
# TestDebounceBoundaryConditions
# ---------------------------------------------------------------------------

class TestDebounceBoundaryConditions:
    """Boundary conditions for debounce parameter."""

    def test_debounce_zero_is_immediate(self, state, manager):
        manager.update(StateKey.BRIGHTNESS, 3.0, debounce=0.0)
        assert state.display_settings.brightness == 3.0

    def test_negative_debounce_is_immediate(self, state, manager):
        manager.update(StateKey.BRIGHTNESS, 4.0, debounce=-1.0)
        assert state.display_settings.brightness == 4.0

    def test_poll_debounce_on_empty_queue(self, timed_manager):
        # Should be a safe no-op.
        timed_manager.poll_debounce()
        assert timed_manager.needs_refresh() is False

    def test_flush_pending_on_empty_queue(self, timed_manager):
        timed_manager.flush_pending()
        assert timed_manager.needs_refresh() is False

    def test_double_consume_refresh(self, manager):
        manager.update(StateKey.BRIGHTNESS, 1.0)
        assert manager.consume_refresh() is True
        # Second consume should return False (flags already cleared).
        assert manager.consume_refresh() is False
        assert manager.needs_refresh() is False
