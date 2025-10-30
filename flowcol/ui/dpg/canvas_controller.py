"""Canvas interaction controller for FlowCol UI - mouse, keyboard, hit detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import math

from flowcol import defaults
from flowcol.app import actions
from flowcol.types import Conductor

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg  # type: ignore
except ImportError:
    dpg = None


def _point_in_conductor(conductor: Conductor, x: float, y: float) -> bool:
    """Test whether canvas coordinate lands inside conductor mask."""
    cx, cy = conductor.position
    local_x = int(round(x - cx))
    local_y = int(round(y - cy))
    mask = conductor.mask
    if local_x < 0 or local_y < 0:
        return False
    if local_y >= mask.shape[0] or local_x >= mask.shape[1]:
        return False
    return mask[local_y, local_x] > 0.5


def _clone_conductor(conductor: Conductor) -> Conductor:
    """Deep-copy conductor data for clipboard."""
    interior = None
    if conductor.interior_mask is not None:
        interior = conductor.interior_mask.copy()
    original = None
    if conductor.original_mask is not None:
        original = conductor.original_mask.copy()
    original_interior = None
    if conductor.original_interior_mask is not None:
        original_interior = conductor.original_interior_mask.copy()
    return Conductor(
        mask=conductor.mask.copy(),
        voltage=conductor.voltage,
        position=conductor.position,
        interior_mask=interior,
        original_mask=original,
        original_interior_mask=original_interior,
        scale_factor=conductor.scale_factor,
        blur_sigma=conductor.blur_sigma,
        blur_is_fractional=conductor.blur_is_fractional,
        smear_enabled=conductor.smear_enabled,
        smear_sigma=conductor.smear_sigma,
    )


class CanvasController:
    """Handles canvas mouse/keyboard input, hit detection, and conductor manipulation."""

    def __init__(self, app: "FlowColApp"):
        self.app = app

        # Mouse/drag state
        self.drag_active: bool = False
        self.drag_last_pos: Tuple[float, float] = (0.0, 0.0)
        self.mouse_down_last: bool = False
        self.mouse_wheel_delta: float = 0.0

        # Keyboard state
        self.backspace_down_last: bool = False
        self.ctrl_c_down_last: bool = False
        self.ctrl_v_down_last: bool = False

        # Selection state
        self.selected_region: Optional[str] = None  # "surface" or "interior"
        self.clipboard_conductor: Optional[Conductor] = None

    def on_mouse_wheel(self, sender, app_data) -> None:
        """Capture mouse wheel delta from DPG handler."""
        self.mouse_wheel_delta = float(app_data)

    def is_mouse_over_canvas(self) -> bool:
        """Check if mouse is within canvas bounds."""
        if dpg is None or self.app.canvas_id is None:
            return False

        # Use absolute coordinates for hit testing
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.app.canvas_id)
        rect_max = dpg.get_item_rect_max(self.app.canvas_id)

        return (rect_min[0] <= mouse_x <= rect_max[0] and
                rect_min[1] <= mouse_y <= rect_max[1])

    def get_canvas_mouse_pos(self) -> Tuple[float, float]:
        """Convert screen mouse position to canvas coordinates."""
        assert dpg is not None and self.app.canvas_id is not None
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.app.canvas_id)
        # Get screen-space coordinates relative to canvas
        screen_x = mouse_x - rect_min[0]
        screen_y = mouse_y - rect_min[1]
        # Apply inverse scale to get canvas-space coordinates
        canvas_x = screen_x / self.app.display_scale if self.app.display_scale > 0 else screen_x
        canvas_y = screen_y / self.app.display_scale if self.app.display_scale > 0 else screen_y
        return canvas_x, canvas_y

    def find_hit_conductor(self, x: float, y: float) -> int:
        """Find which conductor (if any) is at canvas coordinates. Returns index or -1."""
        with self.app.state_lock:
            conductors = self.app.state.project.conductors
            for idx in reversed(range(len(conductors))):
                if _point_in_conductor(conductors[idx], x, y):
                    return idx
        return -1

    def detect_region_at_point(self, canvas_x: float, canvas_y: float) -> tuple[int, Optional[str]]:
        """Detect which conductor region is at canvas point.

        Returns (conductor_idx, region) where region is "surface" or "interior" or None.
        """
        cache = self.app.state.render_cache
        if cache is None or cache.conductor_masks is None or cache.interior_masks is None:
            return -1, None

        # Masks are at canvas_resolution (same as canvas), so coordinates map directly
        # The render texture may be thumbnailed for display, but masks are full resolution
        mask_x = int(canvas_x)
        mask_y = int(canvas_y)

        # Check each conductor in reverse order (top to bottom)
        for idx in reversed(range(len(self.app.state.project.conductors))):
            if idx >= len(cache.interior_masks) or idx >= len(cache.conductor_masks):
                continue

            # Check interior first (it's inside, so higher priority)
            interior_mask = cache.interior_masks[idx]
            if interior_mask is not None:
                h, w = interior_mask.shape
                if 0 <= mask_y < h and 0 <= mask_x < w:
                    if interior_mask[mask_y, mask_x] > 0.5:
                        return idx, "interior"

            # Check surface
            surface_mask = cache.conductor_masks[idx]
            if surface_mask is not None:
                h, w = surface_mask.shape
                if 0 <= mask_y < h and 0 <= mask_x < w:
                    if surface_mask[mask_y, mask_x] > 0.5:
                        return idx, "surface"

        return -1, None

    def scale_conductor(self, idx: int, factor: float) -> bool:
        """Scale conductor by factor. Returns True if successful."""
        if dpg is None:
            return False
        factor = max(float(factor), 0.05)
        with self.app.state_lock:
            self.app.state.set_selected(idx)
            changed = actions.scale_conductor(self.app.state, idx, factor)
            if changed:
                self.app.texture_manager.clear_conductor_texture(idx)
        if not changed:
            dpg.set_value("status_text", "Scaling limit reached.")
            return False

        self.app._mark_canvas_dirty()
        self.app.conductor_controls.update_conductor_slider_labels()
        self.app.postprocess_panel.update_region_properties_panel()
        dpg.set_value("status_text", f"Scaled C{idx + 1} by {factor:.2f}×")
        return True

    def process_canvas_mouse(self) -> None:
        """Process mouse input on canvas (clicks, drags, wheel)."""
        if dpg is None or self.app.canvas_id is None:
            return

        mouse_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        pressed = mouse_down and not self.mouse_down_last
        released = (not mouse_down) and self.mouse_down_last

        with self.app.state_lock:
            mode = self.app.state.view_mode

        wheel_delta = self.mouse_wheel_delta
        self.mouse_wheel_delta = 0.0
        over_canvas = self.is_mouse_over_canvas()

        # Mouse wheel scaling in edit mode
        if mode == "edit" and over_canvas and abs(wheel_delta) > 1e-5:
            with self.app.state_lock:
                selected_idx = self.app.state.selected_idx
            # Only scale if a conductor is selected
            if selected_idx >= 0:
                scale_factor = math.exp(wheel_delta * defaults.SCROLL_SCALE_SENSITIVITY)
                scale_factor = max(0.05, min(scale_factor, 20.0))
                if self.scale_conductor(selected_idx, scale_factor):
                    x, y = self.get_canvas_mouse_pos()
                    self.drag_last_pos = (x, y)

        # Render mode: click to select region for colorization
        if mode != "edit":
            if pressed and self.is_mouse_over_canvas():
                x, y = self.get_canvas_mouse_pos()
                with self.app.state_lock:
                    # Use region detection in render mode for colorization
                    hit_idx, hit_region = self.detect_region_at_point(x, y)
                    self.app.state.set_selected(hit_idx)
                    self.selected_region = hit_region
                self.app.conductor_controls.update_conductor_slider_labels()
                self.app.postprocess_panel.update_region_properties_panel()
            self.mouse_down_last = mouse_down
            return

        # Edit mode: click to select and drag to move
        if pressed and self.is_mouse_over_canvas():
            x, y = self.get_canvas_mouse_pos()
            with self.app.state_lock:
                project = self.app.state.project
                hit_idx = -1
                for idx in reversed(range(len(project.conductors))):
                    if _point_in_conductor(project.conductors[idx], x, y):
                        hit_idx = idx
                        break

                self.app.state.set_selected(hit_idx)
                self.selected_region = None  # Region detection only in render mode
                if hit_idx >= 0:
                    self.drag_active = True
                    self.drag_last_pos = (x, y)
                else:
                    self.drag_active = False
            self.app.conductor_controls.update_conductor_slider_labels()
            self.app.postprocess_panel.update_region_properties_panel()
            self.app._mark_canvas_dirty()

        # Drag conductor
        if self.drag_active and mouse_down:
            x, y = self.get_canvas_mouse_pos()
            dx = x - self.drag_last_pos[0]
            dy = y - self.drag_last_pos[1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                with self.app.state_lock:
                    idx = self.app.state.selected_idx
                    if 0 <= idx < len(self.app.state.project.conductors):
                        actions.move_conductor(self.app.state, idx, dx, dy)
                self.drag_last_pos = (x, y)
                self.app._mark_canvas_dirty()

        if self.drag_active and released:
            self.drag_active = False

        self.mouse_down_last = mouse_down

    def process_keyboard_shortcuts(self) -> None:
        """Process keyboard shortcuts (delete, copy, paste)."""
        # Import key constants
        from flowcol.ui.dpg.app import BACKSPACE_KEY, CTRL_KEY, C_KEY, V_KEY

        if dpg is None or BACKSPACE_KEY is None:
            return

        # Only process shortcuts when mouse is over canvas (not over UI controls)
        # This prevents keyboard shortcuts from firing while typing in input fields
        over_canvas = self.is_mouse_over_canvas()
        if not over_canvas:
            self.backspace_down_last = False
            return

        # Backspace to delete
        backspace_down = dpg.is_key_down(BACKSPACE_KEY)
        if backspace_down and not self.backspace_down_last:
            with self.app.state_lock:
                mode = self.app.state.view_mode
                idx = self.app.state.selected_idx
                conductor_count = len(self.app.state.project.conductors)
                can_delete = (mode == "edit" and 0 <= idx < conductor_count)
            if can_delete:
                with self.app.state_lock:
                    actions.remove_conductor(self.app.state, idx)
                    self.app.texture_manager.clear_all_conductor_textures()
                self.app._mark_canvas_dirty()
                self.app.conductor_controls.rebuild_conductor_controls()
                dpg.set_value("status_text", "Conductor deleted")
        self.backspace_down_last = backspace_down

        # Ctrl+C to copy
        if CTRL_KEY and C_KEY:
            ctrl_down = dpg.is_key_down(CTRL_KEY)
            c_down = dpg.is_key_down(C_KEY)
            ctrl_c_down = ctrl_down and c_down
            if ctrl_c_down and not self.ctrl_c_down_last:
                with self.app.state_lock:
                    mode = self.app.state.view_mode
                    idx = self.app.state.selected_idx
                    conductor_count = len(self.app.state.project.conductors)
                    can_copy = (mode == "edit" and 0 <= idx < conductor_count)
                if can_copy:
                    with self.app.state_lock:
                        self.clipboard_conductor = _clone_conductor(self.app.state.project.conductors[idx])
                    dpg.set_value("status_text", f"Copied C{idx + 1}")
            self.ctrl_c_down_last = ctrl_c_down

        # Ctrl+V to paste
        if CTRL_KEY and V_KEY:
            ctrl_down = dpg.is_key_down(CTRL_KEY)
            v_down = dpg.is_key_down(V_KEY)
            ctrl_v_down = ctrl_down and v_down
            if ctrl_v_down and not self.ctrl_v_down_last:
                with self.app.state_lock:
                    mode = self.app.state.view_mode
                    can_paste = (mode == "edit" and self.clipboard_conductor is not None)
                if can_paste:
                    with self.app.state_lock:
                        # Clone the clipboard conductor and offset it
                        pasted = _clone_conductor(self.clipboard_conductor)
                        # Offset by 30px down-right so it's visible
                        px, py = pasted.position
                        pasted.position = (px + 30.0, py + 30.0)
                        actions.add_conductor(self.app.state, pasted)
                        new_idx = len(self.app.state.project.conductors) - 1
                        self.app.state.set_selected(new_idx)
                    self.app._mark_canvas_dirty()
                    self.app.conductor_controls.rebuild_conductor_controls()
                    self.app.conductor_controls.update_conductor_slider_labels()
                    dpg.set_value("status_text", f"Pasted as C{new_idx + 1}")
            self.ctrl_v_down_last = ctrl_v_down
