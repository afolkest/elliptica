"""Canvas interaction controller for Elliptica UI - mouse, keyboard, hit detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import math

from elliptica import defaults
from elliptica.app import actions
from elliptica.types import Conductor

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg  # type: ignore
except ImportError:
    dpg = None


MIN_HIT_TARGET = 10  # Minimum clickable area in pixels


def _point_in_conductor(conductor: Conductor, x: float, y: float) -> bool:
    """Test whether canvas coordinate lands inside conductor hit area.

    For small conductors, expands bounding box to MIN_HIT_TARGET for easier selection.
    For larger conductors, uses pixel-perfect mask testing.
    """
    cx, cy = conductor.position
    mask = conductor.mask
    h, w = mask.shape

    # Calculate expanded bounds for small conductors
    expand_x = max(0, (MIN_HIT_TARGET - w) / 2)
    expand_y = max(0, (MIN_HIT_TARGET - h) / 2)

    # Check expanded bounding box first
    if x < cx - expand_x or x >= cx + w + expand_x:
        return False
    if y < cy - expand_y or y >= cy + h + expand_y:
        return False

    # For small conductors, bounding box hit is enough
    if w <= MIN_HIT_TARGET or h <= MIN_HIT_TARGET:
        return True

    # For larger conductors, do pixel-perfect test
    local_x = int(round(x - cx))
    local_y = int(round(y - cy))
    if local_x < 0 or local_y < 0:
        return False
    if local_y >= h or local_x >= w:
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
        edge_smooth_sigma=conductor.edge_smooth_sigma,
        smear_enabled=conductor.smear_enabled,
        smear_sigma=conductor.smear_sigma,
    )


class CanvasController:
    """Handles canvas mouse/keyboard input, hit detection, and conductor manipulation."""

    def __init__(self, app: "EllipticaApp"):
        self.app = app

        # Mouse/drag state
        self.drag_active: bool = False
        self.drag_last_pos: Tuple[float, float] = (0.0, 0.0)
        self.mouse_down_last: bool = False
        self.mouse_wheel_delta: float = 0.0

        # Box selection state
        self.box_select_active: bool = False
        self.box_select_start: Tuple[float, float] = (0.0, 0.0)
        self.box_select_end: Tuple[float, float] = (0.0, 0.0)

        # Keyboard state
        self.backspace_down_last: bool = False
        self.ctrl_c_down_last: bool = False
        self.ctrl_v_down_last: bool = False
        self.shift_down: bool = False

        # Clipboard (list for multi-select copy)
        self.clipboard_conductors: list[Conductor] = []

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

    def find_conductors_in_box(self, x1: float, y1: float, x2: float, y2: float) -> set[int]:
        """Find all conductors with actual mask pixels inside selection box."""
        # Normalize box coordinates
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        result = set()
        with self.app.state_lock:
            conductors = self.app.state.project.conductors
            for idx, conductor in enumerate(conductors):
                cx, cy = conductor.position
                mask = conductor.mask
                h, w = mask.shape

                # Quick bounding box rejection
                if cx + w < min_x or cx > max_x or cy + h < min_y or cy > max_y:
                    continue

                # Calculate overlap region in mask-local coordinates
                local_min_x = max(0, int(min_x - cx))
                local_max_x = min(w, int(max_x - cx) + 1)
                local_min_y = max(0, int(min_y - cy))
                local_max_y = min(h, int(max_y - cy) + 1)

                # Check if any mask pixels in overlap region are set
                if local_max_x > local_min_x and local_max_y > local_min_y:
                    region = mask[local_min_y:local_max_y, local_min_x:local_max_x]
                    if region.max() > 0.5:
                        result.add(idx)
        return result

    def detect_region_at_point(self, canvas_x: float, canvas_y: float) -> tuple[int, Optional[str]]:
        """Detect which conductor region is at canvas point.

        Returns (conductor_idx, region) where region is "surface" or "interior" or None.
        """
        cache = self.app.state.render_cache
        if cache is None or cache.conductor_masks is None or cache.interior_masks is None:
            return -1, None

        # Masks are at FULL RENDER RESOLUTION (canvas × multiplier × supersample)
        # Scale canvas coordinates to match mask resolution
        scale = cache.multiplier * cache.supersample
        mask_x = int(canvas_x * scale)
        mask_y = int(canvas_y * scale)

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

    def scale_selected_conductors(self, factor: float) -> bool:
        """Scale all selected conductors by factor. Returns True if any scaled."""
        if dpg is None:
            return False
        factor = max(float(factor), 0.05)
        any_changed = False
        with self.app.state_lock:
            indices = list(self.app.state.selected_indices)
            for idx in indices:
                changed = actions.scale_conductor(self.app.state, idx, factor)
                if changed:
                    self.app.display_pipeline.texture_manager.clear_conductor_texture(idx)
                    any_changed = True

        if not any_changed:
            dpg.set_value("status_text", "Scaling limit reached.")
            return False

        self.app.canvas_renderer.mark_dirty()
        self.app.boundary_controls.update_slider_labels()
        self.app.postprocess_panel.update_region_properties_panel()
        count = len(indices)
        if count == 1:
            dpg.set_value("status_text", f"Scaled C{indices[0] + 1} by {factor:.2f}×")
        else:
            dpg.set_value("status_text", f"Scaled {count} objects by {factor:.2f}×")
        return True

    def is_file_dialog_showing(self) -> bool:
        """Check if any file dialog is currently showing."""
        if dpg is None:
            return False

        file_io = self.app.file_io
        dialogs = [
            file_io.load_project_dialog_id,
            file_io.save_project_dialog_id,
            file_io.conductor_file_dialog_id,
        ]

        for dialog_id in dialogs:
            if dialog_id is not None and dpg.does_item_exist(dialog_id):
                if dpg.is_item_shown(dialog_id):
                    return True
        return False

    def is_modal_open(self) -> bool:
        """Check if any modal dialog is open that should block canvas input."""
        if dpg is None:
            return False

        # Render settings modal
        if self.app.render_modal.modal_open:
            return True

        # Export modal
        export = self.app.image_export
        if export.export_modal_id is not None and dpg.does_item_exist(export.export_modal_id):
            if dpg.is_item_shown(export.export_modal_id):
                return True

        # Overwrite confirmation modal
        if export._overwrite_confirm_id is not None and dpg.does_item_exist(export._overwrite_confirm_id):
            if dpg.is_item_shown(export._overwrite_confirm_id):
                return True

        return False

    def process_canvas_mouse(self) -> None:
        """Process mouse input on canvas (clicks, drags, wheel)."""
        if dpg is None or self.app.canvas_id is None:
            return

        # Don't process canvas input when file dialogs or modals are open
        if self.is_file_dialog_showing() or self.is_modal_open():
            return

        mouse_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        pressed = mouse_down and not self.mouse_down_last
        released = (not mouse_down) and self.mouse_down_last

        # Track shift key for multi-select
        from elliptica.ui.dpg.app import SHIFT_KEY
        self.shift_down = SHIFT_KEY is not None and dpg.is_key_down(SHIFT_KEY)

        with self.app.state_lock:
            mode = self.app.state.view_mode

        wheel_delta = self.mouse_wheel_delta
        self.mouse_wheel_delta = 0.0
        over_canvas = self.is_mouse_over_canvas()

        # Mouse wheel scaling in edit mode
        if mode == "edit" and over_canvas and abs(wheel_delta) > 1e-5:
            with self.app.state_lock:
                has_selection = len(self.app.state.selected_indices) > 0
            if has_selection:
                scale_factor = math.exp(wheel_delta * defaults.SCROLL_SCALE_SENSITIVITY)
                scale_factor = max(0.05, min(scale_factor, 20.0))
                if self.scale_selected_conductors(scale_factor):
                    x, y = self.get_canvas_mouse_pos()
                    self.drag_last_pos = (x, y)

        # Render mode: click to select region for colorization (single-select only)
        if mode != "edit":
            if pressed and self.is_mouse_over_canvas():
                x, y = self.get_canvas_mouse_pos()
                with self.app.state_lock:
                    hit_idx, hit_region = self.detect_region_at_point(x, y)
                    self.app.state.set_selected(hit_idx)
                    if hit_region is not None:
                        self.app.state.selected_region_type = hit_region
                self.app.canvas_renderer.invalidate_selection_contour()
                self.app.boundary_controls.update_header_labels()
                self.app.postprocess_panel.update_region_properties_panel()
                self.app.canvas_renderer.mark_dirty()
            self.mouse_down_last = mouse_down
            return

        # Edit mode: click/shift-click to select, drag to move, box select from empty space
        if pressed and self.is_mouse_over_canvas():
            x, y = self.get_canvas_mouse_pos()
            hit_idx = self.find_hit_conductor(x, y)

            with self.app.state_lock:
                if hit_idx >= 0:
                    # Clicked on a conductor
                    if self.shift_down:
                        # Shift-click: toggle selection
                        self.app.state.toggle_selected(hit_idx)
                    else:
                        # Normal click: if clicking on already-selected, keep selection for drag
                        # Otherwise, replace selection with clicked conductor
                        if hit_idx not in self.app.state.selected_indices:
                            self.app.state.set_selected(hit_idx)
                    self.app.state.selected_region_type = "surface"
                    self.drag_active = True
                    self.drag_last_pos = (x, y)
                else:
                    # Clicked on empty space
                    if not self.shift_down:
                        # Clear selection and start box select
                        self.app.state.clear_selection()
                    # Start box selection
                    self.box_select_active = True
                    self.box_select_start = (x, y)
                    self.box_select_end = (x, y)
                    self.drag_active = False

            self.app.boundary_controls.update_header_labels()
            self.app.postprocess_panel.update_region_properties_panel()
            self.app.canvas_renderer.mark_dirty()

        # Update box selection or drag
        if mouse_down and over_canvas:
            x, y = self.get_canvas_mouse_pos()

            if self.box_select_active:
                # Update box selection rectangle
                self.box_select_end = (x, y)
                self.app.canvas_renderer.mark_dirty()

            elif self.drag_active:
                # Drag all selected conductors
                dx = x - self.drag_last_pos[0]
                dy = y - self.drag_last_pos[1]
                if abs(dx) > 0.1 or abs(dy) > 0.1:
                    with self.app.state_lock:
                        for idx in self.app.state.selected_indices:
                            if 0 <= idx < len(self.app.state.project.conductors):
                                actions.move_conductor(self.app.state, idx, dx, dy)
                    self.drag_last_pos = (x, y)
                    self.app.canvas_renderer.mark_dirty()

        # Handle release
        if released:
            if self.box_select_active:
                # Finish box selection
                x, y = self.get_canvas_mouse_pos()
                self.box_select_end = (x, y)
                hits = self.find_conductors_in_box(
                    self.box_select_start[0], self.box_select_start[1],
                    self.box_select_end[0], self.box_select_end[1]
                )
                with self.app.state_lock:
                    if self.shift_down:
                        # Add to existing selection
                        self.app.state.selected_indices.update(hits)
                    else:
                        # Replace selection
                        self.app.state.selected_indices = hits
                self.box_select_active = False
                self.app.boundary_controls.update_header_labels()
                self.app.postprocess_panel.update_region_properties_panel()
                self.app.canvas_renderer.mark_dirty()

            self.drag_active = False

        self.mouse_down_last = mouse_down

    def process_keyboard_shortcuts(self) -> None:
        """Process keyboard shortcuts (delete, copy, paste)."""
        from elliptica.ui.dpg.app import BACKSPACE_KEY, CTRL_KEY, C_KEY, V_KEY

        if dpg is None or BACKSPACE_KEY is None:
            return

        # Don't process keyboard shortcuts when file dialogs or modals are open
        if self.is_file_dialog_showing() or self.is_modal_open():
            self.backspace_down_last = False
            self.ctrl_c_down_last = False
            self.ctrl_v_down_last = False
            return

        # Only process shortcuts when mouse is over canvas (not over UI controls)
        over_canvas = self.is_mouse_over_canvas()
        if not over_canvas:
            self.backspace_down_last = False
            self.ctrl_c_down_last = False
            self.ctrl_v_down_last = False
            return

        # Backspace to delete all selected
        backspace_down = dpg.is_key_down(BACKSPACE_KEY)
        if backspace_down and not self.backspace_down_last:
            with self.app.state_lock:
                mode = self.app.state.view_mode
                indices_to_delete = sorted(self.app.state.selected_indices, reverse=True)
                can_delete = mode == "edit" and len(indices_to_delete) > 0
            if can_delete:
                with self.app.state_lock:
                    # Delete in reverse order to maintain valid indices
                    for idx in indices_to_delete:
                        actions.remove_conductor(self.app.state, idx)
                    self.app.display_pipeline.texture_manager.clear_all_conductor_textures()
                self.app.canvas_renderer.mark_dirty()
                self.app.boundary_controls.rebuild_controls()
                count = len(indices_to_delete)
                msg = "Conductor deleted" if count == 1 else f"{count} conductors deleted"
                dpg.set_value("status_text", msg)
        self.backspace_down_last = backspace_down

        # Ctrl+C to copy all selected
        if CTRL_KEY and C_KEY:
            ctrl_down = dpg.is_key_down(CTRL_KEY)
            c_down = dpg.is_key_down(C_KEY)
            ctrl_c_down = ctrl_down and c_down
            if ctrl_c_down and not self.ctrl_c_down_last:
                with self.app.state_lock:
                    mode = self.app.state.view_mode
                    selected = list(self.app.state.selected_indices)
                    can_copy = mode == "edit" and len(selected) > 0
                if can_copy:
                    with self.app.state_lock:
                        self.clipboard_conductors = [
                            _clone_conductor(self.app.state.project.conductors[idx])
                            for idx in sorted(selected)
                        ]
                    count = len(self.clipboard_conductors)
                    msg = f"Copied C{selected[0] + 1}" if count == 1 else f"Copied {count} objects"
                    dpg.set_value("status_text", msg)
            self.ctrl_c_down_last = ctrl_c_down

        # Ctrl+V to paste all copied (maintaining relative positions)
        if CTRL_KEY and V_KEY:
            ctrl_down = dpg.is_key_down(CTRL_KEY)
            v_down = dpg.is_key_down(V_KEY)
            ctrl_v_down = ctrl_down and v_down
            if ctrl_v_down and not self.ctrl_v_down_last:
                with self.app.state_lock:
                    mode = self.app.state.view_mode
                    can_paste = mode == "edit" and len(self.clipboard_conductors) > 0
                if can_paste:
                    with self.app.state_lock:
                        new_indices = set()
                        for conductor in self.clipboard_conductors:
                            pasted = _clone_conductor(conductor)
                            # Offset by 30px down-right
                            px, py = pasted.position
                            pasted.position = (px + 30.0, py + 30.0)
                            actions.add_conductor(self.app.state, pasted)
                            new_indices.add(len(self.app.state.project.conductors) - 1)
                        # Select all pasted conductors
                        self.app.state.selected_indices = new_indices
                    self.app.canvas_renderer.mark_dirty()
                    self.app.boundary_controls.rebuild_controls()
                    self.app.boundary_controls.update_slider_labels()
                    count = len(new_indices)
                    msg = f"Pasted {count} object{'s' if count > 1 else ''}"
                    dpg.set_value("status_text", msg)
            self.ctrl_v_down_last = ctrl_v_down
