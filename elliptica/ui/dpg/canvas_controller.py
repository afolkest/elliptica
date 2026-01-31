"""Canvas interaction controller for Elliptica UI - mouse, keyboard, hit detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, List
import math
import time

from elliptica import defaults
from elliptica.app import actions
from elliptica.types import BoundaryObject, clone_boundary_object

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg  # type: ignore
except ImportError:
    dpg = None


MIN_HIT_TARGET = 10  # Base minimum clickable area in screen pixels


def _point_in_boundary(boundary: BoundaryObject, x: float, y: float, zoom: float = 1.0) -> bool:
    """Test whether canvas coordinate lands inside boundary hit area.

    For small boundaries, expands bounding box to MIN_HIT_TARGET for easier selection.
    For larger boundaries, uses pixel-perfect mask testing.
    The hit target is scaled by inverse zoom to maintain consistent screen-space click area.

    Args:
        boundary: The boundary object to test
        x, y: Canvas coordinates
        zoom: Current zoom level (used to scale hit target)
    """
    cx, cy = boundary.position
    mask = boundary.mask
    h, w = mask.shape

    # Scale hit target by inverse zoom to maintain consistent screen-space size
    effective_hit_target = MIN_HIT_TARGET / zoom if zoom > 0 else MIN_HIT_TARGET

    # Calculate expanded bounds for small boundaries
    expand_x = max(0, (effective_hit_target - w) / 2)
    expand_y = max(0, (effective_hit_target - h) / 2)

    # Check expanded bounding box first
    if x < cx - expand_x or x >= cx + w + expand_x:
        return False
    if y < cy - expand_y or y >= cy + h + expand_y:
        return False

    # For small boundaries, bounding box hit is enough
    if w <= effective_hit_target or h <= effective_hit_target:
        return True

    # For larger boundaries, do pixel-perfect test
    local_x = int(round(x - cx))
    local_y = int(round(y - cy))
    if local_x < 0 or local_y < 0:
        return False
    if local_y >= h or local_x >= w:
        return False
    return mask[local_y, local_x] > 0.5


class CanvasController:
    """Handles canvas mouse/keyboard input, hit detection, and boundary manipulation."""

    def __init__(self, app: "EllipticaApp"):
        self.app = app

        # Mouse/drag state
        self.drag_active: bool = False
        self.drag_last_pos: Tuple[float, float] = (0.0, 0.0)
        self.mouse_down_last: bool = False
        self.mouse_wheel_delta: float = 0.0

        # Drag throttling - accumulate small movements and commit periodically
        self._drag_accumulated_dx: float = 0.0
        self._drag_accumulated_dy: float = 0.0
        self._drag_last_commit_time: float = 0.0
        self._DRAG_THROTTLE_SEC: float = 0.016  # ~60 FPS max update rate

        # Scale gesture tracking (for contour skip during scroll-to-scale)
        self.scaling_active: bool = False
        self._last_scale_time: float = 0.0
        self._SCALE_DEBOUNCE_SEC: float = 0.15  # Time after last scroll to consider scaling "done"

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
        self.clipboard_boundaries: List[BoundaryObject] = []

        # Zoom/pan state
        self.zoom: float = 1.0          # 1.0 = 100%
        self.pan_x: float = 0.0         # Pan offset in canvas coords
        self.pan_y: float = 0.0
        self.panning: bool = False      # Currently in pan drag
        self.pan_start_mouse: Tuple[float, float] = (0.0, 0.0)  # Absolute screen coords
        self.pan_start_offset: Tuple[float, float] = (0.0, 0.0)

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
        """Convert screen mouse position to canvas coordinates (zoom/pan aware)."""
        assert dpg is not None and self.app.canvas_id is not None
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.app.canvas_id)

        # Screen coords relative to canvas widget
        screen_x = mouse_x - rect_min[0]
        screen_y = mouse_y - rect_min[1]

        # Inverse display_scale
        scale = self.app.display_scale if self.app.display_scale > 0 else 1.0
        screen_x /= scale
        screen_y /= scale

        # Inverse zoom and add pan offset
        zoom = self.zoom if self.zoom > 0 else 1.0
        canvas_x = screen_x / zoom + self.pan_x
        canvas_y = screen_y / zoom + self.pan_y

        return canvas_x, canvas_y

    def find_hit_boundary(self, x: float, y: float) -> int:
        """Find which boundary (if any) is at canvas coordinates. Returns index or -1."""
        with self.app.state_lock:
            boundaries = self.app.state.project.boundary_objects
            for idx in reversed(range(len(boundaries))):
                if _point_in_boundary(boundaries[idx], x, y, self.zoom):
                    return idx
        return -1

    def find_boundaries_in_box(self, x1: float, y1: float, x2: float, y2: float) -> set[int]:
        """Find all boundaries with actual mask pixels inside selection box."""
        # Normalize box coordinates
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        result = set()
        with self.app.state_lock:
            boundaries = self.app.state.project.boundary_objects
            for idx, boundary in enumerate(boundaries):
                cx, cy = boundary.position
                mask = boundary.mask
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
        """Detect which boundary region is at canvas point.

        Returns (boundary_idx, region) where region is "surface" or "interior" or None.
        """
        cache = self.app.state.render_cache
        if cache is None or cache.boundary_masks is None or cache.interior_masks is None:
            return -1, None

        # Masks are at FULL RENDER RESOLUTION (canvas × multiplier × supersample)
        # Scale canvas coordinates to match mask resolution
        scale = cache.multiplier * cache.supersample
        mask_x = int(canvas_x * scale)
        mask_y = int(canvas_y * scale)

        # Check each boundary in reverse order (top to bottom)
        for idx in reversed(range(len(self.app.state.project.boundary_objects))):
            if idx >= len(cache.interior_masks) or idx >= len(cache.boundary_masks):
                continue

            # Check interior first (it's inside, so higher priority)
            interior_mask = cache.interior_masks[idx]
            if interior_mask is not None:
                h, w = interior_mask.shape
                if 0 <= mask_y < h and 0 <= mask_x < w:
                    if interior_mask[mask_y, mask_x] > 0.5:
                        return idx, "interior"

            # Check surface
            surface_mask = cache.boundary_masks[idx]
            if surface_mask is not None:
                h, w = surface_mask.shape
                if 0 <= mask_y < h and 0 <= mask_x < w:
                    if surface_mask[mask_y, mask_x] > 0.5:
                        return idx, "surface"

        return -1, None

    def scale_selected_boundaries(self, factor: float) -> bool:
        """Scale all selected boundaries by factor. Returns True if any scaled."""
        if dpg is None:
            return False
        factor = max(float(factor), 0.05)
        any_changed = False
        with self.app.state_lock:
            indices = list(self.app.state.selected_indices)
            for idx in indices:
                changed = actions.scale_boundary(self.app.state, idx, factor)
                if changed:
                    self.app.display_pipeline.texture_manager.clear_boundary_texture(idx)
                    self.app.canvas_renderer.invalidate_boundary_contour_cache(idx)
                    any_changed = True

        if not any_changed:
            dpg.set_value("status_text", "Scaling limit reached.")
            return False

        self.app.canvas_renderer.mark_dirty()
        self.app.boundary_controls.update_slider_labels()
        self.app.postprocess_panel.update_region_properties_panel()
        count = len(indices)
        if count == 1:
            dpg.set_value("status_text", f"Scaled B{indices[0] + 1} by {factor:.2f}×")
        else:
            dpg.set_value("status_text", f"Scaled {count} objects by {factor:.2f}×")
        return True

    def _clamp_pan(self) -> None:
        """Zoom-aware pan clamping to allow viewing any part of the canvas.

        At zoom=1, allows panning with ~10% margin beyond edges.
        At higher zoom, expands limits so user can view canvas edges and corners.
        """
        with self.app.state_lock:
            canvas_w, canvas_h = self.app.state.project.canvas_resolution

        # Get current scale factors
        display_scale = self.app.display_scale if self.app.display_scale > 0 else 1.0
        zoom = self.zoom if self.zoom > 0 else 1.0
        total_scale = display_scale * zoom

        # Get window size (drawlist size) to calculate visible area
        window_w, window_h = canvas_w, canvas_h  # Fallback
        if self.app.canvas_id is not None and dpg is not None:
            rect = dpg.get_item_rect_size(self.app.canvas_id)
            if rect and rect[0] > 0 and rect[1] > 0:
                window_w, window_h = rect

        # Calculate visible area in canvas coordinates
        visible_w = window_w / total_scale
        visible_h = window_h / total_scale

        # Margin: allow 10% of canvas beyond edges for breathing room
        margin_x = canvas_w * 0.1
        margin_y = canvas_h * 0.1

        # Pan limits: allow seeing entire canvas edge-to-edge plus margin.
        # When pan_x = min_pan_x, left edge of canvas is near screen center.
        # When pan_x = max_pan_x, right edge of canvas is near screen center.
        min_pan_x = -visible_w / 2 - margin_x
        max_pan_x = canvas_w - visible_w / 2 + margin_x
        min_pan_y = -visible_h / 2 - margin_y
        max_pan_y = canvas_h - visible_h / 2 + margin_y

        # Handle case where visible area > canvas + margin (zoomed out far)
        # In this case, center the canvas
        if min_pan_x > max_pan_x:
            min_pan_x = max_pan_x = (canvas_w - visible_w) / 2
        if min_pan_y > max_pan_y:
            min_pan_y = max_pan_y = (canvas_h - visible_h) / 2

        self.pan_x = max(min_pan_x, min(self.pan_x, max_pan_x))
        self.pan_y = max(min_pan_y, min(self.pan_y, max_pan_y))

    def _zoom_toward_cursor(self, wheel_delta: float) -> None:
        """Zoom canvas toward/away from cursor position."""
        if dpg is None or self.app.canvas_id is None:
            return

        old_zoom = self.zoom

        # Exponential zoom for smooth feel (use same sensitivity as boundary scaling)
        zoom_factor = math.exp(wheel_delta * defaults.SCROLL_SCALE_SENSITIVITY)
        new_zoom = old_zoom * zoom_factor

        # Clamp zoom to reasonable bounds (10% to 1000%)
        new_zoom = max(0.1, min(new_zoom, 10.0))

        if abs(new_zoom - old_zoom) < 1e-6:
            return

        # Compute screen coords (before zoom change)
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.app.canvas_id)
        display_scale = self.app.display_scale if self.app.display_scale > 0 else 1.0
        screen_x = (mouse_x - rect_min[0]) / display_scale
        screen_y = (mouse_y - rect_min[1]) / display_scale

        # Canvas point under cursor with OLD zoom
        canvas_x = screen_x / old_zoom + self.pan_x
        canvas_y = screen_y / old_zoom + self.pan_y

        # Update zoom
        self.zoom = new_zoom

        # Adjust pan so same canvas point stays under cursor
        self.pan_x = canvas_x - screen_x / new_zoom
        self.pan_y = canvas_y - screen_y / new_zoom

        # Apply zoom-aware pan bounds (10% margin beyond canvas edges)
        self._clamp_pan()

        self.app._update_canvas_transform()
        self.app.canvas_renderer.mark_dirty()

    def reset_zoom_pan(self) -> None:
        """Reset zoom and pan to default values."""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.panning = False
        self.app._update_canvas_transform()
        self.app.canvas_renderer.mark_dirty()

    def is_file_dialog_showing(self) -> bool:
        """Check if any file dialog is currently showing."""
        if dpg is None:
            return False

        file_io = self.app.file_io
        dialogs = [
            file_io.load_project_dialog_id,
            file_io.save_project_dialog_id,
            file_io.boundary_file_dialog_id,
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

    def is_menu_open(self) -> bool:
        """Check if any dropdown menu or popup is currently open."""
        if dpg is None:
            return False

        # Menus: check active/hovered state
        menu_tags = ["file_menu", "export_menu"]
        for tag in menu_tags:
            if dpg.does_item_exist(tag):
                active = dpg.is_item_active(tag)
                hovered = dpg.is_item_hovered(tag)
                if active or hovered:
                    return True

        # Popups: check if actually visible (not just show=True, which can get stuck)
        popup_tags = ["region_palette_popup", "global_palette_popup"]
        for tag in popup_tags:
            if dpg.does_item_exist(tag) and dpg.is_item_visible(tag):
                return True

        return False

    def process_canvas_mouse(self) -> None:
        """Process mouse input on canvas (clicks, drags, wheel, pan)."""
        if dpg is None or self.app.canvas_id is None:
            return

        from elliptica.ui.dpg.app import SHIFT_KEY, CTRL_KEY, SPACE_KEY

        mouse_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)

        # Don't process canvas input when file dialogs, modals, or menus are open
        # But still track mouse state to prevent desync
        file_dialog = self.is_file_dialog_showing()
        modal_open = self.is_modal_open()
        menu_open = self.is_menu_open()
        if file_dialog or modal_open or menu_open:
            self.mouse_down_last = mouse_down
            self.panning = False
            return
        pressed = mouse_down and not self.mouse_down_last
        released = (not mouse_down) and self.mouse_down_last

        # Check if scaling gesture ended (debounce timeout)
        if self.scaling_active:
            if (time.monotonic() - self._last_scale_time) > self._SCALE_DEBOUNCE_SEC:
                self.scaling_active = False
                self.app.canvas_renderer.mark_dirty()  # Redraw with contour

        # Track modifier keys
        self.shift_down = SHIFT_KEY is not None and dpg.is_key_down(SHIFT_KEY)
        ctrl_down = CTRL_KEY is not None and dpg.is_key_down(CTRL_KEY)
        space_down = SPACE_KEY is not None and dpg.is_key_down(SPACE_KEY)

        with self.app.state_lock:
            mode = self.app.state.view_mode

        wheel_delta = self.mouse_wheel_delta
        self.mouse_wheel_delta = 0.0
        over_canvas = self.is_mouse_over_canvas()

        # End pan when mouse released
        if not mouse_down:
            self.panning = False

        # Pan mode: space + left-drag (check before other interactions)
        # Don't allow pan if box selecting or dragging boundaries
        if space_down and over_canvas and not self.box_select_active and not self.drag_active:
            if pressed and not self.panning:
                # Start panning - store absolute screen position
                self.panning = True
                mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
                self.pan_start_mouse = (mouse_x, mouse_y)
                self.pan_start_offset = (self.pan_x, self.pan_y)

            elif self.panning and mouse_down:
                # Continue panning
                mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

                # Delta in screen space, convert to canvas space
                display_scale = self.app.display_scale if self.app.display_scale > 0 else 1.0
                zoom = self.zoom if self.zoom > 0 else 1.0
                dx = (mouse_x - self.pan_start_mouse[0]) / display_scale / zoom
                dy = (mouse_y - self.pan_start_mouse[1]) / display_scale / zoom

                self.pan_x = self.pan_start_offset[0] - dx
                self.pan_y = self.pan_start_offset[1] - dy

                self._clamp_pan()
                self.app._update_canvas_transform()
                self.app.canvas_renderer.mark_dirty()

        # Skip normal click handling when panning
        if self.panning:
            self.mouse_down_last = mouse_down
            return

        # Mouse wheel handling: zoom (default) or Ctrl+scroll for boundary scaling
        if over_canvas and abs(wheel_delta) > 1e-5 and not self.box_select_active:
            if ctrl_down and mode == "edit":
                # Ctrl+scroll in edit mode: scale selected boundaries
                with self.app.state_lock:
                    has_selection = len(self.app.state.selected_indices) > 0
                if has_selection:
                    scale_factor = math.exp(wheel_delta * defaults.SCROLL_SCALE_SENSITIVITY)
                    scale_factor = max(0.05, min(scale_factor, 20.0))
                    if self.scale_selected_boundaries(scale_factor):
                        x, y = self.get_canvas_mouse_pos()
                        self.drag_last_pos = (x, y)
                        # Track scaling gesture for contour skip
                        self.scaling_active = True
                        self._last_scale_time = time.monotonic()
                        # Reset accumulators to prevent position jump after scaling during drag
                        if self.drag_active:
                            self._drag_accumulated_dx = 0.0
                            self._drag_accumulated_dy = 0.0
            else:
                # Plain scroll: zoom canvas toward cursor (works in both modes)
                self._zoom_toward_cursor(wheel_delta)

        # Render mode: click to select region for colorization (single-select only)
        if mode != "edit":
            over_canvas = self.is_mouse_over_canvas()
            if pressed and over_canvas:
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
        over_canvas_edit = self.is_mouse_over_canvas()
        if pressed and over_canvas_edit:
            x, y = self.get_canvas_mouse_pos()
            hit_idx = self.find_hit_boundary(x, y)

            with self.app.state_lock:
                if hit_idx >= 0:
                    # Clicked on a boundary
                    if self.shift_down:
                        # Shift-click: toggle selection
                        self.app.state.toggle_selected(hit_idx)
                    else:
                        # Normal click: if clicking on already-selected, keep selection for drag
                        # Otherwise, replace selection with clicked boundary
                        if hit_idx not in self.app.state.selected_indices:
                            self.app.state.set_selected(hit_idx)
                    self.app.state.selected_region_type = "surface"
                    self.drag_active = True
                    self.drag_last_pos = (x, y)
                    # Reset drag accumulators to prevent stale values from previous drags
                    self._drag_accumulated_dx = 0.0
                    self._drag_accumulated_dy = 0.0
                    self._drag_last_commit_time = time.monotonic()
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
                # Drag all selected boundaries with throttling
                dx = x - self.drag_last_pos[0]
                dy = y - self.drag_last_pos[1]

                if abs(dx) > 0.1 or abs(dy) > 0.1:
                    # Accumulate movement
                    self._drag_accumulated_dx += dx
                    self._drag_accumulated_dy += dy
                    self.drag_last_pos = (x, y)

                    # Check if we should commit the accumulated movement
                    now = time.monotonic()
                    time_since_commit = now - self._drag_last_commit_time
                    accumulated_dist = (self._drag_accumulated_dx**2 + self._drag_accumulated_dy**2)**0.5

                    # Commit if: enough time passed OR significant movement accumulated
                    if time_since_commit >= self._DRAG_THROTTLE_SEC or accumulated_dist > 3.0:
                        with self.app.state_lock:
                            for idx in self.app.state.selected_indices:
                                if 0 <= idx < len(self.app.state.project.boundary_objects):
                                    actions.move_boundary(
                                        self.app.state, idx,
                                        self._drag_accumulated_dx,
                                        self._drag_accumulated_dy
                                    )

                        # Reset accumulators
                        self._drag_accumulated_dx = 0.0
                        self._drag_accumulated_dy = 0.0
                        self._drag_last_commit_time = now

                        self.app.canvas_renderer.mark_dirty()

        # Handle release
        if released:
            if self.box_select_active:
                # Finish box selection
                x, y = self.get_canvas_mouse_pos()
                self.box_select_end = (x, y)
                hits = self.find_boundaries_in_box(
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

            elif self.drag_active:
                # Flush any remaining accumulated drag movement
                if self._drag_accumulated_dx != 0 or self._drag_accumulated_dy != 0:
                    with self.app.state_lock:
                        for idx in self.app.state.selected_indices:
                            if 0 <= idx < len(self.app.state.project.boundary_objects):
                                actions.move_boundary(
                                    self.app.state, idx,
                                    self._drag_accumulated_dx,
                                    self._drag_accumulated_dy
                                )
                    self._drag_accumulated_dx = 0.0
                    self._drag_accumulated_dy = 0.0
                # Always redraw on drag end to restore contour outline
                self.app.canvas_renderer.mark_dirty()

            self.drag_active = False

        self.mouse_down_last = mouse_down

    def process_keyboard_shortcuts(self) -> None:
        """Process keyboard shortcuts (delete, copy, paste, zoom reset)."""
        from elliptica.ui.dpg.app import BACKSPACE_KEY, CTRL_KEY, C_KEY, V_KEY, HOME_KEY, ZERO_KEY

        if dpg is None or BACKSPACE_KEY is None:
            return

        # Don't process keyboard shortcuts when file dialogs, modals, or menus are open
        if self.is_file_dialog_showing() or self.is_modal_open() or self.is_menu_open():
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
                        actions.remove_boundary(self.app.state, idx)
                    self.app.display_pipeline.texture_manager.clear_all_boundary_textures()
                # Invalidate contour cache (indices shift after deletion)
                self.app.canvas_renderer.invalidate_boundary_contour_cache()
                self.app.canvas_renderer.mark_dirty()
                self.app.boundary_controls.rebuild_controls()
                count = len(indices_to_delete)
                msg = "Boundary deleted" if count == 1 else f"{count} boundaries deleted"
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
                        self.clipboard_boundaries = [
                            clone_boundary_object(self.app.state.project.boundary_objects[idx], preserve_id=False)
                            for idx in sorted(selected)
                        ]
                    count = len(self.clipboard_boundaries)
                    msg = f"Copied B{selected[0] + 1}" if count == 1 else f"Copied {count} objects"
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
                    can_paste = mode == "edit" and len(self.clipboard_boundaries) > 0
                if can_paste:
                    with self.app.state_lock:
                        new_indices = set()
                        for boundary in self.clipboard_boundaries:
                            pasted = clone_boundary_object(boundary, preserve_id=False)
                            # Offset by 30px down-right
                            px, py = pasted.position
                            pasted.position = (px + 30.0, py + 30.0)
                            actions.add_boundary(self.app.state, pasted)
                            new_indices.add(len(self.app.state.project.boundary_objects) - 1)
                        # Select all pasted boundaries
                        self.app.state.selected_indices = new_indices
                    self.app.canvas_renderer.mark_dirty()
                    self.app.boundary_controls.rebuild_controls()
                    self.app.boundary_controls.update_slider_labels()
                    count = len(new_indices)
                    msg = f"Pasted {count} object{'s' if count > 1 else ''}"
                    dpg.set_value("status_text", msg)
            self.ctrl_v_down_last = ctrl_v_down

        # Reset zoom/pan with Home or Ctrl+0
        ctrl_down = CTRL_KEY is not None and dpg.is_key_down(CTRL_KEY)
        home_pressed = HOME_KEY is not None and dpg.is_key_pressed(HOME_KEY)
        zero_pressed = ZERO_KEY is not None and dpg.is_key_pressed(ZERO_KEY)

        if home_pressed or (ctrl_down and zero_pressed):
            self.reset_zoom_pan()
            dpg.set_value("status_text", "Zoom reset to 100%")
