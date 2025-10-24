"""Dear PyGui application for FlowCol."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor, Future
import math
import threading
from typing import Optional, Dict, Tuple

import numpy as np

try:
    import dearpygui.dearpygui as dpg  # type: ignore
except ImportError:  # pragma: no cover - Dear PyGui is optional for unit tests
    dpg = None

from flowcol.app.core import AppState, RenderCache, RenderSettings
from pathlib import Path

from flowcol.app import actions
from flowcol.render import array_to_pil
from flowcol.types import Conductor, Project
from flowcol.pipeline import perform_render
from flowcol import defaults
from flowcol.mask_utils import load_conductor_masks


CONDUCTOR_COLORS = [
    (0.39, 0.59, 1.0, 0.7),
    (1.0, 0.39, 0.59, 0.7),
    (0.59, 1.0, 0.39, 0.7),
    (1.0, 0.78, 0.39, 0.7),
]

MAX_PREVIEW_SIZE = 640
MAX_CANVAS_DIM = 8192

SUPERSAMPLE_CHOICES = defaults.SUPERSAMPLE_CHOICES
SUPERSAMPLE_LABELS = tuple(f"{value:.1f}\u00d7" for value in SUPERSAMPLE_CHOICES)
SUPERSAMPLE_LOOKUP = {label: value for label, value in zip(SUPERSAMPLE_LABELS, SUPERSAMPLE_CHOICES)}

RESOLUTION_CHOICES = defaults.RENDER_RESOLUTION_CHOICES
RESOLUTION_LABELS = tuple(f"{value:g}\u00d7" for value in RESOLUTION_CHOICES)
RESOLUTION_LOOKUP = {label: value for label, value in zip(RESOLUTION_LABELS, RESOLUTION_CHOICES)}

BACKSPACE_KEY = None
if dpg is not None:
    BACKSPACE_KEY = getattr(dpg, "mvKey_Backspace", None)
    if BACKSPACE_KEY is None:
        BACKSPACE_KEY = getattr(dpg, "mvKey_Back", None)


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


def _mask_to_rgba(mask: np.ndarray, color: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert mask to RGBA float texture."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = color[:3]
    rgba[..., 3] = mask.astype(np.float32) * color[3]
    return rgba.reshape(-1)


def _image_to_texture_data(img) -> Tuple[int, int, np.ndarray]:
    """Convert PIL image to float RGBA data."""
    img = img.convert("RGBA")
    width, height = img.size
    rgba = np.asarray(img, dtype=np.float32) / 255.0
    return width, height, rgba.reshape(-1)


def _label_for_supersample(value: float) -> str:
    idx = min(range(len(SUPERSAMPLE_CHOICES)), key=lambda i: abs(SUPERSAMPLE_CHOICES[i] - value))
    return SUPERSAMPLE_LABELS[idx]


def _label_for_multiplier(value: float) -> str:
    idx = min(range(len(RESOLUTION_CHOICES)), key=lambda i: abs(RESOLUTION_CHOICES[i] - value))
    return RESOLUTION_LABELS[idx]


def _clone_conductor(conductor: Conductor) -> Conductor:
    """Deep-copy conductor data for background rendering."""
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
    )


def _snapshot_project(project: Project) -> Project:
    """Create a snapshot of project safe to use off the UI thread."""
    return Project(
        conductors=[_clone_conductor(c) for c in project.conductors],
        canvas_resolution=project.canvas_resolution,
        streamlength_factor=project.streamlength_factor,
        renders=list(project.renders),
    )


@dataclass
class FlowColApp:
    """Coordinator for Dear PyGui widgets and background work."""

    state: AppState = field(default_factory=AppState)
    state_lock: threading.RLock = field(default_factory=threading.RLock)
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))

    render_future: Optional[Future] = None
    render_error: Optional[str] = None

    canvas_id: Optional[int] = None
    texture_registry_id: Optional[int] = None
    render_texture_id: Optional[int] = None
    render_texture_size: Optional[Tuple[int, int]] = None
    viewport_created: bool = False
    edit_controls_id: Optional[int] = None
    render_controls_id: Optional[int] = None
    render_modal_id: Optional[int] = None
    render_supersample_radio_id: Optional[int] = None
    render_multiplier_radio_id: Optional[int] = None
    render_passes_input_id: Optional[int] = None
    render_streamlength_input_id: Optional[int] = None
    render_margin_input_id: Optional[int] = None
    render_seed_input_id: Optional[int] = None
    render_sigma_input_id: Optional[int] = None
    canvas_width_input_id: Optional[int] = None
    canvas_height_input_id: Optional[int] = None
    conductor_file_dialog_id: Optional[str] = None
    conductor_controls_container_id: Optional[int] = None
    conductor_slider_ids: Dict[int, int] = field(default_factory=dict)
    postprocess_downsample_slider_id: Optional[int] = None
    postprocess_clip_slider_id: Optional[int] = None
    postprocess_contrast_slider_id: Optional[int] = None
    postprocess_gamma_slider_id: Optional[int] = None

    conductor_textures: Dict[int, int] = field(default_factory=dict)
    conductor_texture_shapes: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    canvas_dirty: bool = True

    drag_active: bool = False
    drag_last_pos: Tuple[float, float] = (0.0, 0.0)
    mouse_down_last: bool = False
    render_modal_open: bool = False
    backspace_down_last: bool = False
    mouse_wheel_delta: float = 0.0
    mouse_handler_registry_id: Optional[int] = None

    def __post_init__(self) -> None:
        # Seed a demo conductor if project is empty so the canvas has content for manual testing.
        if not self.state.project.conductors:
            self._add_demo_conductor()

    def _on_mouse_wheel(self, sender, app_data) -> None:
        """Capture mouse wheel delta from handler."""
        self.mouse_wheel_delta = float(app_data)

    def require_backend(self) -> None:
        if dpg is None:
            raise RuntimeError("Dear PyGui is not installed. Please `pip install dearpygui` to run the GUI.")

    # ------------------------------------------------------------------
    # Building the interface
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Create viewport, windows, and widgets."""
        self.require_backend()
        dpg.create_context()
        self.texture_registry_id = dpg.add_texture_registry()

        dpg.create_viewport(title="FlowCol", width=1280, height=820)
        self.viewport_created = True

        with dpg.handler_registry() as handler_reg:
            self.mouse_handler_registry_id = handler_reg
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)

        with dpg.window(label="Controls", width=360, height=-1, pos=(10, 10), tag="controls_window"):
            with dpg.group(tag="edit_controls_group") as edit_group:
                self.edit_controls_id = edit_group
                dpg.add_text("Render Controls")
                dpg.add_button(label="Load Conductor...", callback=self._open_conductor_dialog)
                dpg.add_button(label="Render Field", callback=self._open_render_modal)
                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_text("Canvas Size")
                with dpg.group(horizontal=True):
                    self.canvas_width_input_id = dpg.add_input_int(
                        label="Width",
                        min_value=1,
                        min_clamped=True,
                        max_value=32768,
                        max_clamped=True,
                        step=0,
                        width=120,
                    )
                    self.canvas_height_input_id = dpg.add_input_int(
                        label="Height",
                        min_value=1,
                        min_clamped=True,
                        max_value=32768,
                        max_clamped=True,
                        step=0,
                        width=120,
                    )
                dpg.add_button(label="Apply Canvas Size", callback=self._apply_canvas_size)
                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_text("Conductor Voltages")
                self.conductor_controls_container_id = dpg.add_child_window(
                    autosize_x=True,
                    height=240,
                    border=False,
                    tag="conductor_controls_child",
                )

            with dpg.group(tag="render_controls_group") as render_group:
                self.render_controls_id = render_group
                dpg.add_text("Render View")
                dpg.add_button(label="Back to Edit", callback=self._on_back_to_edit_clicked)
                dpg.add_spacer(height=15)
                dpg.add_separator()
                dpg.add_spacer(height=10)
                dpg.add_text("Post-processing")
                dpg.add_spacer(height=10)

                self.postprocess_downsample_slider_id = dpg.add_slider_float(
                    label="Downsampling Blur",
                    default_value=self.state.postprocess_settings.downsample_sigma,
                    min_value=0.1,
                    max_value=2.0,
                    format="%.2f",
                    callback=self._on_downsample_slider,
                    width=200,
                )

                self.postprocess_clip_slider_id = dpg.add_slider_float(
                    label="Clip %",
                    default_value=self.state.postprocess_settings.clip_percent,
                    min_value=0.0,
                    max_value=defaults.MAX_CLIP_PERCENT,
                    format="%.2f%%",
                    callback=self._on_clip_slider,
                    width=200,
                )

                self.postprocess_contrast_slider_id = dpg.add_slider_float(
                    label="Contrast",
                    default_value=self.state.postprocess_settings.contrast,
                    min_value=0.5,
                    max_value=2.0,
                    format="%.2f",
                    callback=self._on_contrast_slider,
                    width=200,
                )

                self.postprocess_gamma_slider_id = dpg.add_slider_float(
                    label="Gamma",
                    default_value=self.state.postprocess_settings.gamma,
                    min_value=0.3,
                    max_value=3.0,
                    format="%.2f",
                    callback=self._on_gamma_slider,
                    width=200,
                )
            dpg.add_spacer(height=10)
            dpg.add_text("Status:")
            dpg.add_text("", tag="status_text")

        with dpg.window(label="Canvas", pos=(380, 10), no_scrollbar=True, no_scroll_with_mouse=True):
            canvas_w, canvas_h = self.state.project.canvas_resolution
            with dpg.drawlist(width=canvas_w, height=canvas_h) as canvas:
                self.canvas_id = canvas

        self._refresh_render_texture()
        self._update_control_visibility()
        self._ensure_conductor_file_dialog()
        self._update_canvas_inputs()
        self._rebuild_conductor_controls()

    # ------------------------------------------------------------------
    # Canvas drawing
    # ------------------------------------------------------------------
    def _mark_canvas_dirty(self) -> None:
        self.canvas_dirty = True

    def _is_mouse_over_canvas(self) -> bool:
        """Check if mouse is within canvas bounds."""
        if dpg is None or self.canvas_id is None:
            return False

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.canvas_id)
        rect_max = dpg.get_item_rect_max(self.canvas_id)

        return (rect_min[0] <= mouse_x <= rect_max[0] and
                rect_min[1] <= mouse_y <= rect_max[1])

    def _get_canvas_mouse_pos(self) -> Tuple[float, float]:
        assert dpg is not None and self.canvas_id is not None
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.canvas_id)
        return mouse_x - rect_min[0], mouse_y - rect_min[1]

    def _find_hit_conductor(self, x: float, y: float) -> int:
        with self.state_lock:
            conductors = self.state.project.conductors
            for idx in reversed(range(len(conductors))):
                if _point_in_conductor(conductors[idx], x, y):
                    return idx
        return -1

    def _ensure_conductor_texture(self, idx: int, mask: np.ndarray) -> int:
        assert dpg is not None and self.texture_registry_id is not None
        tex_id = self.conductor_textures.get(idx)
        width = mask.shape[1]
        height = mask.shape[0]
        existing_shape = self.conductor_texture_shapes.get(idx)
        if tex_id is not None:
            exists = dpg.does_item_exist(tex_id)
            if not exists or existing_shape != (height, width):
                if exists:
                    dpg.delete_item(tex_id)
                tex_id = None
                self.conductor_textures.pop(idx, None)

        rgba_flat = _mask_to_rgba(mask, CONDUCTOR_COLORS[idx % len(CONDUCTOR_COLORS)])

        if tex_id is None:
            tex_id = dpg.add_dynamic_texture(width, height, rgba_flat, parent=self.texture_registry_id)
            self.conductor_textures[idx] = tex_id
        else:
            dpg.set_value(tex_id, rgba_flat)
        self.conductor_texture_shapes[idx] = (height, width)
        return tex_id

    def _ensure_render_texture(self, width: int, height: int) -> int:
        assert dpg is not None and self.texture_registry_id is not None
        if self.render_texture_id is None:
            empty = np.zeros((height, width, 4), dtype=np.float32)
            empty[..., 3] = 1.0
            self.render_texture_id = dpg.add_dynamic_texture(width, height, empty.reshape(-1), parent=self.texture_registry_id)
            self.render_texture_size = (width, height)
        return self.render_texture_id

    def _add_demo_conductor(self) -> None:
        """Populate state with a simple circular conductor for quick manual testing."""
        canvas_w, canvas_h = self.state.project.canvas_resolution
        size = min(canvas_w, canvas_h) // 4 or 128
        y, x = np.ogrid[:size, :size]
        cy = cx = size / 2.0
        radius = size / 2.2
        mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2
        mask = mask.astype(np.float32)
        conductor = Conductor(mask=mask, voltage=1.0, position=((canvas_w - size) / 2.0, (canvas_h - size) / 2.0))
        self.state.project.conductors.append(conductor)
        self.state.selected_idx = len(self.state.project.conductors) - 1
        self.state.field_dirty = True
        self.state.render_dirty = True

    def _apply_postprocessing(self) -> None:
        """Apply postprocessing settings to cached render and update display."""
        from flowcol.render import downsample_lic

        with self.state_lock:
            cache = self.state.render_cache
            if cache is None:
                return

            settings = self.state.postprocess_settings
            result = cache.result

            # Apply downsampling with blur
            target_shape = self.state.project.canvas_resolution
            downsampled = downsample_lic(
                result.array,
                target_shape,
                cache.supersample,
                settings.downsample_sigma,
            )

            # Store postprocessed result in display_array
            cache.display_array = downsampled

        # Update texture with new postprocessed display
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _refresh_render_texture(self) -> None:
        if dpg is None or self.texture_registry_id is None:
            return

        with self.state_lock:
            cache = self.state.render_cache
            settings = self.state.postprocess_settings

            if cache is None:
                arr = np.zeros((32, 32), dtype=np.float32)
                clip = 0.0
                contrast = 1.0
                gamma = 1.0
            else:
                arr = cache.display_array if cache.display_array is not None else cache.result.array
                clip = settings.clip_percent
                contrast = settings.contrast
                gamma = settings.gamma

            pil_img = array_to_pil(
                arr,
                use_color=False,
                clip_percent=clip,
                contrast=contrast,
                gamma=gamma,
            )
            if max(pil_img.size) > MAX_PREVIEW_SIZE:
                pil_img.thumbnail((MAX_PREVIEW_SIZE, MAX_PREVIEW_SIZE))

        width, height, data = _image_to_texture_data(pil_img)

        if self.render_texture_id is None or self.render_texture_size != (width, height):
            if self.render_texture_id is not None:
                dpg.delete_item(self.render_texture_id)
            self.render_texture_id = dpg.add_dynamic_texture(width, height, data, parent=self.texture_registry_id)
            self.render_texture_size = (width, height)
        else:
            dpg.set_value(self.render_texture_id, data)

    def _update_canvas_inputs(self) -> None:
        if dpg is None:
            return
        with self.state_lock:
            width, height = self.state.project.canvas_resolution
        if self.canvas_width_input_id is not None:
            dpg.set_value(self.canvas_width_input_id, int(width))
        if self.canvas_height_input_id is not None:
            dpg.set_value(self.canvas_height_input_id, int(height))

    def _rebuild_conductor_controls(self) -> None:
        if dpg is None or self.conductor_controls_container_id is None:
            return

        dpg.delete_item(self.conductor_controls_container_id, children_only=True)
        self.conductor_slider_ids.clear()

        with self.state_lock:
            conductors = list(self.state.project.conductors)
            selected_idx = self.state.selected_idx

        if not conductors:
            dpg.add_text("No conductors loaded.", parent=self.conductor_controls_container_id)
            return

        for idx, conductor in enumerate(conductors):
            label = f"C{idx + 1}"
            if idx == selected_idx:
                label += " (selected)"
            slider_id = dpg.add_slider_float(
                label=label,
                default_value=float(conductor.voltage),
                min_value=-1.0,
                max_value=1.0,
                format="%.3f",
                callback=self._on_conductor_voltage_slider,
                user_data=idx,
                parent=self.conductor_controls_container_id,
            )
            self.conductor_slider_ids[idx] = slider_id

    def _update_conductor_slider_labels(self, skip_idx: Optional[int] = None) -> None:
        if dpg is None or not self.conductor_slider_ids:
            return

        with self.state_lock:
            conductors = list(self.state.project.conductors)
            selected_idx = self.state.selected_idx

        for idx, slider_id in list(self.conductor_slider_ids.items()):
            if slider_id is None or not dpg.does_item_exist(slider_id):
                continue
            if idx >= len(conductors):
                continue
            label = f"C{idx + 1}"
            if idx == selected_idx:
                label += " (selected)"
            dpg.configure_item(slider_id, label=label)
            if skip_idx is not None and idx == skip_idx:
                continue
            dpg.set_value(slider_id, float(conductors[idx].voltage))

    def _on_conductor_voltage_slider(self, sender, app_data, user_data):
        if dpg is None:
            return
        idx = int(user_data)
        value = float(app_data)
        with self.state_lock:
            actions.set_conductor_voltage(self.state, idx, value)
        self._mark_canvas_dirty()
        dpg.set_value("status_text", f"C{idx + 1} voltage = {value:.3f}")
        self._update_conductor_slider_labels(skip_idx=idx)

    def _scale_conductor(self, idx: int, factor: float) -> bool:
        if dpg is None:
            return False
        factor = max(float(factor), 0.05)
        with self.state_lock:
            self.state.set_selected(idx)
            changed = actions.scale_conductor(self.state, idx, factor)
            if changed:
                self.conductor_textures.pop(idx, None)
                self.conductor_texture_shapes.pop(idx, None)
        if not changed:
            dpg.set_value("status_text", "Scaling limit reached.")
            return False

        self._mark_canvas_dirty()
        self._update_conductor_slider_labels()
        dpg.set_value("status_text", f"Scaled C{idx + 1} by {factor:.2f}×")
        return True

    def _ensure_render_modal(self) -> None:
        if dpg is None or self.render_modal_id is not None:
            return

        with dpg.window(
            label="Render Settings",
            modal=True,
            show=False,
            tag="render_modal",
            no_move=False,
            no_close=True,
            no_collapse=True,
            width=420,
            height=520,
        ) as modal:
            self.render_modal_id = modal

            dpg.add_text("Supersample Factor")
            self.render_supersample_radio_id = dpg.add_radio_button(
                SUPERSAMPLE_LABELS,
                horizontal=True,
            )
            dpg.add_spacer(height=10)

            dpg.add_text("Render Resolution")
            self.render_multiplier_radio_id = dpg.add_radio_button(
                RESOLUTION_LABELS,
                horizontal=True,
            )
            dpg.add_spacer(height=12)

            dpg.add_separator()
            dpg.add_spacer(height=12)

            self.render_passes_input_id = dpg.add_input_int(
                label="LIC Passes",
                min_value=1,
                step=1,
                min_clamped=True,
                width=160,
            )

            self.render_streamlength_input_id = dpg.add_input_float(
                label="Streamlength Factor",
                format="%.4f",
                min_value=1e-6,
                min_clamped=True,
                step=0.0,
                width=200,
            )

            self.render_margin_input_id = dpg.add_input_float(
                label="Padding Margin",
                format="%.3f",
                min_value=0.0,
                min_clamped=True,
                step=0.0,
                width=200,
            )

            self.render_seed_input_id = dpg.add_input_int(
                label="Noise Seed",
                step=1,
                min_clamped=False,
                width=160,
            )

            self.render_sigma_input_id = dpg.add_input_float(
                label="Noise Low-pass Sigma",
                format="%.2f",
                min_value=0.0,
                min_clamped=True,
                step=0.0,
                width=200,
            )

            dpg.add_spacer(height=20)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Render", width=140, callback=self._apply_render_modal)
                dpg.add_button(label="Cancel", width=140, callback=self._cancel_render_modal)

    def _ensure_conductor_file_dialog(self) -> None:
        if dpg is None or self.conductor_file_dialog_id is not None:
            return

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            default_path=str(Path.cwd()),
            callback=self._on_conductor_file_selected,
            cancel_callback=self._on_conductor_file_cancelled,
            width=640,
            height=420,
            tag="conductor_file_dialog",
        ) as dialog:
            self.conductor_file_dialog_id = dialog
            dpg.add_file_extension(".png", color=(150, 180, 255, 255))
            dpg.add_file_extension(".*")

    def _open_render_modal(self, sender, app_data):
        if dpg is None:
            return
        if self.render_future is not None and not self.render_future.done():
            dpg.set_value("status_text", "Render already in progress...")
            return
        self._ensure_render_modal()
        self._update_render_modal_values()
        if self.render_modal_id is not None:
            dpg.configure_item(self.render_modal_id, show=True)
            self.render_modal_open = True

    def _close_render_modal(self) -> None:
        if dpg is None or self.render_modal_id is None:
            return
        dpg.configure_item(self.render_modal_id, show=False)
        self.render_modal_open = False

    def _open_conductor_dialog(self, sender, app_data):
        if dpg is None:
            return
        with self.state_lock:
            if self.state.view_mode != "edit":
                dpg.set_value("status_text", "Switch to edit mode to add conductors.")
                return

        self._ensure_conductor_file_dialog()
        if self.conductor_file_dialog_id is not None:
            dpg.show_item(self.conductor_file_dialog_id)

    def _on_conductor_file_cancelled(self, sender, app_data):
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)
        dpg.set_value("status_text", "Load conductor cancelled.")

    def _on_conductor_file_selected(self, sender, app_data):
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)

        path_str: Optional[str] = None
        if isinstance(app_data, dict):
            # Preferred key exposed by Dear PyGui >=1.11
            path_str = app_data.get("file_path_name")
            if not path_str:
                selections = app_data.get("selections", {})
                if selections:
                    # selections is dict[label] -> path
                    path_str = next(iter(selections.values()))

        if not path_str:
            dpg.set_value("status_text", "No file selected.")
            return

        try:
            mask, interior = load_conductor_masks(path_str)
        except Exception as exc:  # pragma: no cover - PIL errors etc.
            dpg.set_value("status_text", f"Failed to load conductor: {exc}")
            return

        mask_h, mask_w = mask.shape
        if mask_w > MAX_CANVAS_DIM or mask_h > MAX_CANVAS_DIM:
            dpg.set_value("status_text", f"Mask exceeds max dimension {MAX_CANVAS_DIM}px.")
            return

        with self.state_lock:
            project = self.state.project
            if len(project.conductors) == 0:
                canvas_w, canvas_h = project.canvas_resolution
                new_w = max(canvas_w, mask_w)
                new_h = max(canvas_h, mask_h)
                if (new_w, new_h) != project.canvas_resolution:
                    actions.set_canvas_resolution(self.state, new_w, new_h)

            canvas_w, canvas_h = self.state.project.canvas_resolution
            pos = ((canvas_w - mask_w) / 2.0, (canvas_h - mask_h) / 2.0)
            conductor = Conductor(mask=mask, voltage=0.5, position=pos, interior_mask=interior)
            actions.add_conductor(self.state, conductor)
            self.state.view_mode = "edit"

        self._mark_canvas_dirty()
        self._update_control_visibility()
        self._rebuild_conductor_controls()
        self._update_conductor_slider_labels()
        dpg.set_value("status_text", f"Loaded conductor '{Path(path_str).name}'")


    def _update_render_modal_values(self) -> None:
        if dpg is None:
            return
        with self.state_lock:
            settings = replace(self.state.render_settings)
            streamlength = self.state.project.streamlength_factor

        if self.render_supersample_radio_id is not None:
            dpg.set_value(self.render_supersample_radio_id, _label_for_supersample(settings.supersample))

        if self.render_multiplier_radio_id is not None:
            dpg.set_value(self.render_multiplier_radio_id, _label_for_multiplier(settings.multiplier))

        if self.render_passes_input_id is not None:
            dpg.set_value(self.render_passes_input_id, int(settings.num_passes))

        if self.render_streamlength_input_id is not None:
            dpg.set_value(self.render_streamlength_input_id, float(streamlength))

        if self.render_margin_input_id is not None:
            dpg.set_value(self.render_margin_input_id, float(settings.margin))

        if self.render_seed_input_id is not None:
            dpg.set_value(self.render_seed_input_id, int(settings.noise_seed))

        if self.render_sigma_input_id is not None:
            dpg.set_value(self.render_sigma_input_id, float(settings.noise_sigma))

    def _cancel_render_modal(self, sender=None, app_data=None) -> None:
        self._close_render_modal()

    def _apply_render_modal(self, sender=None, app_data=None) -> None:
        if dpg is None:
            return

        supersample_label = dpg.get_value(self.render_supersample_radio_id) if self.render_supersample_radio_id else SUPERSAMPLE_LABELS[0]
        multiplier_label = dpg.get_value(self.render_multiplier_radio_id) if self.render_multiplier_radio_id else RESOLUTION_LABELS[0]

        supersample = SUPERSAMPLE_LOOKUP.get(supersample_label, SUPERSAMPLE_CHOICES[0])
        multiplier = RESOLUTION_LOOKUP.get(multiplier_label, RESOLUTION_CHOICES[0])

        passes = int(dpg.get_value(self.render_passes_input_id)) if self.render_passes_input_id is not None else defaults.DEFAULT_RENDER_PASSES
        streamlength = float(dpg.get_value(self.render_streamlength_input_id)) if self.render_streamlength_input_id is not None else defaults.DEFAULT_STREAMLENGTH_FACTOR
        margin = float(dpg.get_value(self.render_margin_input_id)) if self.render_margin_input_id is not None else defaults.DEFAULT_PADDING_MARGIN
        noise_seed = int(dpg.get_value(self.render_seed_input_id)) if self.render_seed_input_id is not None else defaults.DEFAULT_NOISE_SEED
        noise_sigma = float(dpg.get_value(self.render_sigma_input_id)) if self.render_sigma_input_id is not None else defaults.DEFAULT_NOISE_SIGMA

        # Clamp to valid ranges similar to pygame UI
        passes = max(passes, 1)
        streamlength = max(streamlength, 1e-6)
        margin = max(margin, 0.0)
        noise_sigma = max(noise_sigma, 0.0)

        with self.state_lock:
            actions.set_supersample(self.state, supersample)
            actions.set_render_multiplier(self.state, multiplier)
            actions.set_num_passes(self.state, passes)
            actions.set_margin(self.state, margin)
            actions.set_noise_seed(self.state, noise_seed)
            actions.set_noise_sigma(self.state, noise_sigma)
            actions.set_streamlength_factor(self.state, streamlength)

        self._close_render_modal()
        self._start_render_job()

    # ------------------------------------------------------------------
    # Canvas input processing
    # ------------------------------------------------------------------
    def _process_canvas_mouse(self) -> None:
        if dpg is None or self.canvas_id is None:
            return

        mouse_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        pressed = mouse_down and not self.mouse_down_last
        released = (not mouse_down) and self.mouse_down_last

        with self.state_lock:
            mode = self.state.view_mode

        wheel_delta = self.mouse_wheel_delta
        self.mouse_wheel_delta = 0.0
        over_canvas = self._is_mouse_over_canvas()

        if mode == "edit" and over_canvas and abs(wheel_delta) > 1e-5:
            with self.state_lock:
                selected_idx = self.state.selected_idx
            # Only scale if a conductor is selected
            if selected_idx >= 0:
                scale_factor = math.exp(wheel_delta * 0.12)
                scale_factor = max(0.05, min(scale_factor, 20.0))
                if self._scale_conductor(selected_idx, scale_factor):
                    x, y = self._get_canvas_mouse_pos()
                    self.drag_last_pos = (x, y)

        if mode != "edit":
            if pressed and self._is_mouse_over_canvas():
                x, y = self._get_canvas_mouse_pos()
                with self.state_lock:
                    project = self.state.project
                    hit_idx = -1
                    for idx in reversed(range(len(project.conductors))):
                        if _point_in_conductor(project.conductors[idx], x, y):
                            hit_idx = idx
                            break
                    self.state.set_selected(hit_idx)
                self._update_conductor_slider_labels()
            self.mouse_down_last = mouse_down
            return

        if pressed and self._is_mouse_over_canvas():
            x, y = self._get_canvas_mouse_pos()
            with self.state_lock:
                project = self.state.project
                hit_idx = -1
                for idx in reversed(range(len(project.conductors))):
                    if _point_in_conductor(project.conductors[idx], x, y):
                        hit_idx = idx
                        break

                self.state.set_selected(hit_idx)
                if hit_idx >= 0:
                    self.drag_active = True
                    self.drag_last_pos = (x, y)
                else:
                    self.drag_active = False
            self._update_conductor_slider_labels()
            self._mark_canvas_dirty()

        if self.drag_active and mouse_down:
            x, y = self._get_canvas_mouse_pos()
            dx = x - self.drag_last_pos[0]
            dy = y - self.drag_last_pos[1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                with self.state_lock:
                    idx = self.state.selected_idx
                    if 0 <= idx < len(self.state.project.conductors):
                        actions.move_conductor(self.state, idx, dx, dy)
                self.drag_last_pos = (x, y)
                self._mark_canvas_dirty()

        if self.drag_active and released:
            self.drag_active = False

        self.mouse_down_last = mouse_down

    def _process_keyboard_shortcuts(self) -> None:
        if dpg is None or BACKSPACE_KEY is None:
            return

        backspace_down = dpg.is_key_down(BACKSPACE_KEY)
        if backspace_down and not self.backspace_down_last:
            with self.state_lock:
                mode = self.state.view_mode
                idx = self.state.selected_idx
                conductor_count = len(self.state.project.conductors)
                can_delete = (mode == "edit" and 0 <= idx < conductor_count)
            if can_delete:
                with self.state_lock:
                    actions.remove_conductor(self.state, idx)
                    self.conductor_textures.clear()
                    self.conductor_texture_shapes.clear()
                self._mark_canvas_dirty()
                self._rebuild_conductor_controls()
                dpg.set_value("status_text", "Conductor deleted")
        self.backspace_down_last = backspace_down

    def _apply_canvas_size(self, sender=None, app_data=None) -> None:
        if dpg is None:
            return

        width = int(dpg.get_value(self.canvas_width_input_id)) if self.canvas_width_input_id is not None else self.state.project.canvas_resolution[0]
        height = int(dpg.get_value(self.canvas_height_input_id)) if self.canvas_height_input_id is not None else self.state.project.canvas_resolution[1]

        width = max(1, min(width, 32768))
        height = max(1, min(height, 32768))

        with self.state_lock:
            current_size = self.state.project.canvas_resolution
            if current_size == (width, height):
                return
            actions.set_canvas_resolution(self.state, width, height)
        self._update_canvas_inputs()
        self._mark_canvas_dirty()
        dpg.set_value("status_text", f"Canvas resized to {width}×{height}")

    def _on_back_to_edit_clicked(self, sender, app_data):
        with self.state_lock:
            if self.state.view_mode != "edit":
                self.state.view_mode = "edit"
                self.drag_active = False
                self._mark_canvas_dirty()
        self._update_control_visibility()
        dpg.set_value("status_text", "Edit mode.")

    def _on_downsample_slider(self, sender, app_data):
        """Handle downsampling blur sigma slider change."""
        value = float(app_data)
        with self.state_lock:
            self.state.postprocess_settings.downsample_sigma = value
        self._apply_postprocessing()

    def _on_clip_slider(self, sender, app_data):
        """Handle clip percent slider change."""
        value = float(app_data)
        with self.state_lock:
            self.state.postprocess_settings.clip_percent = value
        # Clip is display-only, just refresh texture
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_contrast_slider(self, sender, app_data):
        """Handle contrast slider change."""
        value = float(app_data)
        with self.state_lock:
            self.state.postprocess_settings.contrast = value
        # Contrast is display-only, just refresh texture
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_gamma_slider(self, sender, app_data):
        """Handle gamma slider change."""
        value = float(app_data)
        with self.state_lock:
            self.state.postprocess_settings.gamma = value
        # Gamma is display-only, just refresh texture
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _redraw_canvas(self) -> None:
        if dpg is None or self.canvas_id is None:
            return
        self.canvas_dirty = False

        self._refresh_render_texture()

        with self.state_lock:
            project = self.state.project
            selected_idx = self.state.selected_idx
            canvas_w, canvas_h = project.canvas_resolution
            conductors = list(project.conductors)
            render_cache = self.state.render_cache
            view_mode = self.state.view_mode

        dpg.delete_item(self.canvas_id, children_only=True)

        dpg.configure_item(self.canvas_id, width=canvas_w, height=canvas_h)

        dpg.draw_rectangle((0, 0), (canvas_w, canvas_h), color=(60, 60, 60, 255), fill=(20, 20, 20, 255), parent=self.canvas_id)

        if view_mode == "render" and render_cache and self.render_texture_id is not None and self.render_texture_size:
            tex_w, tex_h = self.render_texture_size
            if tex_w > 0 and tex_h > 0:
                scale_x = canvas_w / tex_w
                scale_y = canvas_h / tex_h
                pmax = (tex_w * scale_x, tex_h * scale_y)
                dpg.draw_image(
                    self.render_texture_id,
                    (0, 0),
                    pmax,
                    uv_min=(0.0, 0.0),
                    uv_max=(1.0, 1.0),
                    parent=self.canvas_id,
                )

        if view_mode == "edit" or render_cache is None:
            for idx, conductor in enumerate(conductors):
                tex_id = self._ensure_conductor_texture(idx, conductor.mask)
                x0, y0 = conductor.position
                width = conductor.mask.shape[1]
                height = conductor.mask.shape[0]
                dpg.draw_image(
                    tex_id,
                    pmin=(x0, y0),
                    pmax=(x0 + width, y0 + height),
                    uv_min=(0.0, 0.0),
                    uv_max=(1.0, 1.0),
                    parent=self.canvas_id,
                )
                if idx == selected_idx:
                    dpg.draw_rectangle(
                        (x0, y0),
                        (x0 + width, y0 + height),
                        color=(255, 255, 100, 200),
                        thickness=2.0,
                        parent=self.canvas_id,
                    )

    # ------------------------------------------------------------------
    # Canvas input processing
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _start_render_job(self) -> None:
        if self.render_future is not None and not self.render_future.done():
            return

        self.render_error = None
        dpg.set_value("status_text", "Rendering...")

        def job() -> bool:
            with self.state_lock:
                settings_snapshot: RenderSettings = replace(self.state.render_settings)
                project_snapshot = _snapshot_project(self.state.project)
            result = perform_render(
                project_snapshot,
                settings_snapshot.multiplier,
                settings_snapshot.supersample,
                settings_snapshot.num_passes,
                settings_snapshot.margin,
                settings_snapshot.noise_seed,
                settings_snapshot.noise_sigma,
                project_snapshot.streamlength_factor,
            )
            if result is None:
                return False

            # Apply initial postprocessing to create display array
            from flowcol.render import downsample_lic
            with self.state_lock:
                postprocess = self.state.postprocess_settings
                target_shape = project_snapshot.canvas_resolution
                display_array = downsample_lic(
                    result.array,
                    target_shape,
                    settings_snapshot.supersample,
                    postprocess.downsample_sigma,
                )

            cache = RenderCache(
                result=result,
                multiplier=settings_snapshot.multiplier,
                supersample=settings_snapshot.supersample,
                display_array=display_array,
            )

            with self.state_lock:
                self.state.render_cache = cache
                self.state.field_dirty = False
                self.state.render_dirty = False
                self.state.view_mode = "render"
            return True

        self.render_future = self.executor.submit(job)

    def _poll_render_future(self) -> None:
        if self.render_future is None:
            return
        if not self.render_future.done():
            return

        success = False
        try:
            success = bool(self.render_future.result())
        except Exception as exc:  # pragma: no cover - unexpected
            success = False
            self.render_error = str(exc)

        self.render_future = None

        if success:
            self._mark_canvas_dirty()
            self._refresh_render_texture()
            self.drag_active = False
            with self.state_lock:
                self.state.view_mode = "render"
            self._update_control_visibility()
            dpg.set_value("status_text", "Render complete.")
        else:
            msg = self.render_error or "Render failed (possibly due to excessive resolution)."
            dpg.set_value("status_text", msg)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def render(self) -> None:
        """Run Dear PyGui event loop."""
        self.require_backend()
        if not self.viewport_created:
            self.build()

        dpg.setup_dearpygui()
        dpg.show_viewport()

        try:
            while dpg.is_dearpygui_running():
                self._process_canvas_mouse()
                self._process_keyboard_shortcuts()
                self._poll_render_future()
                if self.canvas_dirty:
                    self._redraw_canvas()
                dpg.render_dearpygui_frame()
        finally:
            self.executor.shutdown(wait=False)
            dpg.destroy_context()


    def _update_control_visibility(self) -> None:
        if dpg is None:
            return
        with self.state_lock:
            mode = self.state.view_mode
        if self.edit_controls_id is not None:
            dpg.configure_item(self.edit_controls_id, show=(mode == "edit"))
        if self.render_controls_id is not None:
            dpg.configure_item(self.render_controls_id, show=(mode == "render"))
        self._update_conductor_slider_labels()

def run() -> None:
    """Launch FlowCol Dear PyGui application."""
    FlowColApp().render()
