"""File I/O controller for Elliptica UI - boundary import and project save/load."""

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from elliptica.app import actions
from elliptica.mask_utils import load_boundary_masks
from elliptica.pde import PDERegistry
from elliptica.serialization import load_project, save_project, load_render_cache, save_render_cache
from elliptica.types import BoundaryObject

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


MAX_CANVAS_DIM = 8192


class FileIOController:
    """Controller for file I/O operations - boundary import and project save/load."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app

        # File dialog IDs
        self.boundary_file_dialog_id: Optional[int] = None
        self.save_project_dialog_id: Optional[int] = None
        self.load_project_dialog_id: Optional[int] = None

        # Track current project path for auto-save
        self.current_project_path: Optional[str] = None

        # Overwrite confirmation
        self._pending_save_project_path: Optional[Path] = None
        self._overwrite_confirm_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Boundary import
    # ------------------------------------------------------------------

    def ensure_boundary_file_dialog(self) -> None:
        """Create the boundary file dialog if it doesn't exist."""
        if dpg is None or self.boundary_file_dialog_id is not None:
            return

        # Default to assets/masks if it exists, otherwise use cwd
        masks_path = Path.cwd() / "assets" / "masks"
        default_path = str(masks_path) if masks_path.exists() else str(Path.cwd())

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            default_path=default_path,
            callback=self.on_boundary_file_selected,
            cancel_callback=self.on_boundary_file_cancelled,
            width=640,
            height=420,
            tag="boundary_file_dialog",
        ) as dialog:
            self.boundary_file_dialog_id = dialog
            dpg.add_file_extension(".png", color=(150, 180, 255, 255))
            dpg.add_file_extension(".*")

    def open_boundary_dialog(self, sender=None, app_data=None) -> None:
        """Open the boundary file dialog.

        Checks if in edit mode before opening.
        """
        if dpg is None:
            return
        with self.app.state_lock:
            if self.app.state.view_mode != "edit":
                dpg.set_value("status_text", "Switch to edit mode to add boundaries.")
                return

        self.ensure_boundary_file_dialog()
        if self.boundary_file_dialog_id is not None:
            dpg.show_item(self.boundary_file_dialog_id)

    def on_boundary_file_cancelled(self, sender=None, app_data=None) -> None:
        """Handle boundary file dialog cancellation."""
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)
        dpg.set_value("status_text", "Load boundary cancelled.")

    def on_boundary_file_selected(self, sender=None, app_data=None) -> None:
        """Handle boundary file selection and import."""
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)

        path_str: Optional[str] = None
        if isinstance(app_data, dict):
            # Try selections dict FIRST - it's more reliable than file_path_name
            # (DPG has a bug where file_path_name becomes ".png" on second use)
            selections = app_data.get("selections", {})
            if selections:
                path_str = next(iter(selections.values()))

            if not path_str:
                # Fallback: try file_path_name
                path_str = app_data.get("file_path_name")

            if not path_str:
                # Fallback: combine current_path + file_name
                current_path = app_data.get("current_path", "")
                file_name = app_data.get("file_name", "")
                if current_path and file_name:
                    path_str = str(Path(current_path) / file_name)

        if not path_str:
            dpg.set_value("status_text", "No file selected.")
            return

        # Convert to absolute path to handle relative paths
        try:
            path_obj = Path(path_str)
            if not path_obj.is_absolute():
                path_obj = path_obj.resolve()
            path_str = str(path_obj)
        except Exception:
            pass  # If path resolution fails, try with original path_str

        try:
            mask, interior = load_boundary_masks(path_str)
        except Exception as exc:  # pragma: no cover - PIL errors etc.
            dpg.set_value("status_text", f"Failed to load boundary: {exc}")
            return

        mask_h, mask_w = mask.shape
        if mask_w > MAX_CANVAS_DIM or mask_h > MAX_CANVAS_DIM:
            dpg.set_value("status_text", f"Mask exceeds max dimension {MAX_CANVAS_DIM}px.")
            return

        with self.app.state_lock:
            project = self.app.state.project
            canvas_w, canvas_h = project.canvas_resolution
            # Offset each new boundary by 30px down-right from center so they're all visible
            num_boundaries = len(project.boundary_objects)
            offset = num_boundaries * 30.0
            pos = ((canvas_w - mask_w) / 2.0 + offset, (canvas_h - mask_h) / 2.0 + offset)
            boundary = BoundaryObject(mask=mask, params={"voltage": 0.5}, position=pos, interior_mask=interior)
            actions.add_boundary(self.app.state, boundary)
            self.app.state.view_mode = "edit"

        self.app.canvas_renderer.mark_dirty()

        # Resize drawlist widget if canvas was expanded
        if self.app.canvas_id is not None:
            with self.app.state_lock:
                canvas_w, canvas_h = self.app.state.project.canvas_resolution
            dpg.configure_item(self.app.canvas_id, width=canvas_w, height=canvas_h)

        self.app._update_canvas_inputs()  # Update canvas size display text
        self.app._update_canvas_transform()  # Recalculate scale for potentially new canvas size
        self.app._update_control_visibility()
        self.app.boundary_controls.rebuild_controls()
        self.app.boundary_controls.update_slider_labels()
        dpg.set_value("status_text", f"Loaded boundary '{Path(path_str).name}'")

    # ------------------------------------------------------------------
    # Project save/load
    # ------------------------------------------------------------------

    def new_project(self, sender=None, app_data=None) -> None:
        """Create a new project, resetting to default state with a demo boundary."""
        if dpg is None:
            return

        # Reset PDERegistry to default (poisson)
        PDERegistry.set_active("poisson")

        with self.app.state_lock:
            # Reset to default state
            from elliptica.app.core import AppState
            new_state = AppState()

            # Copy over the new state
            self.app.state.project = new_state.project
            self.app.state.render_settings = new_state.render_settings
            self.app.state.display_settings = new_state.display_settings
            self.app.state.boundary_color_settings = new_state.boundary_color_settings
            self.app.state.clear_selection()
            self.app.state.view_mode = "edit"
            self.app.state.field_dirty = True
            self.app.state.render_dirty = True
            self.app.state.render_cache = None

        # Clear current project path
        self.current_project_path = None

        # Add demo boundary
        self.app._add_demo_boundary()

        # Reset zoom/pan to defaults for new project
        self.app.canvas_controller.reset_zoom_pan()

        # Update UI to reflect new state
        self.app.canvas_renderer.mark_dirty()
        self.app._update_canvas_inputs()

        # Resize drawlist widget to match new canvas resolution
        if self.app.canvas_id is not None:
            canvas_w, canvas_h = self.app.state.project.canvas_resolution
            dpg.configure_item(self.app.canvas_id, width=canvas_w, height=canvas_h)

        self.app._update_canvas_transform()
        self.app._update_control_visibility()
        self.app.boundary_controls.rebuild_controls()
        self.app.boundary_controls.update_slider_labels()
        self.sync_ui_from_state()
        self.app.cache_panel.update_cache_status_display()

        dpg.set_value("status_text", "New project created.")

    def ensure_save_project_dialog(self) -> None:
        """Create the save project dialog if it doesn't exist."""
        if dpg is None or self.save_project_dialog_id is not None:
            return

        # Default to projects/ directory
        projects_path = Path.cwd() / "projects"
        default_path = str(projects_path) if projects_path.exists() else str(Path.cwd())

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            default_path=default_path,
            callback=self.on_save_project_file_selected,
            cancel_callback=self.on_save_project_cancelled,
            width=640,
            height=420,
            tag="save_project_dialog",
            default_filename="project.elliptica",
        ) as dialog:
            self.save_project_dialog_id = dialog
            dpg.add_file_extension(".elliptica", color=(180, 255, 150, 255))
            dpg.add_file_extension(".*")

    def ensure_load_project_dialog(self) -> None:
        """Create the load project dialog if it doesn't exist."""
        if dpg is None or self.load_project_dialog_id is not None:
            return

        # Default to projects/ directory
        projects_path = Path.cwd() / "projects"
        default_path = str(projects_path) if projects_path.exists() else str(Path.cwd())

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            default_path=default_path,
            callback=self.on_load_project_file_selected,
            cancel_callback=self.on_load_project_cancelled,
            width=640,
            height=420,
            tag="load_project_dialog",
        ) as dialog:
            self.load_project_dialog_id = dialog
            dpg.add_file_extension(".elliptica", color=(180, 255, 150, 255))
            dpg.add_file_extension(".*")

    def open_save_project_dialog(self, sender=None, app_data=None) -> None:
        """Open the save project dialog."""
        if dpg is None:
            return
        self.ensure_save_project_dialog()
        if self.save_project_dialog_id is not None:
            dpg.show_item(self.save_project_dialog_id)

    def open_load_project_dialog(self, sender=None, app_data=None) -> None:
        """Open the load project dialog."""
        if dpg is None:
            return
        self.ensure_load_project_dialog()
        if self.load_project_dialog_id is not None:
            dpg.show_item(self.load_project_dialog_id)

    def on_save_project_cancelled(self, sender=None, app_data=None) -> None:
        """Handle save project dialog cancellation."""
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)
        dpg.set_value("status_text", "Save project cancelled.")

    def on_load_project_cancelled(self, sender=None, app_data=None) -> None:
        """Handle load project dialog cancellation."""
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)
        dpg.set_value("status_text", "Load project cancelled.")

    def on_save_project_file_selected(self, sender=None, app_data=None) -> None:
        """Handle save project file selection - save project and cache."""
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)

        path_str = self._extract_file_path(app_data, prefer_typed_name=True)
        if not path_str:
            dpg.set_value("status_text", "No file selected.")
            return

        path_obj = Path(path_str)
        if path_obj.suffix != '.elliptica':
            path_obj = path_obj.with_suffix('.elliptica')

        # Check if file exists and confirm overwrite
        if path_obj.exists():
            self._pending_save_project_path = path_obj
            self._show_overwrite_confirm()
            return

        self._do_save_project(path_obj)

    def _do_save_project(self, path_obj: Path) -> None:
        """Actually perform the project save."""
        if dpg is None:
            return

        try:
            with self.app.state_lock:
                save_project(self.app.state, str(path_obj))

                # Track current project path
                self.current_project_path = str(path_obj)

                # Also save render cache if it exists
                if self.app.state.render_cache is not None:
                    cache_path = path_obj.with_suffix('.elliptica.cache')
                    save_render_cache(self.app.state.render_cache, self.app.state.project, str(cache_path))
                    cache_size_mb = cache_path.stat().st_size / 1024 / 1024
                    dpg.set_value("status_text", f"Saved project + cache ({cache_size_mb:.1f} MB): {path_obj.name}")
                else:
                    dpg.set_value("status_text", f"Saved project: {path_obj.name}")
        except Exception as exc:
            dpg.set_value("status_text", f"Failed to save project: {exc}")

    def _show_overwrite_confirm(self) -> None:
        """Show overwrite confirmation dialog for project save."""
        if dpg is None:
            return

        # Create dialog if needed
        if self._overwrite_confirm_id is None:
            with dpg.window(
                label="Confirm Overwrite",
                modal=True,
                show=False,
                tag="project_overwrite_confirm_modal",
                no_resize=True,
                no_collapse=True,
                width=350,
                height=120,
            ) as modal:
                self._overwrite_confirm_id = modal
                dpg.add_text("", tag="project_overwrite_confirm_text")
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Overwrite", width=100, callback=self._on_overwrite_confirmed)
                    dpg.add_button(label="Cancel", width=100, callback=self._on_overwrite_cancelled)

        # Set the message
        filename = self._pending_save_project_path.name if self._pending_save_project_path else "file"
        dpg.set_value("project_overwrite_confirm_text", f"'{filename}' already exists. Overwrite?")

        # Center and show
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.configure_item(
            self._overwrite_confirm_id,
            pos=((viewport_width - 350) // 2, (viewport_height - 120) // 2),
            show=True
        )

    def _on_overwrite_confirmed(self, sender=None, app_data=None) -> None:
        """Handle overwrite confirmation."""
        if dpg is None:
            return

        dpg.configure_item("project_overwrite_confirm_modal", show=False)

        if self._pending_save_project_path:
            self._do_save_project(self._pending_save_project_path)
            self._pending_save_project_path = None

    def _on_overwrite_cancelled(self, sender=None, app_data=None) -> None:
        """Handle overwrite cancellation."""
        if dpg is None:
            return

        dpg.configure_item("project_overwrite_confirm_modal", show=False)
        self._pending_save_project_path = None
        dpg.set_value("status_text", "Save cancelled.")

    def on_load_project_file_selected(self, sender=None, app_data=None) -> None:
        """Handle load project file selection - load project and cache."""
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)

        path_str = self._extract_file_path(app_data)
        if not path_str:
            dpg.set_value("status_text", "No file selected.")
            return

        try:
            new_state = load_project(path_str)

            # Try to load render cache
            project_path = Path(path_str)
            cache_path = project_path.with_suffix('.elliptica.cache')
            loaded_cache = load_render_cache(str(cache_path), new_state.project)

            # Sync PDERegistry to loaded project's PDE type
            PDERegistry.set_active(new_state.project.pde_type)

            with self.app.state_lock:
                # Replace current state with loaded state
                self.app.state.project = new_state.project
                self.app.state.render_settings = new_state.render_settings
                self.app.state.display_settings = new_state.display_settings
                self.app.state.boundary_color_settings = new_state.boundary_color_settings
                self.app.state.clear_selection()
                self.app.state.view_mode = "edit"
                self.app.state.field_dirty = True
                self.app.state.render_dirty = True
                self.app.state.render_cache = loaded_cache

            # Track current project path
            self.current_project_path = path_str

            # Rebuild display fields from loaded cache
            if loaded_cache is not None:
                self.app.cache_panel.rebuild_cache_display_fields()

            # Reset zoom/pan to defaults when loading project
            self.app.canvas_controller.reset_zoom_pan()

            # Update UI to reflect loaded state
            self.app.canvas_renderer.mark_dirty()
            self.app._update_canvas_inputs()  # Update canvas size input fields

            # Resize drawlist widget to match loaded canvas resolution
            if self.app.canvas_id is not None:
                canvas_w, canvas_h = self.app.state.project.canvas_resolution
                dpg.configure_item(self.app.canvas_id, width=canvas_w, height=canvas_h)

            self.app._update_canvas_transform()  # Recalculate scale for new canvas resolution
            self.app._update_control_visibility()
            self.app.boundary_controls.rebuild_controls()
            self.app.boundary_controls.update_slider_labels()
            self.sync_ui_from_state()
            self.app.cache_panel.update_cache_status_display()

            # Status message
            if loaded_cache:
                shape = loaded_cache.result.array.shape
                dpg.set_value("status_text", f"Loaded project with render cache ({shape[1]}×{shape[0]})")
            else:
                dpg.set_value("status_text", f"Loaded project: {Path(path_str).name}")
        except Exception as exc:
            dpg.set_value("status_text", f"Failed to load project: {exc}")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _extract_file_path(self, app_data, prefer_typed_name: bool = False) -> Optional[str]:
        """Extract file path from DPG file dialog callback data.

        Args:
            app_data: Callback data from DPG file dialog
            prefer_typed_name: If True, prioritize file_name field over selections.
                              Use True for save dialogs, False for load dialogs.
        """
        if not isinstance(app_data, dict):
            return None

        if prefer_typed_name:
            # For save dialogs: prioritize the typed filename over clicked file
            current_path = app_data.get("current_path", "")
            file_name = app_data.get("file_name", "")
            if current_path and file_name:
                return str(Path(current_path) / file_name)

            # Fallback: file_path_name
            path_str = app_data.get("file_path_name")
            if path_str:
                return path_str

            # Last resort: selections
            selections = app_data.get("selections", {})
            if selections:
                return next(iter(selections.values()))
        else:
            # For load dialogs: prioritize selections (clicked file)
            selections = app_data.get("selections", {})
            if selections:
                return next(iter(selections.values()))

            # Fallback: file_path_name
            path_str = app_data.get("file_path_name")
            if path_str:
                return path_str

            # Fallback: combine current_path + file_name
            current_path = app_data.get("current_path", "")
            file_name = app_data.get("file_name", "")
            if current_path and file_name:
                return str(Path(current_path) / file_name)

        return None

    def sync_ui_from_state(self) -> None:
        """Sync UI controls to match current state after loading."""
        if dpg is None:
            return

        with self.app.state_lock:
            # Canvas resolution
            if self.app.canvas_width_input_id is not None:
                dpg.set_value(self.app.canvas_width_input_id, self.app.state.project.canvas_resolution[0])
            if self.app.canvas_height_input_id is not None:
                dpg.set_value(self.app.canvas_height_input_id, self.app.state.project.canvas_resolution[1])

            # Display settings
            panel = self.app.postprocess_panel
            if panel.postprocess_clip_low_slider_id is not None:
                dpg.set_value(panel.postprocess_clip_low_slider_id, self.app.state.display_settings.clip_low_percent)
            if panel.postprocess_clip_high_slider_id is not None:
                dpg.set_value(panel.postprocess_clip_high_slider_id, self.app.state.display_settings.clip_high_percent)
            if panel.postprocess_brightness_slider_id is not None:
                dpg.set_value(panel.postprocess_brightness_slider_id, self.app.state.display_settings.brightness)
            if panel.postprocess_contrast_slider_id is not None:
                dpg.set_value(panel.postprocess_contrast_slider_id, self.app.state.display_settings.contrast)
            if panel.postprocess_gamma_slider_id is not None:
                dpg.set_value(panel.postprocess_gamma_slider_id, self.app.state.display_settings.gamma)
            if dpg.does_item_exist("saturation_slider"):
                dpg.set_value("saturation_slider", self.app.state.display_settings.saturation)

            # PDE type combo - use full label format from pde_label_map
            if self.app.pde_combo_id is not None:
                label = self.app._label_for_active_pde()
                if label:
                    dpg.set_value(self.app.pde_combo_id, label)

            # Color mode radio - restore expressions mode if color_config was set
            has_color_config = self.app.state.color_config is not None
            if dpg.does_item_exist("color_mode_radio"):
                mode_value = "Expressions" if has_color_config else "Palette"
                dpg.set_value("color_mode_radio", mode_value)
                panel.color_mode = "expressions" if has_color_config else "palette"
                # Update visibility of mode groups
                dpg.configure_item("palette_mode_group", show=(panel.color_mode == "palette"))
                dpg.configure_item("expressions_mode_group", show=(panel.color_mode == "expressions"))

            # Lightness expression checkbox and input (global mode)
            lightness_expr = self.app.state.display_settings.lightness_expr
            if dpg.does_item_exist("lightness_expr_checkbox"):
                dpg.set_value("lightness_expr_checkbox", lightness_expr is not None)
            if dpg.does_item_exist("lightness_expr_group"):
                dpg.configure_item("lightness_expr_group", show=(lightness_expr is not None))
            if dpg.does_item_exist("lightness_expr_input") and lightness_expr is not None:
                dpg.set_value("lightness_expr_input", lightness_expr)

        # Sync global palette UI text (outside lock - no state modification)
        self.sync_palette_ui()

        # Update context-dependent UI (slider states, region controls visibility, etc.)
        self.app.postprocess_panel.update_context_ui()

    def sync_palette_ui(self) -> None:
        """Sync global palette UI text to match current state.

        Called when switching to render mode or loading a project.
        """
        if dpg is None:
            return

        with self.app.state_lock:
            palette_text = (self.app.state.display_settings.palette
                          if self.app.state.display_settings.color_enabled
                          else "Grayscale")

        dpg.configure_item("global_palette_button", label=palette_text)

    def auto_save_cache(self) -> None:
        """Auto-save render cache to disk (called after successful render)."""
        if self.current_project_path is None:
            return

        try:
            project_path = Path(self.current_project_path)
            cache_path = project_path.with_suffix('.elliptica.cache')
            with self.app.state_lock:
                if self.app.state.render_cache is not None:
                    save_render_cache(self.app.state.render_cache, self.app.state.project, str(cache_path))
                    cache_size_mb = cache_path.stat().st_size / 1024 / 1024
                    dpg.set_value("status_text", f"Render complete. Cache saved ({cache_size_mb:.1f} MB)")
        except Exception as exc:
            dpg.set_value("status_text", f"Render complete. Failed to save cache: {exc}")
