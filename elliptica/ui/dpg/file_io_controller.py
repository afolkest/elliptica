"""File I/O controller for Elliptica UI - conductor import and project save/load."""

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from elliptica.app import actions
from elliptica.mask_utils import load_conductor_masks
from elliptica.serialization import load_project, save_project, load_render_cache, save_render_cache
from elliptica.types import Conductor

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


MAX_CANVAS_DIM = 8192


class FileIOController:
    """Controller for file I/O operations - conductor import and project save/load."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app

        # File dialog IDs
        self.conductor_file_dialog_id: Optional[int] = None
        self.save_project_dialog_id: Optional[int] = None
        self.load_project_dialog_id: Optional[int] = None

        # Track current project path for auto-save
        self.current_project_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Conductor import
    # ------------------------------------------------------------------

    def ensure_conductor_file_dialog(self) -> None:
        """Create the conductor file dialog if it doesn't exist."""
        if dpg is None or self.conductor_file_dialog_id is not None:
            return

        # Default to assets/masks if it exists, otherwise use cwd
        masks_path = Path.cwd() / "assets" / "masks"
        default_path = str(masks_path) if masks_path.exists() else str(Path.cwd())

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            default_path=default_path,
            callback=self.on_conductor_file_selected,
            cancel_callback=self.on_conductor_file_cancelled,
            width=640,
            height=420,
            tag="conductor_file_dialog",
        ) as dialog:
            self.conductor_file_dialog_id = dialog
            dpg.add_file_extension(".png", color=(150, 180, 255, 255))
            dpg.add_file_extension(".*")

    def open_conductor_dialog(self, sender=None, app_data=None) -> None:
        """Open the conductor file dialog.

        Checks if in edit mode before opening.
        """
        if dpg is None:
            return
        with self.app.state_lock:
            if self.app.state.view_mode != "edit":
                dpg.set_value("status_text", "Switch to edit mode to add conductors.")
                return

        self.ensure_conductor_file_dialog()
        if self.conductor_file_dialog_id is not None:
            dpg.show_item(self.conductor_file_dialog_id)

    def on_conductor_file_cancelled(self, sender=None, app_data=None) -> None:
        """Handle conductor file dialog cancellation."""
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)
        dpg.set_value("status_text", "Load conductor cancelled.")

    def on_conductor_file_selected(self, sender=None, app_data=None) -> None:
        """Handle conductor file selection and import."""
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
            mask, interior = load_conductor_masks(path_str)
        except Exception as exc:  # pragma: no cover - PIL errors etc.
            dpg.set_value("status_text", f"Failed to load conductor: {exc}")
            return

        mask_h, mask_w = mask.shape
        if mask_w > MAX_CANVAS_DIM or mask_h > MAX_CANVAS_DIM:
            dpg.set_value("status_text", f"Mask exceeds max dimension {MAX_CANVAS_DIM}px.")
            return

        with self.app.state_lock:
            project = self.app.state.project
            if len(project.conductors) == 0:
                canvas_w, canvas_h = project.canvas_resolution
                new_w = max(canvas_w, mask_w)
                new_h = max(canvas_h, mask_h)
                if (new_w, new_h) != project.canvas_resolution:
                    actions.set_canvas_resolution(self.app.state, new_w, new_h)

            canvas_w, canvas_h = self.app.state.project.canvas_resolution
            # Offset each new conductor by 30px down-right from center so they're all visible
            num_conductors = len(project.conductors)
            offset = num_conductors * 30.0
            pos = ((canvas_w - mask_w) / 2.0 + offset, (canvas_h - mask_h) / 2.0 + offset)
            conductor = Conductor(mask=mask, voltage=0.5, position=pos, interior_mask=interior)
            actions.add_conductor(self.app.state, conductor)
            self.app.state.view_mode = "edit"

        self.app.canvas_renderer.mark_dirty()

        # Resize drawlist widget if canvas was expanded
        if self.app.canvas_id is not None:
            with self.app.state_lock:
                canvas_w, canvas_h = self.app.state.project.canvas_resolution
            dpg.configure_item(self.app.canvas_id, width=canvas_w, height=canvas_h)

        self.app._update_canvas_scale()  # Recalculate scale for potentially new canvas size
        self.app._update_control_visibility()
        self.app.conductor_controls.rebuild_conductor_controls()
        self.app.conductor_controls.update_conductor_slider_labels()
        self.app.boundary_controls.rebuild_controls()
        dpg.set_value("status_text", f"Loaded conductor '{Path(path_str).name}'")

    # ------------------------------------------------------------------
    # Project save/load
    # ------------------------------------------------------------------

    def new_project(self, sender=None, app_data=None) -> None:
        """Create a new project, resetting to default state with a demo conductor."""
        if dpg is None:
            return

        with self.app.state_lock:
            # Reset to default state
            from elliptica.app.core import AppState
            new_state = AppState()

            # Copy over the new state
            self.app.state.project = new_state.project
            self.app.state.render_settings = new_state.render_settings
            self.app.state.display_settings = new_state.display_settings
            self.app.state.conductor_color_settings = new_state.conductor_color_settings
            self.app.state.selected_idx = -1
            self.app.state.view_mode = "edit"
            self.app.state.field_dirty = True
            self.app.state.render_dirty = True
            self.app.state.render_cache = None

        # Clear current project path
        self.current_project_path = None

        # Add demo conductor
        self.app._add_demo_conductor()

        # Update UI to reflect new state
        self.app.canvas_renderer.mark_dirty()
        self.app._update_canvas_inputs()

        # Resize drawlist widget to match new canvas resolution
        if self.app.canvas_id is not None:
            canvas_w, canvas_h = self.app.state.project.canvas_resolution
            dpg.configure_item(self.app.canvas_id, width=canvas_w, height=canvas_h)

        self.app._update_canvas_scale()
        self.app._update_control_visibility()
        self.app.conductor_controls.rebuild_conductor_controls()
        self.app.conductor_controls.update_conductor_slider_labels()
        self.app.boundary_controls.rebuild_controls()
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
            modal=True,
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
            dpg.add_file_extension(".flowcol", color=(150, 180, 255, 255))  # Legacy support
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
            # Order matters - first extension is the default filter
            # Use .* first so all project files are visible by default
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".elliptica", color=(180, 255, 150, 255))
            dpg.add_file_extension(".flowcol", color=(150, 180, 255, 255))  # Legacy support

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

        # Ensure valid extension (.elliptica preferred, .flowcol for legacy)
        path_obj = Path(path_str)
        if path_obj.suffix not in ('.elliptica', '.flowcol'):
            path_obj = path_obj.with_suffix('.elliptica')

        try:
            with self.app.state_lock:
                save_project(self.app.state, str(path_obj))

                # Track current project path
                self.current_project_path = str(path_obj)

                # Also save render cache if it exists (use matching cache extension)
                if self.app.state.render_cache is not None:
                    cache_suffix = '.elliptica.cache' if path_obj.suffix == '.elliptica' else '.flowcol.cache'
                    cache_path = path_obj.with_suffix(cache_suffix)
                    save_render_cache(self.app.state.render_cache, self.app.state.project, str(cache_path))
                    cache_size_mb = cache_path.stat().st_size / 1024 / 1024
                    dpg.set_value("status_text", f"Saved project + cache ({cache_size_mb:.1f} MB): {path_obj.name}")
                else:
                    dpg.set_value("status_text", f"Saved project: {path_obj.name}")
        except Exception as exc:
            dpg.set_value("status_text", f"Failed to save project: {exc}")

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

            # Try to load render cache (try matching extension first, then fallback)
            project_path = Path(path_str)
            loaded_cache = None
            if project_path.suffix == '.elliptica':
                cache_paths = [project_path.with_suffix('.elliptica.cache'), project_path.with_suffix('.flowcol.cache')]
            else:
                cache_paths = [project_path.with_suffix('.flowcol.cache'), project_path.with_suffix('.elliptica.cache')]
            for cache_path in cache_paths:
                loaded_cache = load_render_cache(str(cache_path), new_state.project)
                if loaded_cache is not None:
                    break

            with self.app.state_lock:
                # Replace current state with loaded state
                self.app.state.project = new_state.project
                self.app.state.render_settings = new_state.render_settings
                self.app.state.display_settings = new_state.display_settings
                self.app.state.conductor_color_settings = new_state.conductor_color_settings
                self.app.state.selected_idx = -1
                self.app.state.view_mode = "edit"
                self.app.state.field_dirty = True
                self.app.state.render_dirty = True
                self.app.state.render_cache = loaded_cache

            # Track current project path
            self.current_project_path = path_str

            # Rebuild display fields from loaded cache
            if loaded_cache is not None:
                self.app.cache_panel.rebuild_cache_display_fields()

            # Update UI to reflect loaded state
            self.app.canvas_renderer.mark_dirty()
            self.app._update_canvas_inputs()  # Update canvas size input fields

            # Resize drawlist widget to match loaded canvas resolution
            if self.app.canvas_id is not None:
                canvas_w, canvas_h = self.app.state.project.canvas_resolution
                dpg.configure_item(self.app.canvas_id, width=canvas_w, height=canvas_h)

            self.app._update_canvas_scale()  # Recalculate scale for new canvas resolution
            self.app._update_control_visibility()
            self.app.conductor_controls.rebuild_conductor_controls()
            self.app.conductor_controls.update_conductor_slider_labels()
            self.app.boundary_controls.rebuild_controls()
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
            if panel.postprocess_clip_slider_id is not None:
                dpg.set_value(panel.postprocess_clip_slider_id, self.app.state.display_settings.clip_percent)
            if panel.postprocess_brightness_slider_id is not None:
                dpg.set_value(panel.postprocess_brightness_slider_id, self.app.state.display_settings.brightness)
            if panel.postprocess_contrast_slider_id is not None:
                dpg.set_value(panel.postprocess_contrast_slider_id, self.app.state.display_settings.contrast)
            if panel.postprocess_gamma_slider_id is not None:
                dpg.set_value(panel.postprocess_gamma_slider_id, self.app.state.display_settings.gamma)

            # Sync global palette UI text
            palette_text = (self.app.state.display_settings.palette
                          if self.app.state.display_settings.color_enabled
                          else "Grayscale")
            dpg.set_value("global_palette_current_text", f"Current: {palette_text}")

    def auto_save_cache(self) -> None:
        """Auto-save render cache to disk (called after successful render)."""
        if self.current_project_path is None:
            return

        try:
            project_path = Path(self.current_project_path)
            cache_suffix = '.elliptica.cache' if project_path.suffix == '.elliptica' else '.flowcol.cache'
            cache_path = project_path.with_suffix(cache_suffix)
            with self.app.state_lock:
                if self.app.state.render_cache is not None:
                    save_render_cache(self.app.state.render_cache, self.app.state.project, str(cache_path))
                    cache_size_mb = cache_path.stat().st_size / 1024 / 1024
                    dpg.set_value("status_text", f"Render complete. Cache saved ({cache_size_mb:.1f} MB)")
        except Exception as exc:
            dpg.set_value("status_text", f"Render complete. Failed to save cache: {exc}")
