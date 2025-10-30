"""Postprocessing panel controller for FlowCol UI - sliders, color, and region properties."""

from typing import Optional, TYPE_CHECKING

from flowcol import defaults
from flowcol.app import actions

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class PostprocessingPanel:
    """Controller for postprocessing sliders, color controls, and region properties."""

    def __init__(self, app: "FlowColApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main FlowColApp instance
        """
        self.app = app

        # Widget IDs for postprocessing sliders
        self.postprocess_downsample_slider_id: Optional[int] = None
        self.postprocess_clip_slider_id: Optional[int] = None
        self.postprocess_brightness_slider_id: Optional[int] = None
        self.postprocess_contrast_slider_id: Optional[int] = None
        self.postprocess_gamma_slider_id: Optional[int] = None

        # Widget IDs for color controls
        self.color_enabled_checkbox_id: Optional[int] = None

        # Widget IDs for region properties
        self.surface_enabled_checkbox_id: Optional[int] = None
        self.interior_enabled_checkbox_id: Optional[int] = None
        self.smear_enabled_checkbox_id: Optional[int] = None
        self.smear_sigma_slider_id: Optional[int] = None

    def build_postprocessing_ui(self, parent, palette_colormaps: dict) -> None:
        """Build postprocessing sliders, color controls, and region properties UI.

        Args:
            parent: Parent widget ID to add postprocessing widgets to
            palette_colormaps: Dict mapping palette names to colormap tags
        """
        if dpg is None:
            return

        dpg.add_text("Post-processing", parent=parent)
        dpg.add_spacer(height=10, parent=parent)

        self.postprocess_downsample_slider_id = dpg.add_slider_float(
            label="Downsampling Blur",
            default_value=self.app.state.display_settings.downsample_sigma,
            min_value=0.0,
            max_value=2.0,
            format="%.2f",
            callback=self.on_downsample_slider,
            width=200,
            parent=parent,
        )

        self.postprocess_clip_slider_id = dpg.add_slider_float(
            label="Clip %",
            default_value=self.app.state.display_settings.clip_percent,
            min_value=0.0,
            max_value=defaults.MAX_CLIP_PERCENT,
            format="%.2f%%",
            callback=self.on_clip_slider,
            width=200,
            parent=parent,
        )

        self.postprocess_brightness_slider_id = dpg.add_slider_float(
            label="Brightness",
            default_value=self.app.state.display_settings.brightness,
            min_value=-0.5,
            max_value=0.5,
            format="%.2f",
            callback=self.on_brightness_slider,
            width=200,
            parent=parent,
        )

        self.postprocess_contrast_slider_id = dpg.add_slider_float(
            label="Contrast",
            default_value=self.app.state.display_settings.contrast,
            min_value=0.5,
            max_value=2.0,
            format="%.2f",
            callback=self.on_contrast_slider,
            width=200,
            parent=parent,
        )

        self.postprocess_gamma_slider_id = dpg.add_slider_float(
            label="Gamma",
            default_value=self.app.state.display_settings.gamma,
            min_value=0.3,
            max_value=3.0,
            format="%.2f",
            callback=self.on_gamma_slider,
            width=200,
            parent=parent,
        )

        dpg.add_spacer(height=10, parent=parent)
        dpg.add_separator(parent=parent)
        dpg.add_text("Colorization", parent=parent)
        dpg.add_spacer(height=10, parent=parent)

        self.color_enabled_checkbox_id = dpg.add_checkbox(
            label="Enable Color",
            default_value=self.app.state.display_settings.color_enabled,
            callback=self.on_color_enabled,
            parent=parent,
        )

        # Build global palette selection UI
        self._build_global_palette_ui(parent, palette_colormaps)

        dpg.add_spacer(height=10, parent=parent)
        dpg.add_separator(parent=parent)

        # Build region properties UI
        self._build_region_properties_ui(parent, palette_colormaps)

    def _build_global_palette_ui(self, parent, palette_colormaps: dict) -> None:
        """Build global palette selection UI with popup menu.

        Args:
            parent: Parent widget ID
            palette_colormaps: Dict mapping palette names to colormap tags
        """
        if dpg is None:
            return

        from flowcol.render import list_color_palettes
        palette_names = list(list_color_palettes())

        # Global palette selection with popup menu
        dpg.add_text("Global Palette", parent=parent)
        global_palette_button = dpg.add_button(
            label="Choose Palette...",
            width=200,
            tag="global_palette_button",
            parent=parent,
        )
        dpg.add_text(
            f"Current: {self.app.state.display_settings.palette}",
            tag="global_palette_current_text",
            parent=parent,
        )

        # Popup menu for global palette selection
        with dpg.popup(global_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="global_palette_popup"):
            dpg.add_text("Select a palette:")
            dpg.add_separator()
            with dpg.child_window(width=380, height=300):
                for palette_name in palette_names:
                    colormap_tag = palette_colormaps[palette_name]
                    btn = dpg.add_colormap_button(
                        label=palette_name,
                        width=350,
                        height=25,
                        callback=self.on_global_palette_button,
                        user_data=palette_name,
                        tag=f"global_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                    )
                    dpg.bind_colormap(btn, colormap_tag)

    def _build_region_properties_ui(self, parent, palette_colormaps: dict) -> None:
        """Build region properties UI (surface/interior palettes + smear).

        Args:
            parent: Parent widget ID
            palette_colormaps: Dict mapping palette names to colormap tags
        """
        if dpg is None:
            return

        from flowcol.render import list_color_palettes
        palette_names = list(list_color_palettes())

        # Region properties (shown when conductor selected in render mode)
        with dpg.collapsing_header(label="Region Properties", default_open=True, tag="region_properties_header", parent=parent):
            dpg.add_text("Select a conductor region to customize", tag="region_hint_text")

            dpg.add_spacer(height=5)
            dpg.add_text("Surface (Field Lines)", tag="surface_label")
            self.surface_enabled_checkbox_id = dpg.add_checkbox(
                label="Enable Custom Palette",
                callback=self.on_surface_enabled,
                tag="surface_enabled_checkbox",
            )
            # Surface palette popup
            surface_palette_button = dpg.add_button(
                label="Choose Surface Palette...",
                width=200,
                tag="surface_palette_button",
            )
            dpg.add_text("Current: None", tag="surface_palette_current_text")

            with dpg.popup(surface_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="surface_palette_popup"):
                dpg.add_text("Select surface palette:")
                dpg.add_separator()
                with dpg.child_window(width=380, height=250):
                    for palette_name in palette_names:
                        colormap_tag = palette_colormaps[palette_name]
                        btn = dpg.add_colormap_button(
                            label=palette_name,
                            width=350,
                            height=25,
                            callback=self.on_surface_palette_button,
                            user_data=palette_name,
                            tag=f"surface_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                        )
                        dpg.bind_colormap(btn, colormap_tag)

            dpg.add_spacer(height=10)
            dpg.add_text("Interior (Hollow Region)", tag="interior_label")
            self.interior_enabled_checkbox_id = dpg.add_checkbox(
                label="Enable Custom Palette",
                callback=self.on_interior_enabled,
                tag="interior_enabled_checkbox",
            )
            # Interior palette popup
            interior_palette_button = dpg.add_button(
                label="Choose Interior Palette...",
                width=200,
                tag="interior_palette_button",
            )
            dpg.add_text("Current: None", tag="interior_palette_current_text")

            with dpg.popup(interior_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="interior_palette_popup"):
                dpg.add_text("Select interior palette:")
                dpg.add_separator()
                with dpg.child_window(width=380, height=250):
                    for palette_name in palette_names:
                        colormap_tag = palette_colormaps[palette_name]
                        btn = dpg.add_colormap_button(
                            label=palette_name,
                            width=350,
                            height=25,
                            callback=self.on_interior_palette_button,
                            user_data=palette_name,
                            tag=f"interior_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                        )
                        dpg.bind_colormap(btn, colormap_tag)

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_text("Interior Smear")
            self.smear_enabled_checkbox_id = dpg.add_checkbox(
                label="Enable Interior Smear",
                callback=self.on_smear_enabled,
                tag="smear_enabled_checkbox",
            )
            self.smear_sigma_slider_id = dpg.add_slider_float(
                label="Blur Sigma",
                min_value=0.1,
                max_value=10.0,
                format="%.1f px",
                callback=self.on_smear_sigma,
                tag="smear_sigma_slider",
                width=200,
            )

    def update_region_properties_panel(self) -> None:
        """Update region properties panel based on current selection."""
        if dpg is None:
            return

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                settings = self.app.state.conductor_color_settings.get(selected.id)
                if settings:
                    # Update checkboxes only (colormap buttons are always visible)
                    dpg.set_value("surface_enabled_checkbox", settings.surface.enabled)
                    dpg.set_value("interior_enabled_checkbox", settings.interior.enabled)

                # Update smear controls
                dpg.set_value("smear_enabled_checkbox", selected.smear_enabled)
                dpg.set_value("smear_sigma_slider", selected.smear_sigma)
                # Show/hide slider based on smear enabled
                dpg.configure_item("smear_sigma_slider", show=selected.smear_enabled)

    # ------------------------------------------------------------------
    # Postprocessing slider callbacks
    # ------------------------------------------------------------------

    def on_downsample_slider(self, sender=None, app_data=None) -> None:
        """Handle downsampling blur sigma slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            self.app.state.display_settings.downsample_sigma = value

        # GPU is fast enough for real-time updates - no debouncing needed!
        self.app._apply_postprocessing()

    def on_clip_slider(self, sender=None, app_data=None) -> None:
        """Handle clip percent slider change."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            self.app.state.display_settings.clip_percent = value
            self.app.state.invalidate_base_rgb()

        # Clip is display-only, just refresh texture
        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def on_brightness_slider(self, sender=None, app_data=None) -> None:
        """Handle brightness slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            self.app.state.display_settings.brightness = value
            self.app.state.invalidate_base_rgb()

        # GPU is fast enough for real-time updates - no debouncing needed!
        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def on_contrast_slider(self, sender=None, app_data=None) -> None:
        """Handle contrast slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            self.app.state.display_settings.contrast = value
            self.app.state.invalidate_base_rgb()

        # GPU is fast enough for real-time updates - no debouncing needed!
        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def on_gamma_slider(self, sender=None, app_data=None) -> None:
        """Handle gamma slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            self.app.state.display_settings.gamma = value
            self.app.state.invalidate_base_rgb()

        # GPU is fast enough for real-time updates - no debouncing needed!
        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    # ------------------------------------------------------------------
    # Color and palette callbacks
    # ------------------------------------------------------------------

    def on_color_enabled(self, sender=None, app_data=None) -> None:
        """Handle color enabled checkbox change."""
        if dpg is None:
            return

        with self.app.state_lock:
            actions.set_color_enabled(self.app.state, app_data)

        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def on_global_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle global colormap button click."""
        if dpg is None or user_data is None:
            return

        palette_name = user_data
        with self.app.state_lock:
            actions.set_palette(self.app.state, palette_name)

        # Update current palette display
        dpg.set_value("global_palette_current_text", f"Current: {palette_name}")
        dpg.configure_item("global_palette_popup", show=False)

        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def on_surface_enabled(self, sender=None, app_data=None) -> None:
        """Handle surface custom palette checkbox."""
        if dpg is None:
            return

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_style_enabled(self.app.state, selected.id, "surface", app_data)

        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def on_surface_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle surface colormap button click."""
        if dpg is None or user_data is None:
            return

        palette_name = user_data
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_palette(self.app.state, selected.id, "surface", palette_name)

        # Update current palette display
        dpg.set_value("surface_palette_current_text", f"Current: {palette_name}")
        dpg.configure_item("surface_palette_popup", show=False)

        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def on_interior_enabled(self, sender=None, app_data=None) -> None:
        """Handle interior custom color checkbox."""
        if dpg is None:
            return

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_style_enabled(self.app.state, selected.id, "interior", app_data)

        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def on_interior_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle interior colormap button click."""
        if dpg is None or user_data is None:
            return

        palette_name = user_data
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_palette(self.app.state, selected.id, "interior", palette_name)

        # Update current palette display
        dpg.set_value("interior_palette_current_text", f"Current: {palette_name}")
        dpg.configure_item("interior_palette_popup", show=False)

        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    # ------------------------------------------------------------------
    # Smear callbacks
    # ------------------------------------------------------------------

    def on_smear_enabled(self, sender=None, app_data=None) -> None:
        """Toggle interior smear for selected conductor region."""
        if dpg is None:
            return

        print(f"DEBUG on_smear_enabled: CALLED with app_data={app_data}")
        with self.app.state_lock:
            idx = self.app.state.selected_idx
            print(f"DEBUG on_smear_enabled: selected_idx={idx}, num_conductors={len(self.app.state.project.conductors)}")
            if idx >= 0 and idx < len(self.app.state.project.conductors):
                self.app.state.project.conductors[idx].smear_enabled = bool(app_data)
                print(f"DEBUG on_smear_enabled: Set conductor[{idx}].smear_enabled = {bool(app_data)}")
                # Verify all conductor smear states after update
                all_smear = [c.smear_enabled for c in self.app.state.project.conductors]
                print(f"DEBUG on_smear_enabled: All conductor smear_enabled states = {all_smear}")
            else:
                print(f"DEBUG on_smear_enabled: INVALID idx, not updating")

        self.update_region_properties_panel()
        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def on_smear_sigma(self, sender=None, app_data=None) -> None:
        """Adjust smear blur sigma for selected conductor."""
        if dpg is None:
            return

        with self.app.state_lock:
            idx = self.app.state.selected_idx
            if idx >= 0 and idx < len(self.app.state.project.conductors):
                self.app.state.project.conductors[idx].smear_sigma = float(app_data)

        self.app.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()
