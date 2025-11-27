"""Postprocessing panel controller for Elliptica UI - sliders, color, and region properties."""

import time
from typing import Optional, TYPE_CHECKING

from elliptica import defaults
from elliptica.app import actions

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class PostprocessingPanel:
    """Controller for postprocessing sliders, color controls, and region properties."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app

        # Widget IDs for postprocessing sliders
        self.postprocess_clip_slider_id: Optional[int] = None
        self.postprocess_brightness_slider_id: Optional[int] = None
        self.postprocess_contrast_slider_id: Optional[int] = None
        self.postprocess_gamma_slider_id: Optional[int] = None

        # Widget IDs for region properties
        self.smear_enabled_checkbox_id: Optional[int] = None
        self.smear_sigma_slider_id: Optional[int] = None

        # Widget IDs for per-region postprocessing
        self.surface_separate_processing_checkbox_id: Optional[int] = None
        self.surface_brightness_slider_id: Optional[int] = None
        self.surface_contrast_slider_id: Optional[int] = None
        self.interior_separate_processing_checkbox_id: Optional[int] = None
        self.interior_brightness_slider_id: Optional[int] = None
        self.interior_contrast_slider_id: Optional[int] = None

        # Widget IDs for expression editor
        self.expr_L_input_id: Optional[int] = None
        self.expr_C_input_id: Optional[int] = None
        self.expr_H_input_id: Optional[int] = None
        self.expr_error_text_id: Optional[int] = None
        self.expr_preset_combo_id: Optional[int] = None

        # Color mode: "palette" or "expressions"
        self.color_mode: str = "palette"

        # Delete confirmation modal
        self.delete_confirmation_modal_id: Optional[int] = None
        self.pending_delete_palette: Optional[str] = None

        # Debouncing for expensive smear updates
        self.smear_pending_value: Optional[float] = None
        self.smear_last_update_time: float = 0.0
        self.smear_debounce_delay: float = 0.3  # 300ms delay

        # Debouncing for expensive clip% updates (percentile computation at high res)
        self.clip_pending_value: Optional[float] = None
        self.clip_last_update_time: float = 0.0
        self.clip_debounce_delay: float = 0.3  # 300ms delay

        # Debouncing for expression updates
        self.expr_pending_update: bool = False
        self.expr_last_update_time: float = 0.0
        self.expr_debounce_delay: float = 0.3  # 300ms delay

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

        dpg.add_spacer(height=10, parent=parent)
        dpg.add_separator(parent=parent)
        dpg.add_text("Colorization", parent=parent)
        dpg.add_spacer(height=10, parent=parent)

        # Mode toggle: Palette / Expressions
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_text("Mode:")
            dpg.add_radio_button(
                items=["Palette", "Expressions"],
                default_value="Palette",
                horizontal=True,
                callback=self.on_color_mode_change,
                tag="color_mode_radio",
            )

        dpg.add_spacer(height=10, parent=parent)

        # Palette mode container (shown by default)
        with dpg.group(tag="palette_mode_group", parent=parent):
            # Build global palette selection UI
            self._build_global_palette_ui("palette_mode_group", palette_colormaps)

            # Brightness/contrast/gamma only apply in palette mode
            dpg.add_spacer(height=10)
            self.postprocess_brightness_slider_id = dpg.add_slider_float(
                label="Brightness",
                default_value=self.app.state.display_settings.brightness,
                min_value=defaults.MIN_BRIGHTNESS,
                max_value=defaults.MAX_BRIGHTNESS,
                format="%.2f",
                callback=self.on_brightness_slider,
                width=200,
            )
            self.postprocess_contrast_slider_id = dpg.add_slider_float(
                label="Contrast",
                default_value=self.app.state.display_settings.contrast,
                min_value=defaults.MIN_CONTRAST,
                max_value=defaults.MAX_CONTRAST,
                format="%.2f",
                callback=self.on_contrast_slider,
                width=200,
            )
            self.postprocess_gamma_slider_id = dpg.add_slider_float(
                label="Gamma",
                default_value=self.app.state.display_settings.gamma,
                min_value=defaults.MIN_GAMMA,
                max_value=defaults.MAX_GAMMA,
                format="%.2f",
                callback=self.on_gamma_slider,
                width=200,
            )

        # Expressions mode container (hidden by default)
        with dpg.group(tag="expressions_mode_group", parent=parent, show=False):
            self._build_expression_editor_ui("expressions_mode_group")

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

        from elliptica.render import list_color_palettes
        palette_names = list(list_color_palettes())

        # Global palette selection with popup menu
        dpg.add_text("Global Palette", parent=parent)
        global_palette_button = dpg.add_button(
            label="Choose Palette...",
            width=200,
            tag="global_palette_button",
            parent=parent,
        )

        # Show current state (Grayscale or palette name)
        initial_text = self.app.state.display_settings.palette if self.app.state.display_settings.color_enabled else "Grayscale"
        dpg.add_text(
            f"Current: {initial_text}",
            tag="global_palette_current_text",
            parent=parent,
        )

        # Popup menu for global palette selection
        with dpg.popup(global_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="global_palette_popup"):
            dpg.add_text("Select a palette (right-click to delete):")
            dpg.add_separator()

            # Add "Grayscale (No Color)" option at top
            dpg.add_button(
                label="⊙ Grayscale (No Color)",
                width=350,
                height=30,
                callback=self.on_global_grayscale,
                tag="global_grayscale_btn",
            )
            dpg.add_separator()

            with dpg.child_window(width=380, height=300, tag="global_palette_scrolling_window"):
                for palette_name in palette_names:
                    colormap_tag = palette_colormaps[palette_name]

                    # Create a group for each palette entry (colormap button + delete button)
                    with dpg.group(horizontal=True):
                        btn = dpg.add_colormap_button(
                            label=palette_name,
                            width=310,
                            height=25,
                            callback=self.on_global_palette_button,
                            user_data=palette_name,
                            tag=f"global_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                        )
                        dpg.bind_colormap(btn, colormap_tag)

                        # Add delete button that opens separate modal
                        dpg.add_button(
                            label="X",
                            width=30,
                            height=25,
                            callback=lambda s, a, u: self._open_delete_confirmation(u),
                            user_data=palette_name
                        )

    def _build_expression_editor_ui(self, parent) -> None:
        """Build the expression editor UI for OKLCH color mapping.

        Args:
            parent: Parent widget ID
        """
        if dpg is None:
            return

        from elliptica.colorspace import list_presets, get_preset, AVAILABLE_VARIABLES, AVAILABLE_FUNCTIONS

        preset_names = list_presets()

        # Preset selector
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_text("Preset:")
            self.expr_preset_combo_id = dpg.add_combo(
                items=preset_names,
                default_value=preset_names[0] if preset_names else "",
                width=180,
                callback=self.on_expression_preset_change,
                tag="expr_preset_combo",
            )

        dpg.add_spacer(height=10, parent=parent)

        # L expression
        dpg.add_text("Lightness (L)  [0-1]", parent=parent)
        self.expr_L_input_id = dpg.add_input_text(
            default_value="clipnorm(lic, 0.5, 99.5)",
            width=280,
            height=50,
            multiline=True,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_L_input",
            parent=parent,
        )

        dpg.add_spacer(height=8, parent=parent)

        # C expression
        dpg.add_text("Chroma (C)  [0-0.4]", parent=parent)
        self.expr_C_input_id = dpg.add_input_text(
            default_value="0",
            width=280,
            height=50,
            multiline=True,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_C_input",
            parent=parent,
        )

        dpg.add_spacer(height=8, parent=parent)

        # H expression
        dpg.add_text("Hue (H)  [0-360 degrees]", parent=parent)
        self.expr_H_input_id = dpg.add_input_text(
            default_value="0",
            width=280,
            height=50,
            multiline=True,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_H_input",
            parent=parent,
        )

        dpg.add_spacer(height=8, parent=parent)

        # Error display
        self.expr_error_text_id = dpg.add_text(
            "",
            color=(255, 100, 100),
            tag="expr_error_text",
            parent=parent,
            wrap=280,
        )

        dpg.add_spacer(height=10, parent=parent)

        # Reference section (collapsible)
        with dpg.collapsing_header(label="Reference", default_open=False, parent=parent):
            dpg.add_text("Variables:", color=(150, 200, 255))
            for var_name, var_desc in AVAILABLE_VARIABLES:
                dpg.add_text(f"  {var_name}", color=(200, 200, 200))
                dpg.add_text(f"    {var_desc}", color=(150, 150, 150), wrap=260)

            dpg.add_spacer(height=8)
            dpg.add_text("Functions:", color=(150, 200, 255))
            for func_sig, func_desc in AVAILABLE_FUNCTIONS:
                dpg.add_text(f"  {func_sig}", color=(200, 200, 200))
                dpg.add_text(f"    {func_desc}", color=(150, 150, 150), wrap=260)

        # Load first preset
        if preset_names:
            self._load_preset(preset_names[0])

    def _load_preset(self, preset_name: str) -> None:
        """Load a preset into the expression inputs."""
        if dpg is None:
            return

        from elliptica.colorspace import get_preset

        preset = get_preset(preset_name)
        if preset is None:
            return

        dpg.set_value("expr_L_input", preset.L)
        dpg.set_value("expr_C_input", preset.C)
        dpg.set_value("expr_H_input", preset.H)

        # Clear any error
        dpg.set_value("expr_error_text", "")

        # Trigger update
        self._update_color_config_from_expressions()

    def _build_region_properties_ui(self, parent, palette_colormaps: dict) -> None:
        """Build region properties UI (surface/interior palettes + smear).

        Args:
            parent: Parent widget ID
            palette_colormaps: Dict mapping palette names to colormap tags
        """
        if dpg is None:
            return

        from elliptica.render import list_color_palettes
        palette_names = list(list_color_palettes())

        # Region properties (shown when conductor selected in render mode)
        with dpg.collapsing_header(label="Region Properties", default_open=True, tag="region_properties_header", parent=parent):
            dpg.add_text("Select a conductor region to customize", tag="region_hint_text")

            dpg.add_spacer(height=5)
            dpg.add_text("Conductor", tag="surface_label")
            # Surface palette selection (no checkbox - selecting palette enables it)
            surface_palette_button = dpg.add_button(
                label="Choose palette...",
                width=200,
                tag="surface_palette_button",
            )
            dpg.add_text("Current: Global", tag="surface_palette_current_text")

            with dpg.popup(surface_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="surface_palette_popup"):
                dpg.add_text("Select surface palette:")
                dpg.add_separator()

                # Add "Use Global Palette" option at top
                dpg.add_button(
                    label="⊙ Use Global Palette",
                    width=350,
                    height=30,
                    callback=self.on_surface_use_global,
                    tag="surface_use_global_btn",
                )
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

            # Separate postprocessing controls for surface
            dpg.add_spacer(height=5)
            self.surface_separate_processing_checkbox_id = dpg.add_checkbox(
                label="Separate processing",
                callback=self.on_surface_separate_processing,
                tag="surface_separate_processing_checkbox",
            )
            self.surface_brightness_slider_id = dpg.add_slider_float(
                label="Conductor brightness",
                default_value=defaults.DEFAULT_BRIGHTNESS,
                min_value=defaults.MIN_BRIGHTNESS,
                max_value=defaults.MAX_BRIGHTNESS,
                format="%.2f",
                callback=self.on_surface_brightness,
                tag="surface_brightness_slider",
                width=200,
                show=False,  # Hidden by default
            )
            self.surface_contrast_slider_id = dpg.add_slider_float(
                label="Conductor contrast",
                default_value=defaults.DEFAULT_CONTRAST,
                min_value=defaults.MIN_CONTRAST,
                max_value=defaults.MAX_CONTRAST,
                format="%.2f",
                callback=self.on_surface_contrast,
                tag="surface_contrast_slider",
                width=200,
                show=False,  # Hidden by default
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Conductor interior", tag="interior_label")
            # Interior palette selection (no checkbox - selecting palette enables it)
            interior_palette_button = dpg.add_button(
                label="Choose palette...",
                width=200,
                tag="interior_palette_button",
            )
            dpg.add_text("Current: Global", tag="interior_palette_current_text")

            with dpg.popup(interior_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="interior_palette_popup"):
                dpg.add_text("Select interior palette:")
                dpg.add_separator()

                # Add "Use Global Palette" option at top
                dpg.add_button(
                    label="⊙ Use Global Palette",
                    width=350,
                    height=30,
                    callback=self.on_interior_use_global,
                    tag="interior_use_global_btn",
                )
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

            # Separate postprocessing controls for interior
            dpg.add_spacer(height=5)
            self.interior_separate_processing_checkbox_id = dpg.add_checkbox(
                label="Separate processing",
                callback=self.on_interior_separate_processing,
                tag="interior_separate_processing_checkbox",
            )
            self.interior_brightness_slider_id = dpg.add_slider_float(
                label="Interior brightness",
                default_value=defaults.DEFAULT_BRIGHTNESS,
                min_value=defaults.MIN_BRIGHTNESS,
                max_value=defaults.MAX_BRIGHTNESS,
                format="%.2f",
                callback=self.on_interior_brightness,
                tag="interior_brightness_slider",
                width=200,
                show=False,  # Hidden by default
            )
            self.interior_contrast_slider_id = dpg.add_slider_float(
                label="Interior contrast",
                default_value=defaults.DEFAULT_CONTRAST,
                min_value=defaults.MIN_CONTRAST,
                max_value=defaults.MAX_CONTRAST,
                format="%.2f",
                callback=self.on_interior_contrast,
                tag="interior_contrast_slider",
                width=200,
                show=False,  # Hidden by default
            )

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_text("Conductor smear")
            self.smear_enabled_checkbox_id = dpg.add_checkbox(
                label="Enable smear",
                callback=self.on_smear_enabled,
                tag="smear_enabled_checkbox",
            )
            self.smear_sigma_slider_id = dpg.add_slider_float(
                label="Blur strength",
                min_value=defaults.MIN_SMEAR_SIGMA,
                max_value=defaults.MAX_SMEAR_SIGMA,
                format="%.4f",
                callback=self.on_smear_sigma,
                tag="smear_sigma_slider",
                width=200,
                clamped=True,
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
                    # Update palette current text based on enabled state
                    if settings.surface.enabled:
                        dpg.set_value("surface_palette_current_text", f"Current: {settings.surface.palette}")
                    else:
                        dpg.set_value("surface_palette_current_text", "Current: Global")

                    if settings.interior.enabled:
                        dpg.set_value("interior_palette_current_text", f"Current: {settings.interior.palette}")
                    else:
                        dpg.set_value("interior_palette_current_text", "Current: Global")

                    # Update per-region postprocessing controls
                    # Surface
                    surface_has_overrides = (settings.surface.brightness is not None or
                                            settings.surface.contrast is not None)
                    dpg.set_value("surface_separate_processing_checkbox", surface_has_overrides)
                    dpg.configure_item("surface_brightness_slider", show=surface_has_overrides)
                    dpg.configure_item("surface_contrast_slider", show=surface_has_overrides)
                    if surface_has_overrides:
                        brightness_val = settings.surface.brightness if settings.surface.brightness is not None else self.app.state.display_settings.brightness
                        contrast_val = settings.surface.contrast if settings.surface.contrast is not None else self.app.state.display_settings.contrast
                        dpg.set_value("surface_brightness_slider", brightness_val)
                        dpg.set_value("surface_contrast_slider", contrast_val)

                    # Interior
                    interior_has_overrides = (settings.interior.brightness is not None or
                                             settings.interior.contrast is not None)
                    dpg.set_value("interior_separate_processing_checkbox", interior_has_overrides)
                    dpg.configure_item("interior_brightness_slider", show=interior_has_overrides)
                    dpg.configure_item("interior_contrast_slider", show=interior_has_overrides)
                    if interior_has_overrides:
                        brightness_val = settings.interior.brightness if settings.interior.brightness is not None else self.app.state.display_settings.brightness
                        contrast_val = settings.interior.contrast if settings.interior.contrast is not None else self.app.state.display_settings.contrast
                        dpg.set_value("interior_brightness_slider", brightness_val)
                        dpg.set_value("interior_contrast_slider", contrast_val)

                # Update smear controls
                dpg.set_value("smear_enabled_checkbox", selected.smear_enabled)
                dpg.set_value("smear_sigma_slider", selected.smear_sigma)
                # Show/hide slider based on smear enabled
                dpg.configure_item("smear_sigma_slider", show=selected.smear_enabled)

    # ------------------------------------------------------------------
    # Postprocessing slider callbacks
    # ------------------------------------------------------------------

    def on_clip_slider(self, sender=None, app_data=None) -> None:
        """Handle clip percent slider change with debouncing (percentile computation is expensive at high res)."""
        if dpg is None:
            return

        value = float(app_data)

        # IMMEDIATELY update state so other refreshes use the correct value
        # (otherwise other sliders would trigger refresh with stale clip%)
        with self.app.state_lock:
            self.app.state.display_settings.clip_percent = value
            self.app.state.invalidate_base_rgb()

        # Mark pending to trigger refresh after debounce delay
        self.clip_pending_value = value

        # Record the time of THIS slider change (not the last render)
        # This ensures we wait 300ms after the LAST slider movement
        self.clip_last_update_time = time.time()

    def on_brightness_slider(self, sender=None, app_data=None) -> None:
        """Handle brightness slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            self.app.state.display_settings.brightness = value
            self.app.state.invalidate_base_rgb()

        # GPU is fast enough for real-time updates - no debouncing needed!
        self.app.display_pipeline.refresh_display()

    def on_contrast_slider(self, sender=None, app_data=None) -> None:
        """Handle contrast slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            self.app.state.display_settings.contrast = value
            self.app.state.invalidate_base_rgb()

        # GPU is fast enough for real-time updates - no debouncing needed!
        self.app.display_pipeline.refresh_display()

    def on_gamma_slider(self, sender=None, app_data=None) -> None:
        """Handle gamma slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            self.app.state.display_settings.gamma = value
            self.app.state.invalidate_base_rgb()

        # GPU is fast enough for real-time updates - no debouncing needed!
        self.app.display_pipeline.refresh_display()

    # ------------------------------------------------------------------
    # Color and palette callbacks
    # ------------------------------------------------------------------

    def on_global_grayscale(self, sender=None, app_data=None) -> None:
        """Handle global 'Grayscale (No Color)' button."""
        if dpg is None:
            return

        with self.app.state_lock:
            actions.set_color_enabled(self.app.state, False)

        # Update display
        dpg.set_value("global_palette_current_text", "Current: Grayscale")
        dpg.configure_item("global_palette_popup", show=False)

        self.app.display_pipeline.refresh_display()

    def on_global_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle global colormap button click."""
        if dpg is None or user_data is None:
            return

        palette_name = user_data
        with self.app.state_lock:
            # Auto-enable color when a palette is selected
            actions.set_color_enabled(self.app.state, True)
            actions.set_palette(self.app.state, palette_name)

        # Update current palette display
        dpg.set_value("global_palette_current_text", f"Current: {palette_name}")
        dpg.configure_item("global_palette_popup", show=False)

        self.app.display_pipeline.refresh_display()

    def _ensure_delete_confirmation_modal(self) -> None:
        """Create the delete confirmation modal if it doesn't exist."""
        if dpg is None or self.delete_confirmation_modal_id is not None:
            return

        with dpg.window(
            label="Delete Palette?",
            modal=True,
            show=False,
            tag="delete_palette_modal",
            no_resize=True,
            no_move=False,
            no_close=True,
            no_collapse=True,
            width=350,
            height=150,
            no_open_over_existing_popup=False,
        ) as modal:
            self.delete_confirmation_modal_id = modal

            dpg.add_text("", tag="delete_palette_modal_text")
            dpg.add_text("This cannot be undone.", color=(200, 200, 0))
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Delete",
                    width=75,
                    callback=self._confirm_delete_palette
                )
                dpg.add_button(
                    label="Cancel",
                    width=75,
                    callback=self._cancel_delete_palette
                )

    def _open_delete_confirmation(self, palette_name: str) -> None:
        """Open the delete confirmation modal for a specific palette."""
        if dpg is None:
            return

        self._ensure_delete_confirmation_modal()
        self.pending_delete_palette = palette_name

        dpg.set_value("delete_palette_modal_text", f"Delete '{palette_name}'?")

        # Center the modal on screen
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        modal_width = 350
        modal_height = 150
        pos_x = (viewport_width - modal_width) // 2
        pos_y = (viewport_height - modal_height) // 2

        dpg.configure_item(self.delete_confirmation_modal_id, pos=[pos_x, pos_y])
        dpg.configure_item(self.delete_confirmation_modal_id, show=True)

    def _cancel_delete_palette(self, sender=None, app_data=None) -> None:
        """Cancel palette deletion."""
        if dpg is None or self.delete_confirmation_modal_id is None:
            return
        dpg.configure_item(self.delete_confirmation_modal_id, show=False)
        self.pending_delete_palette = None

    def _confirm_delete_palette(self, sender=None, app_data=None, user_data=None) -> None:
        """Actually delete the palette after confirmation."""
        if dpg is None or self.pending_delete_palette is None:
            return

        from elliptica.render import delete_palette

        palette_name = self.pending_delete_palette
        delete_palette(palette_name)

        # Rebuild DPG colormaps to reflect deletion
        self.app.display_pipeline.texture_manager.rebuild_colormaps()

        # Rebuild the palette popup menu
        self._rebuild_palette_popup()

        # Close the confirmation modal
        dpg.configure_item(self.delete_confirmation_modal_id, show=False)
        self.pending_delete_palette = None

    def _rebuild_palette_popup(self) -> None:
        """Rebuild palette popup after deletion."""
        if dpg is None:
            return

        # Delete old scrolling window content
        if dpg.does_item_exist("global_palette_scrolling_window"):
            dpg.delete_item("global_palette_scrolling_window", children_only=True)

        # Rebuild colormap buttons
        from elliptica.render import list_color_palettes
        palette_names = list(list_color_palettes())

        for palette_name in palette_names:
            colormap_tag = self.app.display_pipeline.texture_manager.palette_colormaps.get(palette_name)
            if not colormap_tag:
                continue

            # Create a group for each palette entry (colormap button + delete button)
            with dpg.group(horizontal=True, parent="global_palette_scrolling_window"):
                btn = dpg.add_colormap_button(
                    label=palette_name,
                    width=310,
                    height=25,
                    callback=self.on_global_palette_button,
                    user_data=palette_name,
                    tag=f"global_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                )
                dpg.bind_colormap(btn, colormap_tag)

                # Add delete button that opens separate modal
                dpg.add_button(
                    label="X",
                    width=30,
                    height=25,
                    callback=lambda s, a, u: self._open_delete_confirmation(u),
                    user_data=palette_name
                )

    def on_surface_use_global(self, sender=None, app_data=None) -> None:
        """Handle surface 'Use Global Palette' button."""
        if dpg is None:
            return

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_style_enabled(self.app.state, selected.id, "surface", False)

        # Update display
        dpg.set_value("surface_palette_current_text", "Current: Global")
        dpg.configure_item("surface_palette_popup", show=False)

        self.app.display_pipeline.refresh_display()

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

        # Note: set_region_palette auto-enables the region
        self.app.display_pipeline.refresh_display()

    def on_interior_use_global(self, sender=None, app_data=None) -> None:
        """Handle interior 'Use Global Palette' button."""
        if dpg is None:
            return

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_style_enabled(self.app.state, selected.id, "interior", False)

        # Update display
        dpg.set_value("interior_palette_current_text", "Current: Global")
        dpg.configure_item("interior_palette_popup", show=False)

        self.app.display_pipeline.refresh_display()

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

        # Note: set_region_palette auto-enables the region
        self.app.display_pipeline.refresh_display()

    # ------------------------------------------------------------------
    # Per-region postprocessing callbacks
    # ------------------------------------------------------------------

    def on_surface_separate_processing(self, sender=None, app_data=None) -> None:
        """Toggle separate postprocessing for surface region."""
        if dpg is None:
            return

        is_enabled = bool(app_data)

        # Show/hide brightness and contrast sliders
        dpg.configure_item("surface_brightness_slider", show=is_enabled)
        dpg.configure_item("surface_contrast_slider", show=is_enabled)

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                if is_enabled:
                    # Initialize with global values
                    global_brightness = self.app.state.display_settings.brightness
                    global_contrast = self.app.state.display_settings.contrast
                    dpg.set_value("surface_brightness_slider", global_brightness)
                    dpg.set_value("surface_contrast_slider", global_contrast)
                    actions.set_region_brightness(self.app.state, selected.id, "surface", global_brightness)
                    actions.set_region_contrast(self.app.state, selected.id, "surface", global_contrast)
                else:
                    # Reset to None (inherit from global)
                    actions.set_region_brightness(self.app.state, selected.id, "surface", None)
                    actions.set_region_contrast(self.app.state, selected.id, "surface", None)

        self.app.display_pipeline.refresh_display()

    def on_surface_brightness(self, sender=None, app_data=None) -> None:
        """Handle surface brightness slider change."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_brightness(self.app.state, selected.id, "surface", value)

        self.app.display_pipeline.refresh_display()

    def on_surface_contrast(self, sender=None, app_data=None) -> None:
        """Handle surface contrast slider change."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_contrast(self.app.state, selected.id, "surface", value)

        self.app.display_pipeline.refresh_display()

    def on_interior_separate_processing(self, sender=None, app_data=None) -> None:
        """Toggle separate postprocessing for interior region."""
        if dpg is None:
            return

        is_enabled = bool(app_data)

        # Show/hide brightness and contrast sliders
        dpg.configure_item("interior_brightness_slider", show=is_enabled)
        dpg.configure_item("interior_contrast_slider", show=is_enabled)

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                if is_enabled:
                    # Initialize with global values
                    global_brightness = self.app.state.display_settings.brightness
                    global_contrast = self.app.state.display_settings.contrast
                    dpg.set_value("interior_brightness_slider", global_brightness)
                    dpg.set_value("interior_contrast_slider", global_contrast)
                    actions.set_region_brightness(self.app.state, selected.id, "interior", global_brightness)
                    actions.set_region_contrast(self.app.state, selected.id, "interior", global_contrast)
                else:
                    # Reset to None (inherit from global)
                    actions.set_region_brightness(self.app.state, selected.id, "interior", None)
                    actions.set_region_contrast(self.app.state, selected.id, "interior", None)

        self.app.display_pipeline.refresh_display()

    def on_interior_brightness(self, sender=None, app_data=None) -> None:
        """Handle interior brightness slider change."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_brightness(self.app.state, selected.id, "interior", value)

        self.app.display_pipeline.refresh_display()

    def on_interior_contrast(self, sender=None, app_data=None) -> None:
        """Handle interior contrast slider change."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_contrast(self.app.state, selected.id, "interior", value)

        self.app.display_pipeline.refresh_display()

    # ------------------------------------------------------------------
    # Smear callbacks
    # ------------------------------------------------------------------

    def on_smear_enabled(self, sender=None, app_data=None) -> None:
        """Toggle interior smear for selected conductor region."""
        if dpg is None:
            return

        with self.app.state_lock:
            idx = self.app.state.selected_idx
            if idx >= 0 and idx < len(self.app.state.project.conductors):
                self.app.state.project.conductors[idx].smear_enabled = bool(app_data)

        self.update_region_properties_panel()
        self.app.display_pipeline.refresh_display()

    def on_smear_sigma(self, sender=None, app_data=None) -> None:
        """Adjust smear blur sigma for selected conductor (debounced for performance)."""
        if dpg is None:
            return

        # Always update the value immediately (for UI responsiveness)
        with self.app.state_lock:
            idx = self.app.state.selected_idx
            if idx >= 0 and idx < len(self.app.state.project.conductors):
                self.app.state.project.conductors[idx].smear_sigma = float(app_data)

        # Mark that we have a pending update (defers expensive render refresh)
        self.smear_pending_value = float(app_data)

        # Record the time of THIS slider change (not the last render)
        # This ensures we wait 300ms after the LAST slider movement
        self.smear_last_update_time = time.time()

    def _apply_clip_update(self, value: float) -> None:
        """Apply clip percent refresh (state already updated in on_clip_slider)."""
        # State was already updated in on_clip_slider - just trigger the deferred refresh
        self.app.display_pipeline.refresh_display()

    def _apply_smear_update(self) -> None:
        """Apply pending smear update (called after debounce delay)."""
        if self.smear_pending_value is None:
            return

        self.smear_pending_value = None
        self.app.display_pipeline.refresh_display()

    def check_clip_debounce(self) -> None:
        """Check if clip update should be applied (called every frame)."""
        if self.clip_pending_value is None:
            return

        current_time = time.time()
        # Only apply if enough time has passed since the last slider movement
        if current_time - self.clip_last_update_time >= self.clip_debounce_delay:
            self._apply_clip_update(self.clip_pending_value)
            self.clip_last_update_time = current_time
            self.clip_pending_value = None

    def check_smear_debounce(self) -> None:
        """Check if smear update should be applied (called every frame)."""
        if self.smear_pending_value is None:
            return

        current_time = time.time()
        # Only apply if enough time has passed since the last slider movement
        if current_time - self.smear_last_update_time >= self.smear_debounce_delay:
            self._apply_smear_update()

    # ------------------------------------------------------------------
    # Expression editor callbacks
    # ------------------------------------------------------------------

    def on_color_mode_change(self, sender=None, app_data=None) -> None:
        """Handle color mode toggle (Palette / Expressions)."""
        if dpg is None:
            return

        mode = app_data  # "Palette" or "Expressions"
        self.color_mode = "palette" if mode == "Palette" else "expressions"

        # Show/hide the appropriate UI groups
        dpg.configure_item("palette_mode_group", show=(self.color_mode == "palette"))
        dpg.configure_item("expressions_mode_group", show=(self.color_mode == "expressions"))

        # Update color_config based on mode
        with self.app.state_lock:
            if self.color_mode == "palette":
                # Clear color_config to use legacy palette mode
                self.app.state.color_config = None
            else:
                # Build ColorConfig from current expressions
                self._update_color_config_from_expressions()

        self.app.display_pipeline.refresh_display()

    def on_expression_preset_change(self, sender=None, app_data=None) -> None:
        """Handle preset selection change."""
        if dpg is None or app_data is None:
            return

        self._load_preset(app_data)

    def on_expression_change(self, sender=None, app_data=None) -> None:
        """Handle expression text change (debounced)."""
        if dpg is None:
            return

        # Mark pending update and record time
        self.expr_pending_update = True
        self.expr_last_update_time = time.time()

    def check_expression_debounce(self) -> None:
        """Check if expression update should be applied (called every frame)."""
        if not self.expr_pending_update:
            return

        current_time = time.time()
        if current_time - self.expr_last_update_time >= self.expr_debounce_delay:
            self._update_color_config_from_expressions()
            self.expr_pending_update = False

    def _update_color_config_from_expressions(self) -> None:
        """Build ColorConfig from current expression inputs and update state."""
        if dpg is None:
            return

        # Only update if in expressions mode
        if self.color_mode != "expressions":
            return

        from elliptica.colorspace import ColorConfig, ColorMapping
        from elliptica.expr import ExprError

        L_expr = dpg.get_value("expr_L_input").strip()
        C_expr = dpg.get_value("expr_C_input").strip()
        H_expr = dpg.get_value("expr_H_input").strip()

        # Try to build ColorConfig
        try:
            config = ColorConfig(
                global_mapping=ColorMapping(L=L_expr, C=C_expr, H=H_expr),
            )

            # Success - clear error and update state
            dpg.set_value("expr_error_text", "")

            with self.app.state_lock:
                self.app.state.color_config = config

            self.app.display_pipeline.refresh_display()

        except ExprError as e:
            # Show error but don't update config
            dpg.set_value("expr_error_text", f"Error: {e}")
