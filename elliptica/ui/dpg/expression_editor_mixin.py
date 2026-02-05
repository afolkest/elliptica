"""Expression editor functionality mixin for PostprocessingPanel.

This mixin provides the OKLCH expression-based color mapping editor UI,
allowing users to define custom color expressions using field variables.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class ExpressionEditorMixin:
    """Mixin providing expression editor functionality for PostprocessingPanel.

    This mixin expects the following attributes to be present on the class:
        - app: EllipticaApp instance
        - color_mode: str ("palette" or "expressions")
        - palette_editor_active: bool (from PaletteEditorMixin)
        - palette_editor_dirty: bool (from PaletteEditorMixin)
        - palette_editor_persist_dirty: bool (from PaletteEditorMixin)
        - palette_editor_refresh_pending: bool (from PaletteEditorMixin)
        - _apply_palette_editor_refresh(): method (from PaletteEditorMixin)
        - _finalize_palette_editor_colormaps(): method (from PaletteEditorMixin)
        - _set_palette_editor_state(): method (from PaletteEditorMixin)
    """

    # Type hints for attributes expected from the main class
    app: "EllipticaApp"
    color_mode: str

    def _init_expression_editor_state(self) -> None:
        """Initialize expression editor-related instance variables.

        Call this from the main class's __init__ method.
        """
        # Color mode is initialized in the main class since it's shared
        pass  # No additional state needed currently

    def _build_expression_editor_ui(self, parent) -> None:
        """Build the expression editor UI for OKLCH color mapping.

        Args:
            parent: Parent widget ID
        """
        if dpg is None:
            return

        from elliptica.colorspace import (
            list_presets,
            AVAILABLE_VARIABLES,
            AVAILABLE_FUNCTIONS,
        )

        preset_names = list_presets()

        # Brief explanation
        HELP_COLOR = (140, 140, 140)
        dpg.add_text(
            "Map field data to color using math expressions.",
            color=HELP_COLOR,
            parent=parent,
            wrap=280,
        )
        dpg.add_text(
            "Use variables like 'mag' (field magnitude) and 'angle'.",
            color=HELP_COLOR,
            parent=parent,
            wrap=280,
        )
        dpg.add_spacer(height=8, parent=parent)

        # Preset selector
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_text("Preset:")
            dpg.add_combo(
                items=preset_names,
                default_value=preset_names[0] if preset_names else "",
                width=-1,  # Fill available width
                callback=self.on_expression_preset_change,
                tag="expr_preset_combo",
            )

        dpg.add_spacer(height=10, parent=parent)

        # L expression - add tooltip
        l_label = dpg.add_text("Lightness (L)  [0-1]", parent=parent)
        with dpg.tooltip(l_label):
            dpg.add_text("Controls brightness. 0 = black, 1 = white.")
        dpg.add_input_text(
            default_value="clipnorm(lic, 0.5, 99.5)",
            width=-1,  # Fill available width
            height=45,
            multiline=True,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_L_input",
            parent=parent,
        )

        dpg.add_spacer(height=6, parent=parent)

        # C expression - add tooltip
        c_label = dpg.add_text("Chroma (C)  [0-0.4]", parent=parent)
        with dpg.tooltip(c_label):
            dpg.add_text("Color intensity/saturation. 0 = grayscale.")
        dpg.add_input_text(
            default_value="0",
            width=-1,  # Fill available width
            height=45,
            multiline=True,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_C_input",
            parent=parent,
        )

        dpg.add_spacer(height=6, parent=parent)

        # H expression - add tooltip
        h_label = dpg.add_text("Hue (H)  [0-360]", parent=parent)
        with dpg.tooltip(h_label):
            dpg.add_text("Color hue angle. 0=red, 120=green, 240=blue.")
        dpg.add_input_text(
            default_value="0",
            width=-1,  # Fill available width
            height=45,
            multiline=True,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_H_input",
            parent=parent,
        )

        # Error display
        dpg.add_text(
            "",
            color=(255, 100, 100),
            tag="expr_error_text",
            parent=parent,
            wrap=-1,  # Wrap to available width
        )

        dpg.add_spacer(height=4, parent=parent)

        # Reference section (collapsible)
        with dpg.collapsing_header(
            label="Expression Reference", default_open=False, parent=parent
        ):
            dpg.add_text("Variables:", color=(150, 200, 255))
            for var_name, var_desc in AVAILABLE_VARIABLES:
                dpg.add_text(f"  {var_name}", color=(200, 200, 200))
                dpg.add_text(f"    {var_desc}", color=(150, 150, 150), wrap=260)

            # PDE-specific variables (dynamic based on active PDE)
            dpg.add_spacer(height=4)
            with dpg.group(tag="pde_specific_vars_group"):
                self._update_pde_specific_vars_display()

            dpg.add_spacer(height=8)
            dpg.add_text("Functions:", color=(150, 200, 255))
            for func_sig, func_desc in AVAILABLE_FUNCTIONS:
                dpg.add_text(f"  {func_sig}", color=(200, 200, 200))
                dpg.add_text(f"    {func_desc}", color=(150, 150, 150), wrap=260)

        # Load first preset
        if preset_names:
            self._load_expression_preset(preset_names[0])

    def _update_pde_specific_vars_display(self) -> None:
        """Update the PDE-specific variables display based on active PDE."""
        if dpg is None:
            return

        # Early exit if the group doesn't exist yet (expression editor not built)
        if not dpg.does_item_exist("pde_specific_vars_group"):
            return

        from elliptica.colorspace import PDE_SPECIFIC_VARIABLES

        # Clear existing content
        dpg.delete_item("pde_specific_vars_group", children_only=True)

        # Get active PDE type
        pde_type = (
            self.app.state.project.pde_type if self.app.state.project else "poisson"
        )
        pde_vars = PDE_SPECIFIC_VARIABLES.get(pde_type, [])

        if pde_vars:
            dpg.add_text(
                f"  ({pde_type}):",
                color=(150, 180, 150),
                parent="pde_specific_vars_group",
            )
            for var_name, var_desc in pde_vars:
                dpg.add_text(
                    f"    {var_name}",
                    color=(180, 200, 180),
                    parent="pde_specific_vars_group",
                )
                dpg.add_text(
                    f"      {var_desc}",
                    color=(140, 150, 140),
                    wrap=250,
                    parent="pde_specific_vars_group",
                )

    def _load_expression_preset(self, preset_name: str) -> None:
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

    # ------------------------------------------------------------------
    # Expression editor callbacks
    # ------------------------------------------------------------------

    def on_color_mode_change(self, sender=None, app_data=None) -> None:
        """Handle color mode toggle (Palette / Expressions)."""
        if dpg is None:
            return

        from elliptica.app.state_manager import StateKey

        mode = app_data  # "Palette" or "Expressions"
        self.color_mode = "palette" if mode == "Palette" else "expressions"

        # Show/hide the appropriate UI groups
        dpg.configure_item("palette_mode_group", show=(self.color_mode == "palette"))
        dpg.configure_item(
            "expressions_mode_group", show=(self.color_mode == "expressions")
        )

        if self.color_mode != "palette" and self.palette_editor_active:
            if self.palette_editor_dirty or self.palette_editor_persist_dirty:
                self._apply_palette_editor_refresh(persist=True)
            self._finalize_palette_editor_colormaps()
            self._set_palette_editor_state(False)
            self.palette_editor_refresh_pending = False

        # Update color_config based on mode
        if self.color_mode == "palette":
            self.app.state_manager.update(StateKey.COLOR_CONFIG, None)
        else:
            self._update_color_config_from_expressions()

    def on_expression_preset_change(self, sender=None, app_data=None) -> None:
        """Handle preset selection change."""
        if dpg is None or app_data is None:
            return

        self._load_expression_preset(app_data)

    def on_expression_change(self, sender=None, app_data=None) -> None:
        """Handle expression text change (debounced)."""
        if dpg is None:
            return
        self._update_color_config_from_expressions(debounce=0.3)

    def _update_color_config_from_expressions(self, *, debounce: float = 0.0) -> None:
        """Build ColorConfig from current expression inputs and update via StateManager.

        Validation runs immediately (instant error feedback). Only valid configs
        enter the StateManager update path. The refresh signal is deferred when
        ``debounce > 0``.
        """
        if dpg is None:
            return

        # Only update if in expressions mode
        if self.color_mode != "expressions":
            return

        from elliptica.app.state_manager import StateKey
        from elliptica.colorspace import ColorConfig, ColorMapping
        from elliptica.expr import ExprError

        L_expr = dpg.get_value("expr_L_input").strip()
        C_expr = dpg.get_value("expr_C_input").strip()
        H_expr = dpg.get_value("expr_H_input").strip()

        try:
            config = ColorConfig(
                global_mapping=ColorMapping(L=L_expr, C=C_expr, H=H_expr),
            )
        except ExprError as e:
            dpg.set_value("expr_error_text", f"Error: {e}")
            return

        dpg.set_value("expr_error_text", "")
        self.app.state_manager.update(StateKey.COLOR_CONFIG, config, debounce=debounce)
