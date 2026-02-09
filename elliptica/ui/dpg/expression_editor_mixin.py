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


HELP_COLOR = (140, 140, 140)
SECTION_COLOR = (150, 200, 255)
ITEM_COLOR = (200, 200, 200)
MUTED_COLOR = (150, 150, 150)

FUNCTION_ORDER: tuple[str, ...] = (
    "clipnorm",
    "pclip",
    "normalize",
    "smoothstep",
    "lerp",
    "clamp",
    "atan2",
    "sin",
    "cos",
    "tan",
    "abs",
    "sqrt",
    "pow",
    "log",
    "log10",
    "exp",
    "min",
    "max",
    "mean",
    "std",
)

FUNCTION_SIGNATURES: dict[str, str] = {
    "clipnorm": "clipnorm(x, lo, hi)",
    "pclip": "pclip(x, lo, hi)",
    "normalize": "normalize(x)",
    "min": "min(x)",
    "max": "max(x)",
    "mean": "mean(x)",
    "std": "std(x)",
    "smoothstep": "smoothstep(lo, hi, x)",
    "lerp": "lerp(a, b, t)",
    "clamp": "clamp(x, lo, hi)",
    "abs": "abs(x)",
    "sqrt": "sqrt(x)",
    "pow": "pow(x, y)",
    "exp": "exp(x)",
    "log": "log(x)",
    "log10": "log10(x)",
    "sin": "sin(x)",
    "cos": "cos(x)",
    "tan": "tan(x)",
    "atan2": "atan2(y, x)",
}

FUNCTION_DESCRIPTIONS: dict[str, str] = {
    "clipnorm": "Percentile clip + normalize to [0,1].",
    "pclip": "Percentile clip without normalization.",
    "normalize": "Normalize by global min/max to [0,1].",
    "min": "Global minimum.",
    "max": "Global maximum.",
    "mean": "Global mean.",
    "std": "Global standard deviation.",
    "smoothstep": "Smooth transition from lo to hi.",
    "lerp": "Linear interpolation.",
    "clamp": "Clamp to [lo, hi].",
    "abs": "Absolute value.",
    "sqrt": "Square root.",
    "pow": "Power.",
    "exp": "Exponential (e^x).",
    "log": "Natural logarithm.",
    "log10": "Base-10 logarithm.",
    "sin": "Sine (radians).",
    "cos": "Cosine (radians).",
    "tan": "Tangent (radians).",
    "atan2": "Signed angle from vector components.",
}

SOLUTION_FIELD_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "dirichlet_mask": "Constraint mask from solver output (1 inside constrained region).",
}


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
        self._expression_reference_signatures: dict[str, tuple[str, tuple[str, ...]]] = {}

    def _build_expression_editor_ui(self, parent) -> None:
        """Build the expression editor UI for OKLCH color mapping.

        Args:
            parent: Parent widget ID
        """
        if dpg is None:
            return

        from elliptica.colorspace import list_presets

        preset_names = list_presets()

        dpg.add_text(
            "Build color from equations for Lightness (L), Chroma (C), and Hue (H).",
            color=HELP_COLOR,
            parent=parent,
            wrap=280,
        )
        dpg.add_text(
            "Variables depend on the active equation and current solver output.",
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

        dpg.add_text(
            "",
            color=MUTED_COLOR,
            parent=parent,
            wrap=280,
            tag="expr_preset_description",
        )

        dpg.add_spacer(height=10, parent=parent)

        l_label = dpg.add_text("Lightness (L)  [0-1]", parent=parent)
        with dpg.tooltip(l_label):
            dpg.add_text("Controls brightness. 0 = black, 1 = white.")
        dpg.add_input_text(
            default_value="clipnorm(lic, 0.5, 99.5)",
            width=-1,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_L_input",
            parent=parent,
            hint="e.g. clipnorm(lic, 0.5, 99.5)",
        )

        dpg.add_spacer(height=6, parent=parent)

        c_label = dpg.add_text("Chroma (C)  [0-0.4]", parent=parent)
        with dpg.tooltip(c_label):
            dpg.add_text("Color intensity/saturation. 0 = grayscale.")
        dpg.add_input_text(
            default_value="0",
            width=-1,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_C_input",
            parent=parent,
            hint="e.g. 0.14 * clipnorm(mag, 1, 99)",
        )

        dpg.add_spacer(height=6, parent=parent)

        h_label = dpg.add_text("Hue (H)  [0-360]", parent=parent)
        with dpg.tooltip(h_label):
            dpg.add_text("Color hue angle. 0=red, 120=green, 240=blue.")
        dpg.add_input_text(
            default_value="0",
            width=-1,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_H_input",
            parent=parent,
            hint="e.g. 180 + 180 * atan2(ey, ex) / pi",
        )

        dpg.add_text(
            "",
            color=(255, 100, 100),
            tag="expr_error_text",
            parent=parent,
            wrap=-1,
        )

        dpg.add_spacer(height=4, parent=parent)
        self._build_expression_reference_ui(
            "expr_reference",
            parent=parent,
            details_label="Reference Details",
            include_details=True,
        )

        if preset_names:
            self._load_expression_preset(preset_names[0])
        else:
            self._refresh_expression_reference_ui("expr_reference")

    def _build_lightness_expression_help_ui(self, parent) -> None:
        """Build compact expression reference for palette lightness expression."""
        if dpg is None:
            return

        with dpg.collapsing_header(
            label="Lightness Expression Help",
            default_open=False,
            parent=parent,
        ):
            dpg.add_text(
                "This expression uses the same language as Expressions mode and "
                "multiplies lightness only.",
                color=HELP_COLOR,
                wrap=280,
            )
            dpg.add_spacer(height=4)
            self._build_expression_reference_ui(
                "lightness_expr_reference",
                parent=None,
                details_label="",
                include_details=False,
            )
            self._refresh_expression_reference_ui("lightness_expr_reference")

    def _build_expression_reference_ui(
        self,
        tag_prefix: str,
        *,
        parent,
        details_label: str,
        include_details: bool,
    ) -> None:
        """Build reusable expression reference block."""
        if dpg is None:
            return

        self._expression_reference_signatures.pop(tag_prefix, None)

        dpg.add_text(
            "",
            color=HELP_COLOR,
            wrap=300,
            tag=f"{tag_prefix}_availability_text",
        )
        dpg.add_text("", color=MUTED_COLOR, wrap=300, tag=f"{tag_prefix}_advanced_text")
        dpg.add_text("", color=HELP_COLOR, wrap=300, tag=f"{tag_prefix}_functions_text")

        dpg.add_spacer(height=4)
        dpg.add_text("Constants: pi, e, tau", color=MUTED_COLOR, wrap=300)
        dpg.add_text(
            "Operators: + - * / ** and parentheses ()",
            color=MUTED_COLOR,
            wrap=300,
        )

        if include_details:
            with dpg.collapsing_header(
                label=details_label,
                default_open=False,
                parent=parent,
            ):
                dpg.add_text("Variables", color=SECTION_COLOR)
                dpg.add_group(tag=f"{tag_prefix}_vars_group")
                dpg.add_spacer(height=4)
                dpg.add_text("Functions", color=SECTION_COLOR)
                dpg.add_group(tag=f"{tag_prefix}_funcs_group")

    def _get_active_pde_type(self) -> str:
        """Return active PDE type string."""
        with self.app.state_lock:
            project = self.app.state.project
            if project is None:
                return "poisson"
            return project.pde_type or "poisson"

    def _get_solution_field_names(self) -> list[str]:
        """Return sorted solver field names from latest render, if available."""
        with self.app.state_lock:
            cache = self.app.state.render_cache
            if (
                cache is None
                or cache.result is None
                or cache.result.solution is None
            ):
                return []
            return sorted(cache.result.solution.keys())

    def _describe_solver_field(self, field_name: str, pde_type: str) -> str:
        """Describe dynamic solver fields that are not in the static reference."""
        if field_name in SOLUTION_FIELD_DESCRIPTION_OVERRIDES:
            return SOLUTION_FIELD_DESCRIPTION_OVERRIDES[field_name]
        if field_name.endswith("_mask"):
            return "Mask output from solver (0/1 values)."
        return f"Additional output field from '{pde_type}' solver."

    def _collect_available_variable_rows(
        self,
    ) -> tuple[str, bool, list[tuple[str, str]], list[tuple[str, str]], list[str]]:
        """Collect currently available variables for quick/reference views."""
        from elliptica.colorspace import AVAILABLE_VARIABLES, PDE_SPECIFIC_VARIABLES

        pde_type = self._get_active_pde_type()
        base_desc = dict(AVAILABLE_VARIABLES)
        pde_desc = dict(PDE_SPECIFIC_VARIABLES.get(pde_type, []))

        with self.app.state_lock:
            cache = self.app.state.render_cache
            has_render = cache is not None and cache.result is not None
            has_vector = (
                has_render
                and cache.result.ex is not None
                and cache.result.ey is not None
            )

        available_names: list[str] = []
        if has_render:
            available_names.append("lic")
            if has_vector:
                available_names.extend(["mag", "ex", "ey"])
            for field_name in self._get_solution_field_names():
                if field_name not in available_names:
                    available_names.append(field_name)
        else:
            available_names = ["lic", "mag", "ex", "ey", *pde_desc.keys()]

        def describe(name: str) -> str:
            if name in base_desc:
                return base_desc[name]
            if name in pde_desc:
                return pde_desc[name]
            return self._describe_solver_field(name, pde_type)

        public_rows: list[tuple[str, str]] = []
        advanced_rows: list[tuple[str, str]] = []
        for name in available_names:
            row = (name, describe(name))
            if name.endswith("_mask"):
                advanced_rows.append(row)
            else:
                public_rows.append(row)

        expected_names = ["lic", "mag", "ex", "ey", *pde_desc.keys()]
        expected_names = list(dict.fromkeys(expected_names))
        return pde_type, has_render, public_rows, advanced_rows, expected_names

    def _get_function_rows(self) -> list[tuple[str, str, str]]:
        """Return function rows in stable display order."""
        from elliptica.expr import list_functions

        function_args = list_functions()

        ordered_names = [name for name in FUNCTION_ORDER if name in function_args]
        extras = sorted(name for name in function_args if name not in ordered_names)
        ordered_names.extend(extras)

        rows: list[tuple[str, str, str]] = []
        for name in ordered_names:
            signature = FUNCTION_SIGNATURES.get(name)
            if signature is None:
                argc = function_args[name]
                args = ", ".join(f"arg{i + 1}" for i in range(argc))
                signature = f"{name}({args})"
            rows.append((name, signature, FUNCTION_DESCRIPTIONS.get(name, "Built-in function.")))

        return rows

    def _refresh_expression_reference_ui(self, tag_prefix: str) -> None:
        """Refresh dynamic reference content with compact availability-first layout."""
        if dpg is None:
            return

        availability_tag = f"{tag_prefix}_availability_text"
        if not dpg.does_item_exist(availability_tag):
            return

        pde_type, has_render, public_rows, advanced_rows, expected_names = (
            self._collect_available_variable_rows()
        )
        function_rows = self._get_function_rows()

        public_names = [name for name, _ in public_rows]
        advanced_names = [name for name, _ in advanced_rows]
        function_names = [name for name, _, _ in function_rows]

        signature = (
            pde_type,
            has_render,
            tuple(public_names),
            tuple(advanced_names),
            tuple(function_names),
        )

        if has_render:
            available_text = ", ".join(public_names) if public_names else "none"
            dpg.set_value(
                availability_tag,
                f"Available now ({pde_type}): {available_text}",
            )
        else:
            dpg.set_value(
                availability_tag,
                f"Expected for {pde_type} (render to confirm): {', '.join(expected_names)}",
            )

        advanced_tag = f"{tag_prefix}_advanced_text"
        if dpg.does_item_exist(advanced_tag):
            if advanced_names:
                dpg.set_value(
                    advanced_tag,
                    f"Advanced solver fields: {', '.join(advanced_names)}",
                )
            else:
                dpg.set_value(advanced_tag, "")

        functions_tag = f"{tag_prefix}_functions_text"
        if dpg.does_item_exist(functions_tag):
            dpg.set_value(functions_tag, f"Functions: {', '.join(function_names)}")

        cached_signatures = getattr(self, "_expression_reference_signatures", {})
        if cached_signatures.get(tag_prefix) == signature:
            return

        cached_signatures[tag_prefix] = signature
        self._expression_reference_signatures = cached_signatures

        vars_group_tag = f"{tag_prefix}_vars_group"
        if dpg.does_item_exist(vars_group_tag):
            dpg.delete_item(vars_group_tag, children_only=True)
            if not has_render:
                dpg.add_text(
                    "Render once to confirm runtime-only fields.",
                    color=MUTED_COLOR,
                    parent=vars_group_tag,
                )
            for name, desc in public_rows:
                dpg.add_text(f"{name}  -  {desc}", color=ITEM_COLOR, parent=vars_group_tag)
            if advanced_rows:
                dpg.add_spacer(height=4, parent=vars_group_tag)
                dpg.add_text("Advanced solver fields:", color=MUTED_COLOR, parent=vars_group_tag)
                for name, desc in advanced_rows:
                    dpg.add_text(
                        f"{name}  -  {desc}",
                        color=MUTED_COLOR,
                        parent=vars_group_tag,
                    )

        funcs_group_tag = f"{tag_prefix}_funcs_group"
        if dpg.does_item_exist(funcs_group_tag):
            dpg.delete_item(funcs_group_tag, children_only=True)
            for _, signature_text, description in function_rows:
                dpg.add_text(
                    f"{signature_text}  -  {description}",
                    color=ITEM_COLOR,
                    parent=funcs_group_tag,
                )

    def _update_pde_specific_vars_display(self) -> None:
        """Compatibility method: refresh all expression reference blocks."""
        if dpg is None:
            return

        self._refresh_expression_reference_ui("expr_reference")
        self._refresh_expression_reference_ui("lightness_expr_reference")

    def _update_expression_preset_description(self, preset_name: str) -> None:
        """Set current preset description text."""
        if dpg is None or not dpg.does_item_exist("expr_preset_description"):
            return

        from elliptica.colorspace import get_preset

        preset = get_preset(preset_name)
        if preset is None:
            dpg.set_value("expr_preset_description", "")
            return

        if preset.description:
            dpg.set_value("expr_preset_description", f"{preset.name}: {preset.description}")
            return

        dpg.set_value("expr_preset_description", preset.name)

    def _load_expression_preset(self, preset_name: str) -> None:
        """Load a preset into the expression inputs."""
        if dpg is None:
            return

        from elliptica.colorspace import get_preset

        preset = get_preset(preset_name)
        if preset is None:
            return

        if dpg.does_item_exist("expr_preset_combo"):
            dpg.set_value("expr_preset_combo", preset_name)
        dpg.set_value("expr_L_input", preset.L)
        dpg.set_value("expr_C_input", preset.C)
        dpg.set_value("expr_H_input", preset.H)
        self._update_expression_preset_description(preset_name)

        dpg.set_value("expr_error_text", "")

        self._refresh_expression_reference_ui("expr_reference")
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
            self._refresh_expression_reference_ui("expr_reference")
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
        self._refresh_expression_reference_ui("expr_reference")

        try:
            config = ColorConfig(
                global_mapping=ColorMapping(L=L_expr, C=C_expr, H=H_expr),
            )
        except ExprError as e:
            dpg.set_value("expr_error_text", f"Error: {e}")
            return

        dpg.set_value("expr_error_text", "")
        self.app.state_manager.update(StateKey.COLOR_CONFIG, config, debounce=debounce)
