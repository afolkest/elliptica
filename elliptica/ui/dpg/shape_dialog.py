"""Shape insertion dialog controller for Elliptica UI."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class ShapeDialogController:
    """Controller for geometric shape insertion dialog."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app
        self.dialog_id = None
        self.shape_combo_id = None
        self.params_group_id = None
        self.param_inputs = {}  # Map param_name -> input_id
        self.current_shape_key = None

    def show_dialog(self, sender=None, app_data=None) -> None:
        """Show the shape insertion dialog."""
        if dpg is None:
            return

        from elliptica.shapes import SHAPES

        # Create dialog if it doesn't exist
        if self.dialog_id is None or not dpg.does_item_exist(self.dialog_id):
            with dpg.window(
                label="Insert Geometric Shape",
                modal=True,
                show=False,
                width=400,
                height=300,
                pos=(300, 200),
                no_resize=True,
            ) as self.dialog_id:
                dpg.add_text("Select a shape and adjust its parameters:")
                dpg.add_spacer(height=10)

                # Shape type dropdown
                shape_names = [spec.name for spec in SHAPES.values()]
                shape_keys = list(SHAPES.keys())
                self.shape_combo_id = dpg.add_combo(
                    label="Shape Type",
                    items=shape_names,
                    default_value=shape_names[0],
                    callback=self._on_shape_changed,
                    user_data=shape_keys,
                    width=200,
                )

                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_spacer(height=10)

                # Dynamic parameter inputs container
                self.params_group_id = dpg.add_group()

                dpg.add_spacer(height=20)

                # Buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Insert", callback=self._on_insert, width=100)
                    dpg.add_button(label="Cancel", callback=self._on_cancel, width=100)

        # Reset to first shape and show dialog
        if self.shape_combo_id is not None:
            shape_keys = list(SHAPES.keys())
            dpg.set_value(self.shape_combo_id, SHAPES[shape_keys[0]].name)
            self._on_shape_changed(self.shape_combo_id, SHAPES[shape_keys[0]].name, shape_keys)

        dpg.show_item(self.dialog_id)
        dpg.focus_item(self.dialog_id)

    def _on_shape_changed(self, sender, app_data, user_data) -> None:
        """Rebuild parameter inputs when shape selection changes."""
        if dpg is None or self.params_group_id is None:
            return

        from elliptica.shapes import SHAPES

        # Find which shape was selected
        shape_keys = user_data
        selected_name = app_data
        shape_key = None
        for key in shape_keys:
            if SHAPES[key].name == selected_name:
                shape_key = key
                break

        if shape_key is None:
            return

        self.current_shape_key = shape_key
        spec = SHAPES[shape_key]

        # Clear existing parameter inputs
        dpg.delete_item(self.params_group_id, children_only=True)
        self.param_inputs.clear()

        # Create inputs for each parameter
        for param in spec.params:
            if param.type == int:
                input_id = dpg.add_slider_int(
                    label=param.name.replace("_", " ").title(),
                    default_value=int(param.default),
                    min_value=int(param.min_val),
                    max_value=int(param.max_val),
                    width=200,
                    parent=self.params_group_id,
                )
            else:  # float
                input_id = dpg.add_slider_float(
                    label=param.name.replace("_", " ").title(),
                    default_value=param.default,
                    min_value=param.min_val,
                    max_value=param.max_val,
                    width=200,
                    parent=self.params_group_id,
                )
            self.param_inputs[param.name] = input_id

    def _on_insert(self, sender=None, app_data=None) -> None:
        """Insert the configured shape as a new conductor."""
        if dpg is None or self.current_shape_key is None:
            return

        from elliptica.shapes import SHAPES
        from elliptica.types import Conductor
        from elliptica.app.actions import add_conductor

        spec = SHAPES[self.current_shape_key]

        # Gather parameter values from UI
        param_values = {}
        for param in spec.params:
            input_id = self.param_inputs.get(param.name)
            if input_id is not None:
                param_values[param.name] = dpg.get_value(input_id)

        # Generate shape masks
        try:
            surface_mask, interior_mask = spec.generator(**param_values)
        except Exception as exc:
            dpg.set_value("status_text", f"Failed to generate shape: {exc}")
            return

        # Position shape centered on canvas with offset for visibility
        with self.app.state_lock:
            canvas_w, canvas_h = self.app.state.project.canvas_resolution
            mask_h, mask_w = surface_mask.shape
            num_conductors = len(self.app.state.project.conductors)
            offset = num_conductors * 30.0
            pos = (
                (canvas_w - mask_w) / 2.0 + offset,
                (canvas_h - mask_h) / 2.0 + offset,
            )

            conductor = Conductor(
                mask=surface_mask,
                voltage=0.5,
                position=pos,
                interior_mask=interior_mask,
            )
            add_conductor(self.app.state, conductor)
            self.app.state.view_mode = "edit"

        # Update UI
        self.app.canvas_renderer.mark_dirty()
        self.app._update_control_visibility()
        self.app.boundary_controls.rebuild_controls()
        self.app.boundary_controls.update_slider_labels()

        dpg.set_value("status_text", f"Inserted {spec.name}")
        dpg.hide_item(self.dialog_id)

    def _on_cancel(self, sender=None, app_data=None) -> None:
        """Close dialog without inserting."""
        if dpg is None or self.dialog_id is None:
            return
        dpg.hide_item(self.dialog_id)
