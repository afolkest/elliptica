"""Boundary controls panel for FlowCol UI - generic parameter sliders."""

from typing import Optional, Dict, TYPE_CHECKING
from flowcol.app import actions
from flowcol.pde import PDERegistry

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class BoundaryControlsPanel:
    """Controller for generic boundary object controls (dynamic sliders)."""

    def __init__(self, app: "FlowColApp"):
        """Initialize controller with reference to main app."""
        self.app = app

        # Widget IDs
        self.container_id: Optional[int] = None
        # Map: conductor_idx -> param_name -> slider_id
        self.slider_ids: Dict[int, Dict[str, int]] = {}

    def build_container(self, parent) -> None:
        """Build controls container in edit controls panel."""
        if dpg is None:
            return

        dpg.add_text("Boundary Parameters", parent=parent)
        self.container_id = dpg.add_child_window(
            autosize_x=True,
            height=400,
            border=False,
            tag="boundary_controls_child",
            no_scroll_with_mouse=False,
            parent=parent,
        )

    def rebuild_controls(self) -> None:
        """Rebuild all sliders based on active PDE metadata."""
        if dpg is None or self.container_id is None:
            return

        dpg.delete_item(self.container_id, children_only=True)
        self.slider_ids.clear()

        with self.app.state_lock:
            conductors = list(self.app.state.project.conductors)
            selected_idx = self.app.state.selected_idx
        
        # Get active PDE definition to know what sliders to build
        pde = PDERegistry.get_active()
        params_meta = pde.boundary_params

        if not conductors:
            dpg.add_text("No boundaries loaded.", parent=self.container_id)
            return

        for idx, conductor in enumerate(conductors):
            label = f"Object {idx + 1}"
            if idx == selected_idx:
                label += " (selected)"
            
            with dpg.collapsing_header(label=label, default_open=True, parent=self.container_id):
                self.slider_ids[idx] = {}
                
                # Dynamic sliders for PDE parameters
                for param in params_meta:
                    current_val = conductor.params.get(param.name, param.default_value)
                    
                    slider_id = dpg.add_slider_float(
                        label=param.display_name,
                        default_value=float(current_val),
                        min_value=param.min_value,
                        max_value=param.max_value,
                        format="%.3f",
                        callback=self.on_param_slider,
                        user_data={"idx": idx, "param": param.name},
                    )
                    self.slider_ids[idx][param.name] = slider_id
                    
                    if param.description:
                        with dpg.tooltip(slider_id):
                            dpg.add_text(param.description)

                # Standard edge smoothing slider (always present)
                dpg.add_slider_float(
                    label="Edge Smooth",
                    default_value=float(conductor.edge_smooth_sigma),
                    min_value=0.0,
                    max_value=5.0,
                    format="%.1f px",
                    callback=self.on_edge_smooth_slider,
                    user_data=idx,
                )

    def update_slider_labels(self, skip_idx: Optional[int] = None) -> None:
        """Update slider labels and values to reflect current state."""
        if dpg is None or not self.slider_ids:
            return

        with self.app.state_lock:
            conductors = list(self.app.state.project.conductors)
            selected_idx = self.app.state.selected_idx

        for idx, param_map in list(self.slider_ids.items()):
            if idx >= len(conductors):
                continue
                
            if skip_idx is not None and idx == skip_idx:
                continue

            conductor = conductors[idx]
            for param_name, slider_id in param_map.items():
                if dpg.does_item_exist(slider_id):
                    val = conductor.params.get(param_name, 0.0)
                    dpg.set_value(slider_id, float(val))

    def on_param_slider(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle generic parameter slider change."""
        if dpg is None or user_data is None:
            return

        idx = int(user_data["idx"])
        param_name = user_data["param"]
        value = float(app_data)

        with self.app.state_lock:
            if idx < len(self.app.state.project.conductors):
                # Update params dict directly
                # TODO: Add an action for this to support undo/redo
                self.app.state.project.conductors[idx].params[param_name] = value
                self.app.state.field_cache = None

        self.app.canvas_renderer.mark_dirty()
        dpg.set_value("status_text", f"Obj {idx + 1} {param_name} = {value:.3f}")
        self.update_slider_labels(skip_idx=idx)

    def on_edge_smooth_slider(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle edge smoothing slider change."""
        if dpg is None or user_data is None:
            return

        idx = int(user_data)
        value = float(app_data)

        with self.app.state_lock:
            if idx < len(self.app.state.project.conductors):
                self.app.state.project.conductors[idx].edge_smooth_sigma = value
                self.app.state.field_cache = None

        self.app.canvas_renderer.mark_dirty()
        dpg.set_value("status_text", f"Obj {idx + 1} edge smoothing = {value:.1f} px")
