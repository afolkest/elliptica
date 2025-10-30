"""Conductor controls panel for FlowCol UI - voltage and blur sliders."""

from typing import Optional, Dict, TYPE_CHECKING

from flowcol.app import actions

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class ConductorControlsPanel:
    """Controller for conductor-specific UI controls (voltage/blur sliders)."""

    def __init__(self, app: "FlowColApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main FlowColApp instance
        """
        self.app = app

        # Widget IDs for conductor controls
        self.conductor_controls_container_id: Optional[int] = None
        self.conductor_slider_ids: Dict[int, int] = {}

    def build_conductor_controls_container(self, parent) -> None:
        """Build conductor controls container in edit controls panel.

        Args:
            parent: Parent widget ID to add container to
        """
        if dpg is None:
            return

        dpg.add_text("Conductor Voltages", parent=parent)
        self.conductor_controls_container_id = dpg.add_child_window(
            autosize_x=True,
            height=400,
            border=False,
            tag="conductor_controls_child",
            no_scroll_with_mouse=False,
            parent=parent,
        )

    def rebuild_conductor_controls(self) -> None:
        """Rebuild all conductor voltage and blur sliders from current project state."""
        if dpg is None or self.conductor_controls_container_id is None:
            return

        dpg.delete_item(self.conductor_controls_container_id, children_only=True)
        self.conductor_slider_ids.clear()

        with self.app.state_lock:
            conductors = list(self.app.state.project.conductors)
            selected_idx = self.app.state.selected_idx

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
                callback=self.on_conductor_voltage_slider,
                user_data=idx,
                parent=self.conductor_controls_container_id,
            )
            self.conductor_slider_ids[idx] = slider_id

            # Add blur slider with fractional toggle
            with dpg.group(horizontal=True, parent=self.conductor_controls_container_id):
                if conductor.blur_is_fractional:
                    max_val = 0.1
                    fmt = "%.3f"
                else:
                    max_val = 20.0
                    fmt = "%.1f px"
                dpg.add_slider_float(
                    label=f"  Blur {idx + 1}",
                    default_value=float(conductor.blur_sigma),
                    min_value=0.0,
                    max_value=max_val,
                    format=fmt,
                    callback=self.on_conductor_blur_slider,
                    user_data=idx,
                    width=200,
                    tag=f"blur_slider_{idx}",
                )
                dpg.add_checkbox(
                    label="Frac",
                    default_value=conductor.blur_is_fractional,
                    callback=self.on_blur_fractional_toggle,
                    user_data=idx,
                    tag=f"blur_frac_checkbox_{idx}",
                )

    def update_conductor_slider_labels(self, skip_idx: Optional[int] = None) -> None:
        """Update conductor slider labels and values to reflect current state.

        Args:
            skip_idx: Optional conductor index to skip updating (for real-time slider feedback)
        """
        if dpg is None or not self.conductor_slider_ids:
            return

        with self.app.state_lock:
            conductors = list(self.app.state.project.conductors)
            selected_idx = self.app.state.selected_idx

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

    # ------------------------------------------------------------------
    # Conductor slider callbacks
    # ------------------------------------------------------------------

    def on_conductor_voltage_slider(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle conductor voltage slider change."""
        if dpg is None or user_data is None:
            return

        idx = int(user_data)
        value = float(app_data)

        with self.app.state_lock:
            actions.set_conductor_voltage(self.app.state, idx, value)

        self.app.canvas_renderer.mark_dirty()
        dpg.set_value("status_text", f"C{idx + 1} voltage = {value:.3f}")
        self.update_conductor_slider_labels(skip_idx=idx)

    def on_conductor_blur_slider(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle conductor blur slider change."""
        if dpg is None or user_data is None:
            return

        idx = int(user_data)
        value = float(app_data)

        with self.app.state_lock:
            if idx < len(self.app.state.project.conductors):
                self.app.state.project.conductors[idx].blur_sigma = value
                self.app.state.field_cache = None
                is_frac = self.app.state.project.conductors[idx].blur_is_fractional

        self.app.canvas_renderer.mark_dirty()
        if is_frac:
            dpg.set_value("status_text", f"C{idx + 1} blur = {value:.3f} (fraction)")
        else:
            dpg.set_value("status_text", f"C{idx + 1} blur = {value:.1f} px")

    def on_blur_fractional_toggle(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle blur fractional/pixels toggle."""
        if dpg is None or user_data is None:
            return

        idx = int(user_data)
        is_fractional = bool(app_data)

        with self.app.state_lock:
            if idx < len(self.app.state.project.conductors):
                conductor = self.app.state.project.conductors[idx]
                conductor.blur_is_fractional = is_fractional
                # Convert value when switching modes
                if is_fractional:
                    # Convert from pixels to fraction (assume ~1000px reference)
                    conductor.blur_sigma = min(conductor.blur_sigma / 1000.0, 0.1)
                else:
                    # Convert from fraction to pixels
                    conductor.blur_sigma = conductor.blur_sigma * 1000.0
                self.app.state.field_cache = None

        # Update slider range and format
        slider_id = f"blur_slider_{idx}"
        if dpg.does_item_exist(slider_id):
            if is_fractional:
                dpg.configure_item(slider_id, max_value=0.1, format="%.3f")
            else:
                dpg.configure_item(slider_id, max_value=20.0, format="%.1f px")
            with self.app.state_lock:
                if idx < len(self.app.state.project.conductors):
                    dpg.set_value(slider_id, self.app.state.project.conductors[idx].blur_sigma)

        self.app.canvas_renderer.mark_dirty()
