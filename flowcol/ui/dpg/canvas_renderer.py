"""Canvas renderer for FlowCol UI - handles all canvas drawing operations."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


# Conductor overlay colors for edit mode (RGBA with alpha)
CONDUCTOR_COLORS = [
    (0.39, 0.59, 1.0, 0.7),
    (1.0, 0.39, 0.59, 0.7),
    (0.59, 1.0, 0.39, 0.7),
    (1.0, 0.78, 0.39, 0.7),
]


class CanvasRenderer:
    """Controller for canvas drawing operations."""

    def __init__(self, app: "FlowColApp"):
        """Initialize renderer with reference to main app.

        Args:
            app: The main FlowColApp instance
        """
        self.app = app
        self.canvas_dirty: bool = True

    def mark_dirty(self) -> None:
        """Mark canvas as needing redraw."""
        self.canvas_dirty = True

    def draw(self) -> None:
        """Redraw the canvas with current state.

        This renders the background, grid (in edit mode), render texture (in render mode),
        and conductor overlays (in edit mode).
        """
        if dpg is None or self.app.canvas_layer_id is None:
            return
        self.canvas_dirty = False

        self.app.display_pipeline.texture_manager.refresh_render_texture()

        with self.app.state_lock:
            project = self.app.state.project
            selected_idx = self.app.state.selected_idx
            canvas_w, canvas_h = project.canvas_resolution
            conductors = list(project.conductors)
            render_cache = self.app.state.render_cache
            view_mode = self.app.state.view_mode

        # Clear the layer (transform persists on layer)
        dpg.delete_item(self.app.canvas_layer_id, children_only=True)

        # Draw background
        dpg.draw_rectangle((0, 0), (canvas_w, canvas_h), color=(60, 60, 60, 255), fill=(20, 20, 20, 255), parent=self.app.canvas_layer_id)

        # Draw grid in edit mode (based on physical units, not pixels)
        if view_mode == "edit":
            # Grid every 0.1 canvas units (10% of canvas dimension)
            grid_fraction = 0.1
            grid_spacing_x = canvas_w * grid_fraction
            grid_spacing_y = canvas_h * grid_fraction
            mid_x = canvas_w / 2.0
            mid_y = canvas_h / 2.0

            # Vertical grid lines
            x = 0.0
            while x <= canvas_w:
                is_midline = abs(x - mid_x) < 0.5
                color = (120, 120, 140, 180) if is_midline else (50, 50, 50, 100)
                thickness = 2.5 if is_midline else 1.0
                dpg.draw_line((x, 0), (x, canvas_h), color=color, thickness=thickness, parent=self.app.canvas_layer_id)
                x += grid_spacing_x

            # Horizontal grid lines
            y = 0.0
            while y <= canvas_h:
                is_midline = abs(y - mid_y) < 0.5
                color = (120, 120, 140, 180) if is_midline else (50, 50, 50, 100)
                thickness = 2.5 if is_midline else 1.0
                dpg.draw_line((0, y), (canvas_w, y), color=color, thickness=thickness, parent=self.app.canvas_layer_id)
                y += grid_spacing_y

        if view_mode == "render" and render_cache and self.app.display_pipeline.texture_manager.render_texture_id is not None and self.app.display_pipeline.texture_manager.render_texture_size:
            tex_w, tex_h = self.app.display_pipeline.texture_manager.render_texture_size
            if tex_w > 0 and tex_h > 0:
                scale_x = canvas_w / tex_w
                scale_y = canvas_h / tex_h
                pmax = (tex_w * scale_x, tex_h * scale_y)
                dpg.draw_image(
                    self.app.display_pipeline.texture_manager.render_texture_id,
                    (0, 0),
                    pmax,
                    uv_min=(0.0, 0.0),
                    uv_max=(1.0, 1.0),
                    parent=self.app.canvas_layer_id,
                )

        if view_mode == "edit" or render_cache is None:
            for idx, conductor in enumerate(conductors):
                tex_id = self.app.display_pipeline.texture_manager.ensure_conductor_texture(idx, conductor.mask, CONDUCTOR_COLORS)
                x0, y0 = conductor.position
                width = conductor.mask.shape[1]
                height = conductor.mask.shape[0]
                dpg.draw_image(
                    tex_id,
                    pmin=(x0, y0),
                    pmax=(x0 + width, y0 + height),
                    uv_min=(0.0, 0.0),
                    uv_max=(1.0, 1.0),
                    parent=self.app.canvas_layer_id,
                )
                if idx == selected_idx:
                    dpg.draw_rectangle(
                        (x0, y0),
                        (x0 + width, y0 + height),
                        color=(255, 255, 100, 200),
                        thickness=2.0,
                        parent=self.app.canvas_layer_id,
                    )
