"""Display pipeline controller for FlowCol UI.

Manages the real-time display pipeline:
- Display cache invalidation (base_rgb)
- Texture refreshes
- Coordinating postprocessing updates
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class DisplayPipelineController:
    """Controller for display pipeline and postprocessing coordination.

    Owns the TextureManager and provides clean API for display updates.
    """

    def __init__(self, app: "FlowColApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main FlowColApp instance
        """
        self.app = app

        # Import here to avoid circular dependency
        from flowcol.ui.dpg.texture_manager import TextureManager
        self.texture_manager = TextureManager(app)

    def refresh_display(self, invalidate_cache: bool = False) -> None:
        """Refresh the display with current postprocessing settings.

        Args:
            invalidate_cache: If True, invalidate base_rgb cache before refresh
        """
        if invalidate_cache:
            with self.app.state_lock:
                cache = self.app.state.render_cache
                if cache is not None:
                    cache.base_rgb = None

        self.texture_manager.refresh_render_texture()
        self.app.canvas_renderer.mark_dirty()

    def invalidate_and_refresh(self) -> None:
        """Invalidate display cache and refresh (for settings that affect base colorization)."""
        self.refresh_display(invalidate_cache=True)
