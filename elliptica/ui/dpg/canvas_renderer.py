"""Canvas renderer for Elliptica UI - handles all canvas drawing operations."""

from typing import TYPE_CHECKING, Optional
import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


# Boundary overlay colors for edit mode (RGBA with alpha)
BOUNDARY_COLORS = [
    (0.39, 0.59, 1.0, 0.7),
    (1.0, 0.39, 0.59, 0.7),
    (0.59, 1.0, 0.39, 0.7),
    (1.0, 0.78, 0.39, 0.7),
]


class CanvasRenderer:
    """Controller for canvas drawing operations."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize renderer with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app
        self.canvas_dirty: bool = True

        # Cache for selection outline contour (render mode)
        self._selection_contours: list[np.ndarray] | None = None
        self._selection_cache_key: tuple = ()  # (selected_idx, render_cache_id)

        # Cache for boundary contours in edit mode: boundary_idx -> (mask_id, contours_at_origin)
        self._boundary_contour_cache: dict[int, tuple[int, list[np.ndarray]]] = {}

    def mark_dirty(self) -> None:
        """Mark canvas as needing redraw."""
        self.canvas_dirty = True

    def invalidate_selection_contour(self) -> None:
        """Clear cached selection contour (call when selection or render changes)."""
        self._selection_contours = None
        self._selection_cache_key = ()

    def invalidate_boundary_contour_cache(self, idx: int = -1) -> None:
        """Clear contour cache for boundary (or all if idx=-1).

        Note: Cache uses id(mask) for validation. This assumes masks are not
        modified in-place. Position drags reuse the same mask object, so the
        cache remains valid. Scaling creates a new mask object, causing a cache miss.
        """
        if idx < 0:
            self._boundary_contour_cache.clear()
        else:
            self._boundary_contour_cache.pop(idx, None)

    def _extract_contours(self, mask: np.ndarray) -> list[np.ndarray]:
        """Extract ordered contour points from a mask.

        Uses skimage if available, otherwise falls back to scipy-based method.
        Returns list of Nx2 arrays of (x, y) points.
        """
        try:
            from skimage import measure
            contours = measure.find_contours(mask, 0.5)
            # Convert (row, col) to (x, y)
            return [np.column_stack([c[:, 1], c[:, 0]]) for c in contours]
        except ImportError:
            return self._extract_contours_fallback(mask)

    def _extract_contours_fallback(self, mask: np.ndarray) -> list[np.ndarray]:
        """Extract contours using scipy (fallback when skimage unavailable)."""
        binary = (mask > 0.5).astype(np.uint8)
        eroded = ndimage.binary_erosion(binary)
        boundary = binary & ~eroded

        # Label connected components of boundary
        labeled, num_features = ndimage.label(boundary)

        contours = []
        for i in range(1, num_features + 1):
            rows, cols = np.where(labeled == i)
            if len(rows) < 20:
                continue

            # Order points using nearest neighbor heuristic
            points = list(zip(cols.astype(float), rows.astype(float)))
            ordered = [points.pop(0)]

            while points:
                last = ordered[-1]
                min_dist = float('inf')
                min_idx = 0
                for j, p in enumerate(points):
                    dist = (p[0] - last[0])**2 + (p[1] - last[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = j
                ordered.append(points.pop(min_idx))

            contours.append(np.array(ordered))

        return contours

    def _get_selection_contours(self, selected_idx: int, region_type: str, render_cache, canvas_w: int, canvas_h: int) -> list[np.ndarray] | None:
        """Get contours for selected region, scaled to canvas coordinates.

        Args:
            selected_idx: Boundary index
            region_type: "surface" or "interior"
            render_cache: Render cache with masks
            canvas_w, canvas_h: Canvas dimensions for scaling
        """
        if render_cache is None:
            return None

        # Pick the right mask list based on region type
        if region_type == "interior":
            masks = render_cache.interior_masks
        else:
            masks = render_cache.boundary_masks

        if masks is None or selected_idx < 0 or selected_idx >= len(masks):
            return None

        mask = masks[selected_idx]
        if mask is None:
            return None

        # Check cache
        cache_key = (selected_idx, region_type, id(render_cache))
        if self._selection_contours is not None and self._selection_cache_key == cache_key:
            return self._selection_contours

        # Extract contours at render resolution
        contours = self._extract_contours(mask)
        if not contours:
            return None

        # Scale to canvas coordinates
        tex_h, tex_w = mask.shape[:2]
        scale_x = canvas_w / tex_w
        scale_y = canvas_h / tex_h

        scaled_contours = []
        for contour in contours:
            scaled = contour.copy()
            scaled[:, 0] *= scale_x
            scaled[:, 1] *= scale_y
            scaled_contours.append(scaled)

        # Cache
        self._selection_contours = scaled_contours
        self._selection_cache_key = cache_key

        return scaled_contours

    def _draw_dashed_contour(self, contour: np.ndarray, parent,
                              dash_length: int = 8, gap_length: int = 6,
                              color=(255, 255, 255, 220), thickness: float = 2.0) -> None:
        """Draw a contour as a dashed white line with black outline for visibility."""
        if len(contour) < 2:
            return

        # Points are roughly 1 pixel apart, so we can use point count for dash length
        period = dash_length + gap_length
        i = 0
        while i < len(contour):
            end = min(i + dash_length, len(contour))
            segment = contour[i:end]
            if len(segment) >= 2:
                points = [tuple(p) for p in segment]
                # Draw black outline first (slightly thicker)
                dpg.draw_polyline(points, color=(0, 0, 0, 180), thickness=thickness + 1.5, parent=parent)
                # Draw white line on top
                dpg.draw_polyline(points, color=color, thickness=thickness, parent=parent)
            i += period

    def _get_boundary_contours(self, boundary, offset_x: float, offset_y: float, idx: int = -1) -> list[np.ndarray]:
        """Extract contours from boundary mask and offset to canvas position.

        Caches contours by mask identity to avoid recomputing during drags.
        The cache key is id(mask), which doesn't change during position drags
        but does change when mask is recreated (e.g., during scaling).

        Args:
            boundary: BoundaryObject with mask
            offset_x, offset_y: Canvas position offset
            idx: Boundary index for caching (-1 disables caching)
        """
        # Check cache - mask object identity doesn't change during position drag
        mask_id = id(boundary.mask)
        if idx >= 0 and idx in self._boundary_contour_cache:
            cached_mask_id, cached_contours = self._boundary_contour_cache[idx]
            if cached_mask_id == mask_id:
                # Reuse cached contours, just apply new offset
                result = []
                for contour in cached_contours:
                    offset_contour = contour.copy()
                    offset_contour[:, 0] += offset_x
                    offset_contour[:, 1] += offset_y
                    result.append(offset_contour)
                return result

        # Cache miss - extract contours at origin and cache
        contours = self._extract_contours(boundary.mask)
        if idx >= 0:
            self._boundary_contour_cache[idx] = (mask_id, contours)

        # Apply offset for return
        result = []
        for contour in contours:
            offset_contour = contour.copy()
            offset_contour[:, 0] += offset_x
            offset_contour[:, 1] += offset_y
            result.append(offset_contour)
        return result

    def draw(self) -> None:
        """Redraw the canvas with current state.

        This renders the background, grid (in edit mode), render texture (in render mode),
        and boundary overlays (in edit mode).
        """
        if dpg is None or self.app.canvas_layer_id is None:
            return
        self.canvas_dirty = False

        with self.app.state_lock:
            project = self.app.state.project
            selected_indices = set(self.app.state.selected_indices)
            selected_idx = self.app.state.get_single_selected_idx()  # For render mode
            selected_region_type = self.app.state.selected_region_type
            canvas_w, canvas_h = project.canvas_resolution
            boundaries = list(project.boundary_objects)
            render_cache = self.app.state.render_cache
            view_mode = self.app.state.view_mode

        # Get box selection state from controller
        box_select_active = self.app.canvas_controller.box_select_active
        box_start = self.app.canvas_controller.box_select_start
        box_end = self.app.canvas_controller.box_select_end

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

            # Draw selection outline for selected region in render mode (single select only)
            if selected_idx >= 0:
                contours = self._get_selection_contours(selected_idx, selected_region_type, render_cache, canvas_w, canvas_h)
                if contours:
                    for contour in contours:
                        self._draw_dashed_contour(contour, self.app.canvas_layer_id)

        if view_mode == "edit" or render_cache is None:
            for idx, boundary in enumerate(boundaries):
                tex_id = self.app.display_pipeline.texture_manager.ensure_boundary_texture(idx, boundary.mask, BOUNDARY_COLORS)
                x0, y0 = boundary.position
                width = boundary.mask.shape[1]
                height = boundary.mask.shape[0]
                dpg.draw_image(
                    tex_id,
                    pmin=(x0, y0),
                    pmax=(x0 + width, y0 + height),
                    uv_min=(0.0, 0.0),
                    uv_max=(1.0, 1.0),
                    parent=self.app.canvas_layer_id,
                )
                # Draw contour outline for selected boundaries (skip during drag/scale for performance)
                if idx in selected_indices and not self.app.canvas_controller.drag_active and not self.app.canvas_controller.scaling_active:
                    contours = self._get_boundary_contours(boundary, x0, y0, idx=idx)
                    for contour in contours:
                        self._draw_dashed_contour(
                            contour, self.app.canvas_layer_id,
                            color=(255, 255, 100, 220),  # Yellow selection color
                            thickness=2.0
                        )

            # Draw box selection rectangle
            if box_select_active:
                x1, y1 = box_start
                x2, y2 = box_end
                dpg.draw_rectangle(
                    (min(x1, x2), min(y1, y2)),
                    (max(x1, x2), max(y1, y2)),
                    color=(100, 150, 255, 200),
                    fill=(100, 150, 255, 40),
                    thickness=1.5,
                    parent=self.app.canvas_layer_id,
                )
