#!/usr/bin/env python3
"""Minimal pygame UI for conductor placement."""

import pygame
import numpy as np
from PIL import Image
from flowcol.types import Conductor, Project, UIState
from flowcol.field import compute_field
from flowcol.render import (
    compute_lic,
    array_to_surface,
    save_render,
    downsample_lic,
    list_color_palettes,
    apply_gaussian_highpass,
    apply_highpass_clahe,
)
from ui_panel import panel, render_menu, highpass_menu

BG_COLOR = (20, 20, 20)
CONDUCTOR_COLORS = [(100, 150, 255, 180), (255, 100, 150, 180), (150, 255, 100, 180), (255, 200, 100, 180)]
SELECTED_COLOR = (255, 255, 100, 200)

MAX_RENDER_DIM = 32768


def mask_to_surface(mask: np.ndarray, color: tuple) -> pygame.Surface:
    """Convert mask array to pygame surface with color."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = color[:3]
    rgba[..., 3] = (mask * color[3]).astype(np.uint8)
    img = Image.fromarray(rgba, mode='RGBA')
    return pygame.image.fromstring(img.tobytes(), img.size, img.mode)


def point_in_conductor(pos: tuple[int, int], conductor: Conductor) -> bool:
    """Check if point is inside conductor mask."""
    x, y = pos
    h, w = conductor.mask.shape
    cx, cy = round(conductor.position[0]), round(conductor.position[1])
    if x < cx or x >= cx + w or y < cy or y >= cy + h:
        return False
    return conductor.mask[int(y - cy), int(x - cx)] > 0.5


def draw_edit_mode(screen: pygame.Surface, project: Project, state: UIState):
    for i, conductor in enumerate(project.conductors):
        color = SELECTED_COLOR if i == state.selected_idx else CONDUCTOR_COLORS[i % len(CONDUCTOR_COLORS)]
        pos = (round(conductor.position[0]), round(conductor.position[1]))
        screen.blit(mask_to_surface(conductor.mask, color), pos)


def draw_render_mode(screen: pygame.Surface, state: UIState):
    if state.rendered_surface:
        screen.blit(state.rendered_surface, (0, 0))


def _current_palette_name(state: UIState) -> str | None:
    palettes = list_color_palettes()
    if not palettes:
        return None
    idx = state.color_palette_index % len(palettes)
    return palettes[idx]


def _reference_resolution(project: Project, state: UIState) -> float:
    compute_h, compute_w = state.current_compute_resolution
    if compute_h > 0 and compute_w > 0:
        return float(min(compute_h, compute_w))
    canvas_min = float(min(project.canvas_resolution))
    if state.current_render_multiplier > 0:
        return canvas_min * state.current_render_multiplier * state.current_supersample
    return canvas_min


def update_postprocess_highres(project: Project, state: UIState) -> np.ndarray | None:
    """Rebuild high-resolution data after post-processing adjustments."""
    if state.original_render_data is None:
        state.highres_render_data = None
        return None

    ref_size = _reference_resolution(project, state)
    working = state.original_render_data.astype(np.float32, copy=True)

    if state.detail_enabled:
        detail_sigma = max(state.detail_sigma_factor, 0.0) * ref_size
        if detail_sigma > 0.0:
            working = apply_gaussian_highpass(working, detail_sigma)

    hp_menu = state.highpass_menu
    if getattr(hp_menu, "enabled", False):
        sigma_px = max(hp_menu.sigma_factor * ref_size, 0.0)
        working = apply_highpass_clahe(
            working,
            sigma_px,
            hp_menu.clip_limit,
            hp_menu.kernel_rows,
            hp_menu.kernel_cols,
            hp_menu.num_bins,
            hp_menu.strength,
        )

    state.highres_render_data = working
    state.postprocess_dirty = False
    return working


def update_render_surface(project: Project, state: UIState, arr: np.ndarray):
    canvas_w, canvas_h = project.canvas_resolution
    render_h, render_w = arr.shape
    palette_name = _current_palette_name(state)
    surface = array_to_surface(
        arr,
        use_color=state.color_enabled,
        palette=palette_name,
        contrast=state.color_contrast,
        gamma=state.color_gamma,
        clip_percent=state.color_clip_percent,
    )
    if render_w <= canvas_w and render_h <= canvas_h:
        state.rendered_surface = surface
    else:
        scale = min(canvas_w / render_w, canvas_h / render_h)
        display_w, display_h = int(render_w * scale), int(render_h * scale)
        state.rendered_surface = pygame.transform.smoothscale(surface, (display_w, display_h))


def recompute_display(project: Project, state: UIState) -> np.ndarray | None:
    if state.original_render_data is None or state.current_render_multiplier <= 0:
        return None

    if state.highres_render_data is None or state.postprocess_dirty:
        if update_postprocess_highres(project, state) is None:
            return None

    render_h, render_w = state.current_render_shape
    if render_w <= 0 or render_h <= 0:
        return None

    sigma = state.downsample.sigma_factor * state.current_supersample
    display_array = downsample_lic(
        state.highres_render_data,
        (render_h, render_w),
        state.current_supersample,
        sigma,
    )

    state.current_render_data = display_array
    update_render_surface(project, state, display_array)
    return display_array


def perform_render(project: Project, state: UIState, multiplier: float):
    canvas_w, canvas_h = project.canvas_resolution

    render_w = max(1, int(round(canvas_w * multiplier)))
    render_h = max(1, int(round(canvas_h * multiplier)))

    supersample = state.supersample_factor
    scale = multiplier * supersample
    margin_physical = state.margin_factor * float(min(canvas_w, canvas_h))
    margin_tuple = (margin_physical, margin_physical)
    domain_w = canvas_w + 2.0 * margin_physical
    domain_h = canvas_h + 2.0 * margin_physical
    compute_w = max(1, int(round(domain_w * scale)))
    compute_h = max(1, int(round(domain_h * scale)))

    if compute_w > MAX_RENDER_DIM or compute_h > MAX_RENDER_DIM:
        return False

    ex, ey = compute_field(project, multiplier, supersample, margin_tuple)
    num_passes = max(1, state.render_menu.num_passes)
    min_compute = min(compute_w, compute_h)
    streamlength_pixels = max(int(round(project.streamlength_factor * min_compute)), 1)
    seed = state.noise_seed
    lic_array = compute_lic(
        ex,
        ey,
        streamlength_pixels,
        num_passes=num_passes,
        seed=seed,
        noise_sigma=state.noise_sigma,
    )

    canvas_scaled_w = max(1, int(round(canvas_w * scale)))
    canvas_scaled_h = max(1, int(round(canvas_h * scale)))
    offset_x = int(round(margin_physical * scale))
    offset_y = int(round(margin_physical * scale))
    offset_x = min(offset_x, max(0, lic_array.shape[1] - canvas_scaled_w))
    offset_y = min(offset_y, max(0, lic_array.shape[0] - canvas_scaled_h))
    crop_x0 = max(0, offset_x)
    crop_y0 = max(0, offset_y)
    crop_x1 = min(crop_x0 + canvas_scaled_w, lic_array.shape[1])
    crop_y1 = min(crop_y0 + canvas_scaled_h, lic_array.shape[0])
    lic_cropped = lic_array[crop_y0:crop_y1, crop_x0:crop_x1]

    state.original_render_data = lic_cropped.astype(np.float32, copy=True)
    state.highres_render_data = None
    state.current_render_multiplier = multiplier
    state.current_supersample = supersample
    state.current_noise_seed = seed
    state.current_compute_resolution = (compute_h, compute_w)
    state.current_canvas_scaled = lic_cropped.shape
    state.current_margin = margin_physical
    state.current_render_shape = (render_h, render_w)
    state.render_mode = "render"
    state.postprocess_dirty = True

    display_array = recompute_display(project, state)
    if display_array is not None:
        palette_name = _current_palette_name(state)
        save_render(
            display_array,
            project,
            multiplier,
            use_color=state.color_enabled,
            palette=palette_name,
            contrast=state.color_contrast,
            gamma=state.color_gamma,
            clip_percent=state.color_clip_percent,
        )
    state.downsample.dirty = False

    return True


def handle_events(state: UIState, project: Project, canvas_res: tuple[int, int]):
    running = True
    mouse_pos = pygame.mouse.get_pos()
    mouse_down = False
    key_pressed = None
    menu_state = state.render_menu

    for event in pygame.event.get():
        menu_active = menu_state.is_open or state.highpass_menu.is_open
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_down = True
            if not menu_active and mouse_pos[0] < canvas_res[0] and state.render_mode == "edit":
                state.selected_idx = -1
                for i in reversed(range(len(project.conductors))):
                    if point_in_conductor(mouse_pos, project.conductors[i]):
                        state.selected_idx = i
                        state.mouse_dragging = True
                        state.last_mouse_pos = mouse_pos
                        break
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            state.mouse_dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if not menu_active and state.mouse_dragging and state.selected_idx >= 0:
                dx = float(mouse_pos[0] - state.last_mouse_pos[0])
                dy = float(mouse_pos[1] - state.last_mouse_pos[1])
                conductor = project.conductors[state.selected_idx]
                conductor.position = (conductor.position[0] + dx, conductor.position[1] + dy)
                state.last_mouse_pos = mouse_pos
                state.field_dirty = True
        elif event.type == pygame.KEYDOWN:
            key_pressed = event.key
            if event.key == pygame.K_ESCAPE:
                hp_menu = state.highpass_menu
                if hp_menu.is_open:
                    hp_menu.is_open = False
                    hp_menu.focused_field = -1
                    hp_menu.pending_clear = -1
                    hp_menu.sigma_factor_text = f"{hp_menu.sigma_factor:.4f}"
                    hp_menu.clip_text = f"{hp_menu.clip_limit:.4f}"
                    hp_menu.kernel_rows_text = str(hp_menu.kernel_rows)
                    hp_menu.kernel_cols_text = str(hp_menu.kernel_cols)
                    hp_menu.num_bins_text = str(hp_menu.num_bins)
                elif menu_state.is_open:
                    menu_state.is_open = False
                    menu_state.input_focused = False
                    menu_state.streamlength_input_focused = False
                    menu_state.streamlength_pending_clear = False
                    menu_state.streamlength_text = f"{project.streamlength_factor:.4f}"
                    menu_state.pending_streamlength_factor = project.streamlength_factor
                    menu_state.margin_input_focused = False
                    menu_state.margin_pending_clear = False
                    menu_state.margin_text = f"{state.margin_factor:.3f}"
                    menu_state.pending_margin_factor = state.margin_factor
                    menu_state.seed_input_focused = False
                    menu_state.seed_pending_clear = False
                    state.noise_seed_text = str(state.noise_seed)
                    if menu_state.num_passes == 0:
                        menu_state.num_passes = 1
                else:
                    running = False
            elif event.key == pygame.K_DELETE and state.selected_idx >= 0:
                del project.conductors[state.selected_idx]
                state.selected_idx = -1
                state.field_dirty = True

    return running, mouse_pos, mouse_down, key_pressed


def main():
    pygame.init()

    canvas_res = (800, 600)
    panel_width = 220

    project = Project(canvas_resolution=canvas_res)
    state = UIState(project=project)
    state.render_menu.pending_streamlength_factor = project.streamlength_factor
    state.render_menu.streamlength_text = f"{project.streamlength_factor:.4f}"
    state.render_menu.margin_text = f"{state.margin_factor:.3f}"
    state.render_menu.pending_margin_factor = state.margin_factor
    state.render_menu.noise_sigma_text = f"{state.noise_sigma:.2f}"
    state.render_menu.pending_noise_sigma = state.noise_sigma
    state.highpass_menu.sigma_factor_text = f"{state.highpass_menu.sigma_factor:.4f}"
    state.noise_seed_text = str(state.noise_seed)
    state.downsample.sigma_text = f"{state.downsample.sigma_factor:.2f}"
    state.canvas_width_text = str(project.canvas_resolution[0])
    state.canvas_height_text = str(project.canvas_resolution[1])

    window_res = (canvas_res[0] + panel_width, canvas_res[1])
    screen = pygame.display.set_mode(window_res)
    pygame.display.set_caption("FlowCol")
    clock = pygame.time.Clock()

    running = True
    while running:
        canvas_res = project.canvas_resolution
        running, mouse_pos, mouse_down, key_pressed = handle_events(state, project, canvas_res)
        if not running:
            break

        screen.fill(BG_COLOR)

        if state.render_mode == "edit":
            draw_edit_mode(screen, project, state)
        else:
            draw_render_mode(screen, state)

        action = panel(screen, project, state, mouse_pos, mouse_down, key_pressed)

        highpass_state = state.highpass_menu
        if highpass_state.is_open:
            hp_action = highpass_menu(screen, state, mouse_pos, mouse_down, key_pressed)
            if hp_action == -999:
                highpass_state.is_open = False
                highpass_state.focused_field = -1
                highpass_state.pending_clear = -1
                highpass_state.sigma_factor_text = f"{highpass_state.sigma_factor:.4f}"
                highpass_state.clip_text = f"{highpass_state.clip_limit:.4f}"
                highpass_state.kernel_rows_text = str(highpass_state.kernel_rows)
                highpass_state.kernel_cols_text = str(highpass_state.kernel_cols)
                highpass_state.num_bins_text = str(highpass_state.num_bins)
                highpass_state.strength_dragging = False
            elif isinstance(hp_action, tuple):
                sigma_factor, clip, rows, cols, bins, strength = hp_action
                sigma_factor = max(sigma_factor, 1e-5)
                clip = max(clip, 1e-4)
                rows = max(rows, 1)
                cols = max(cols, 1)
                bins = max(bins, 2)
                strength = np.clip(strength, 0.0, 1.0)

                highpass_state.sigma_factor = sigma_factor
                highpass_state.clip_limit = clip
                highpass_state.kernel_rows = rows
                highpass_state.kernel_cols = cols
                highpass_state.num_bins = bins
                highpass_state.strength = strength

                highpass_state.sigma_factor_text = f"{sigma_factor:.4f}"
                highpass_state.clip_text = f"{clip:.4f}"
                highpass_state.kernel_rows_text = str(rows)
                highpass_state.kernel_cols_text = str(cols)
                highpass_state.num_bins_text = str(bins)

                highpass_state.is_open = False
                highpass_state.focused_field = -1
                highpass_state.pending_clear = -1
                highpass_state.strength_dragging = False
                highpass_state.enabled = True
                state.postprocess_dirty = True
                if state.render_mode == "render":
                    recompute_display(project, state)
                    state.downsample.dirty = False
                else:
                    state.downsample.dirty = True
        elif state.render_mode == "render" and state.downsample.dirty:
            state.downsample.dirty = False
            recompute_display(project, state)
        menu_state = state.render_menu

        if menu_state.is_open:
            menu_action = render_menu(screen, state, mouse_pos, mouse_down, key_pressed)
            if menu_action == -999:
                menu_state.is_open = False
                menu_state.input_focused = False
                menu_state.streamlength_input_focused = False
                menu_state.streamlength_pending_clear = False
                menu_state.streamlength_text = f"{project.streamlength_factor:.4f}"
                menu_state.pending_streamlength_factor = project.streamlength_factor
                menu_state.margin_input_focused = False
                menu_state.margin_pending_clear = False
                menu_state.margin_text = f"{state.margin_factor:.3f}"
                menu_state.pending_margin_factor = state.margin_factor
                menu_state.seed_input_focused = False
                menu_state.seed_pending_clear = False
                state.noise_seed_text = str(state.noise_seed)
                if menu_state.num_passes == 0:
                    menu_state.num_passes = 1
            elif menu_action and menu_action > 0:
                action = menu_action
                menu_state.is_open = False
                menu_state.input_focused = False
                menu_state.streamlength_input_focused = False
                menu_state.streamlength_pending_clear = False
                menu_state.margin_input_focused = False
                menu_state.margin_pending_clear = False
                menu_state.seed_input_focused = False
                menu_state.seed_pending_clear = False
                project.streamlength_factor = menu_state.pending_streamlength_factor
                menu_state.streamlength_text = f"{project.streamlength_factor:.4f}"
                state.margin_factor = menu_state.pending_margin_factor
                menu_state.margin_text = f"{state.margin_factor:.3f}"
                state.noise_seed_text = str(state.noise_seed)
                if menu_state.num_passes == 0:
                    menu_state.num_passes = 1

        if action == -2:
            expected_window = (project.canvas_resolution[0] + panel_width, project.canvas_resolution[1])
            screen = pygame.display.set_mode(expected_window)
        elif action and action > 0:
            perform_render(project, state, action)
        elif action == -1:
            state.render_mode = "edit"
            expected_window = (project.canvas_resolution[0] + panel_width, project.canvas_resolution[1])
            screen = pygame.display.set_mode(expected_window)
        elif action == -3:
            if state.original_render_data is not None:
                state.detail_enabled = False
                state.detail_sigma_factor = 0.02
                state.detail_factor_text = f"{state.detail_sigma_factor:.3f}"
                state.detail_input_focused = False
                state.detail_pending_clear = False
                hp_menu = state.highpass_menu
                hp_menu.enabled = False
                state.highres_render_data = None
                state.postprocess_dirty = True
                if state.render_mode == "render":
                    recompute_display(project, state)
                    state.downsample.dirty = False
                else:
                    state.downsample.dirty = True

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
