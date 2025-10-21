#!/usr/bin/env python3
"""Minimal pygame UI for conductor placement."""

import pygame
import numpy as np
from PIL import Image
from flowcol.types import Conductor, Project, UIState
from flowcol.field import compute_field
from flowcol.render import compute_lic, array_to_surface, save_render, apply_highpass_clahe
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


def update_render_surface(project: Project, state: UIState, arr: np.ndarray):
    canvas_w, canvas_h = project.canvas_resolution
    render_h, render_w = arr.shape
    surface = array_to_surface(arr)
    if render_w <= canvas_w and render_h <= canvas_h:
        state.rendered_surface = surface
    else:
        scale = min(canvas_w / render_w, canvas_h / render_h)
        display_w, display_h = int(render_w * scale), int(render_h * scale)
        state.rendered_surface = pygame.transform.smoothscale(surface, (display_w, display_h))


def perform_render(project: Project, state: UIState, multiplier: int):
    canvas_w, canvas_h = project.canvas_resolution
    render_w, render_h = canvas_w * multiplier, canvas_h * multiplier

    if render_w > MAX_RENDER_DIM or render_h > MAX_RENDER_DIM:
        return False

    ex, ey = compute_field(project, multiplier)
    num_passes = max(1, state.render_menu.num_passes)
    lic_array = compute_lic(
        ex,
        ey,
        project.streamlength * multiplier,
        num_passes=num_passes,
        seed=0,
    )
    save_render(lic_array, project, multiplier)

    state.original_render_data = lic_array.copy()
    state.current_render_data = lic_array
    state.current_render_multiplier = multiplier
    state.render_mode = "render"

    update_render_surface(project, state, lic_array)

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
                    hp_menu.sigma_text = f"{hp_menu.sigma:.2f}"
                    hp_menu.clip_text = f"{hp_menu.clip_limit:.4f}"
                    hp_menu.kernel_rows_text = str(hp_menu.kernel_rows)
                    hp_menu.kernel_cols_text = str(hp_menu.kernel_cols)
                    hp_menu.num_bins_text = str(hp_menu.num_bins)
                elif menu_state.is_open:
                    menu_state.is_open = False
                    menu_state.input_focused = False
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

        action = panel(screen, project, state, mouse_pos, mouse_down)

        highpass_state = state.highpass_menu
        if highpass_state.is_open:
            hp_action = highpass_menu(screen, state, mouse_pos, mouse_down, key_pressed)
            if hp_action == -999:
                highpass_state.is_open = False
                highpass_state.focused_field = -1
                highpass_state.pending_clear = -1
                highpass_state.sigma_text = f"{highpass_state.sigma:.2f}"
                highpass_state.clip_text = f"{highpass_state.clip_limit:.4f}"
                highpass_state.kernel_rows_text = str(highpass_state.kernel_rows)
                highpass_state.kernel_cols_text = str(highpass_state.kernel_cols)
                highpass_state.num_bins_text = str(highpass_state.num_bins)
            elif isinstance(hp_action, tuple):
                sigma, clip, rows, cols, bins = hp_action
                sigma = max(sigma, 0.1)
                clip = max(clip, 1e-4)
                rows = max(rows, 1)
                cols = max(cols, 1)
                bins = max(bins, 2)

                highpass_state.sigma = sigma
                highpass_state.clip_limit = clip
                highpass_state.kernel_rows = rows
                highpass_state.kernel_cols = cols
                highpass_state.num_bins = bins

                highpass_state.sigma_text = f"{sigma:.2f}"
                highpass_state.clip_text = f"{clip:.4f}"
                highpass_state.kernel_rows_text = str(rows)
                highpass_state.kernel_cols_text = str(cols)
                highpass_state.num_bins_text = str(bins)

                highpass_state.is_open = False
                highpass_state.focused_field = -1
                highpass_state.pending_clear = -1

                if state.original_render_data is not None:
                    filtered = apply_highpass_clahe(
                        state.original_render_data,
                        sigma,
                        clip,
                        rows,
                        cols,
                        bins,
                    )
                    state.current_render_data = filtered
                    update_render_surface(project, state, filtered)
        menu_state = state.render_menu

        if menu_state.is_open:
            menu_action = render_menu(screen, state, mouse_pos, mouse_down, key_pressed)
            if menu_action == -999:
                menu_state.is_open = False
                menu_state.input_focused = False
                if menu_state.num_passes == 0:
                    menu_state.num_passes = 1
            elif menu_action and menu_action > 0:
                action = menu_action
                menu_state.is_open = False
                menu_state.input_focused = False
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
                state.current_render_data = state.original_render_data.copy()
                update_render_surface(project, state, state.current_render_data)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
