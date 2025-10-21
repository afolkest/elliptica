#!/usr/bin/env python3
"""Minimal pygame UI for conductor placement."""

import pygame
import numpy as np
from PIL import Image
from flowcol.types import Conductor, Project, UIState
from flowcol.field import compute_field
from flowcol.render import compute_lic, array_to_surface, save_render
from ui_panel import panel, render_menu

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


def perform_render(project: Project, state: UIState, multiplier: int):
    canvas_w, canvas_h = project.canvas_resolution
    render_w, render_h = canvas_w * multiplier, canvas_h * multiplier

    if render_w > MAX_RENDER_DIM or render_h > MAX_RENDER_DIM:
        return False

    ex, ey = compute_field(project, multiplier)
    num_passes = max(1, state.render_menu.num_passes)
    lic_array = compute_lic(ex, ey, project, num_passes)
    save_render(lic_array, project, multiplier)

    state.original_render_data = lic_array.copy()
    state.current_render_data = lic_array
    state.current_render_multiplier = multiplier
    state.render_mode = "render"

    render_h, render_w = lic_array.shape
    full_surface = array_to_surface(lic_array)

    if render_w <= canvas_w and render_h <= canvas_h:
        state.rendered_surface = full_surface
    else:
        scale = min(canvas_w / render_w, canvas_h / render_h)
        display_w, display_h = int(render_w * scale), int(render_h * scale)
        state.rendered_surface = pygame.transform.smoothscale(full_surface, (display_w, display_h))

    return True


def handle_events(state: UIState, project: Project, canvas_res: tuple[int, int]):
    running = True
    mouse_pos = pygame.mouse.get_pos()
    mouse_down = False
    key_pressed = None
    menu_state = state.render_menu

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_down = True
            if mouse_pos[0] < canvas_res[0] and state.render_mode == "edit":
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
            if state.mouse_dragging and state.selected_idx >= 0:
                dx = float(mouse_pos[0] - state.last_mouse_pos[0])
                dy = float(mouse_pos[1] - state.last_mouse_pos[1])
                conductor = project.conductors[state.selected_idx]
                conductor.position = (conductor.position[0] + dx, conductor.position[1] + dy)
                state.last_mouse_pos = mouse_pos
                state.field_dirty = True
        elif event.type == pygame.KEYDOWN:
            key_pressed = event.key
            if event.key == pygame.K_ESCAPE:
                if menu_state.is_open:
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
                state.rendered_surface = array_to_surface(state.current_render_data)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
