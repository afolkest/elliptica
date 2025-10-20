import pygame
import numpy as np
import subprocess
from pathlib import Path
from flowcol.types import Conductor
from flowcol.mask_utils import load_conductor_masks

MAX_CANVAS_DIM = 8192


def button(screen, x, y, w, h, text, mouse_pos, clicked):
    """Draw button, return True if just clicked."""
    rect = pygame.Rect(x, y, w, h)
    hover = rect.collidepoint(mouse_pos)
    color = (80, 80, 80) if hover else (50, 50, 50)
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, (100, 100, 100), rect, 1)
    font = pygame.font.Font(None, 24)
    text_surf = font.render(text, True, (200, 200, 200))
    screen.blit(text_surf, (x + (w - text_surf.get_width()) // 2, y + (h - text_surf.get_height()) // 2))
    return hover and clicked


def text_input(screen, x, y, w, h, value, mouse_pos, clicked):
    """Draw simple text input box. Returns new value or None if unchanged."""
    rect = pygame.Rect(x, y, w, h)
    hover = rect.collidepoint(mouse_pos)
    color = (70, 70, 70) if hover else (40, 40, 40)
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, (100, 100, 100), rect, 1)
    font = pygame.font.Font(None, 24)
    text_surf = font.render(str(value), True, (200, 200, 200))
    screen.blit(text_surf, (x + 10, y + (h - text_surf.get_height()) // 2))
    return None


def slider(screen, x, y, w, label, value, min_v, max_v, mouse_pos, is_dragging, mouse_down):
    """Draw slider, return (new_value, is_dragging)."""
    font = pygame.font.Font(None, 20)
    label_surf = font.render(f"{label}: {value:.2f}V", True, (200, 200, 200))
    screen.blit(label_surf, (x, y - 18))

    track_rect = pygame.Rect(x, y, w, 8)
    pygame.draw.rect(screen, (40, 40, 40), track_rect)

    handle_x = x + int((value - min_v) / (max_v - min_v) * w)
    handle_rect = pygame.Rect(handle_x - 6, y - 8, 12, 24)

    hover = handle_rect.collidepoint(mouse_pos)
    if is_dragging:
        new_val = min_v + (mouse_pos[0] - x) / w * (max_v - min_v)
        new_val = np.clip(new_val, min_v, max_v)
        handle_x = x + int((new_val - min_v) / (max_v - min_v) * w)
        handle_rect.x = handle_x - 6
        pygame.draw.rect(screen, (200, 200, 200), handle_rect)
        return (new_val, pygame.mouse.get_pressed()[0])
    else:
        pygame.draw.rect(screen, (150, 150, 150) if hover else (120, 120, 120), handle_rect)
        if hover and mouse_down:
            return (value, True)

    return (value, False)


def render_menu(screen, state, mouse_pos, mouse_down, event_key):
    """Draw render config overlay. Returns multiplier if confirmed, -999 if cancelled, None otherwise."""
    screen_w, screen_h = screen.get_size()

    # Backdrop
    backdrop = pygame.Surface((screen_w, screen_h))
    backdrop.set_alpha(180)
    backdrop.fill((0, 0, 0))
    screen.blit(backdrop, (0, 0))

    # Menu panel
    menu_w, menu_h = 340, 330
    menu_x = (screen_w - menu_w) // 2
    menu_y = (screen_h - menu_h) // 2
    pygame.draw.rect(screen, (40, 40, 40), (menu_x, menu_y, menu_w, menu_h))
    pygame.draw.rect(screen, (100, 100, 100), (menu_x, menu_y, menu_w, menu_h), 2)

    y = menu_y + 20
    font = pygame.font.Font(None, 28)
    title = font.render("Render Settings", True, (220, 220, 220))
    screen.blit(title, (menu_x + (menu_w - title.get_width()) // 2, y))
    y += 50

    # Multiplier buttons
    font = pygame.font.Font(None, 22)
    label = font.render("Resolution Multiplier:", True, (200, 200, 200))
    screen.blit(label, (menu_x + 20, y))
    y += 30

    # Draw multiplier buttons with proper spacing
    button_w = 60
    total_buttons = 4
    spacing = (menu_w - 40 - total_buttons * button_w) // (total_buttons - 1)
    for i, mult in enumerate([1, 2, 4, 8]):
        btn_x = menu_x + 20 + i * (button_w + spacing)
        rect = pygame.Rect(btn_x, y, button_w, 35)
        hover = rect.collidepoint(mouse_pos)
        is_selected = (state.selected_multiplier == mult)

        if is_selected:
            color = (100, 120, 255)
        elif hover:
            color = (80, 80, 80)
        else:
            color = (50, 50, 50)

        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        font_btn = pygame.font.Font(None, 24)
        text_surf = font_btn.render(f"{mult}×", True, (200, 200, 200))
        screen.blit(text_surf, (btn_x + (button_w - text_surf.get_width()) // 2, y + (35 - text_surf.get_height()) // 2))

        if hover and mouse_down:
            state.selected_multiplier = mult

    y += 50

    # LIC passes input
    label = font.render("LIC Passes:", True, (200, 200, 200))
    screen.blit(label, (menu_x + 20, y))
    y += 30

    input_rect = pygame.Rect(menu_x + 20, y, 80, 35)
    input_hover = input_rect.collidepoint(mouse_pos)

    if input_hover and mouse_down:
        state.passes_input_focused = True

    color = (70, 70, 70) if state.passes_input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, input_rect)
    pygame.draw.rect(screen, (100, 100, 100), input_rect, 2 if state.passes_input_focused else 1)

    font_input = pygame.font.Font(None, 24)
    display_text = str(state.num_lic_passes) if state.num_lic_passes > 0 else ""
    text_surf = font_input.render(display_text, True, (200, 200, 200))
    screen.blit(text_surf, (menu_x + 30, y + (35 - text_surf.get_height()) // 2))

    # Handle keyboard input for number
    if state.passes_input_focused and event_key:
        if event_key == pygame.K_BACKSPACE:
            state.num_lic_passes = state.num_lic_passes // 10
        elif pygame.K_0 <= event_key <= pygame.K_9:
            digit = event_key - pygame.K_0
            if state.num_lic_passes == 0:
                state.num_lic_passes = digit
            else:
                new_val = state.num_lic_passes * 10 + digit
                if new_val <= 99:
                    state.num_lic_passes = new_val

    y += 55

    # Render and Cancel buttons
    action = None
    if button(screen, menu_x + 20, y, 130, 35, "Render", mouse_pos, mouse_down):
        action = state.selected_multiplier

    if button(screen, menu_x + 160, y, 130, 35, "Cancel", mouse_pos, mouse_down):
        action = -999

    return action


def panel(screen, project, state, mouse_pos, mouse_down):
    """Draw control panel. Returns: 1/2/4/8 for render, -1 for back to edit, -2 for resize, -3 for reset, -999 for cancel menu, None otherwise."""
    panel_x = project.canvas_resolution[0] + 10
    y = 10
    action = None

    if state.render_mode == "edit":
        if button(screen, panel_x, y, 180, 35, "Load Conductor", mouse_pos, mouse_down):
            result = subprocess.run([
                'osascript', '-e',
                'POSIX path of (choose file of type {"png"} with prompt "Select conductor mask")'
            ], capture_output=True, text=True)
            path = result.stdout.strip()
            if path:
                mask, interior = load_conductor_masks(path)
                mask_h, mask_w = mask.shape

                if mask_w > MAX_CANVAS_DIM or mask_h > MAX_CANVAS_DIM:
                    return action

                canvas_changed = False
                if len(project.conductors) == 0:
                    canvas_w, canvas_h = project.canvas_resolution
                    new_w = max(canvas_w, mask_w)
                    new_h = max(canvas_h, mask_h)
                    if (new_w, new_h) != project.canvas_resolution:
                        project.canvas_resolution = (new_w, new_h)
                        canvas_changed = True

                canvas_w, canvas_h = project.canvas_resolution
                pos = ((canvas_w - mask_w) / 2.0, (canvas_h - mask_h) / 2.0)
                project.conductors.append(Conductor(mask=mask, voltage=0.0, position=pos, interior_mask=interior))
                state.field_dirty = True
                if canvas_changed:
                    action = -2

        y += 45

        if button(screen, panel_x, y, 180, 35, "Render...", mouse_pos, mouse_down):
            state.render_menu_open = True

        y += 50
    else:
        if button(screen, panel_x, y, 180, 35, "Back to Edit", mouse_pos, mouse_down):
            action = -1
        y += 45

        if button(screen, panel_x, y, 180, 35, "Reset to Original", mouse_pos, mouse_down):
            action = -3
        y += 50

    if state.render_mode == "edit":
        font = pygame.font.Font(None, 22)
        title = font.render("Conductor Voltages:", True, (220, 220, 220))
        screen.blit(title, (panel_x, y))
        y += 30

        for i, conductor in enumerate(project.conductors):
            new_v, dragging = slider(
                screen, panel_x, y, 180, f"C{i+1}",
                conductor.voltage, -1.0, 1.0,
                mouse_pos, state.slider_dragging == i, mouse_down
            )
            if new_v != conductor.voltage:
                conductor.voltage = new_v
                state.field_dirty = True
            if dragging:
                state.slider_dragging = i
            elif state.slider_dragging == i and not dragging:
                state.slider_dragging = -1
            y += 40

    return action
