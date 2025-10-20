import pygame
import numpy as np
import subprocess
from pathlib import Path
from flowcol.types import Conductor
from flowcol.mask_utils import load_conductor_masks


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


def panel(screen, project, state, mouse_pos, mouse_down):
    """Draw control panel. Returns multiplier if render requested, -1 if back to edit, else None."""
    if state.render_mode == "edit":
        panel_x = project.canvas_resolution[0] + 10
    else:
        panel_x = 10
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

                if len(project.conductors) == 0:
                    canvas_w, canvas_h = project.canvas_resolution
                    new_w = max(canvas_w, mask_w)
                    new_h = max(canvas_h, mask_h)
                    project.canvas_resolution = (new_w, new_h)

                canvas_w, canvas_h = project.canvas_resolution
                pos = ((canvas_w - mask_w) // 2, (canvas_h - mask_h) // 2)
                project.conductors.append(Conductor(mask=mask, voltage=0.0, position=pos, interior_mask=interior))
                state.field_dirty = True

        y += 45

        font = pygame.font.Font(None, 22)
        title = font.render("Render:", True, (220, 220, 220))
        screen.blit(title, (panel_x, y))
        y += 30

        for mult in [1, 2, 4, 8]:
            if button(screen, panel_x, y, 85, 30, f"{mult}×", mouse_pos, mouse_down):
                action = mult
            y += 35

        y += 15
    else:
        if button(screen, panel_x, y, 180, 35, "Back to Edit", mouse_pos, mouse_down):
            action = -1
        y += 50

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
