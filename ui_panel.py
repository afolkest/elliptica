import pygame
import numpy as np
import subprocess
from flowcol.types import Conductor
from flowcol.mask_utils import load_conductor_masks

MAX_CANVAS_DIM = 8192
SUPERSAMPLE_CHOICES = [1.0, 1.5, 2.0, 3.0]


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


def slider(screen, x, y, w, label, value, min_v, max_v, mouse_pos, is_dragging, mouse_down, unit="V"):
    """Draw slider, return (new_value, is_dragging)."""
    font = pygame.font.Font(None, 20)
    suffix = f"{value:.2f}{unit}" if unit else f"{value:.2f}"
    label_surf = font.render(f"{label}: {suffix}", True, (200, 200, 200))
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
    menu = state.render_menu

    # Backdrop
    backdrop = pygame.Surface((screen_w, screen_h))
    backdrop.set_alpha(180)
    backdrop.fill((0, 0, 0))
    screen.blit(backdrop, (0, 0))

    # Menu panel
    menu_w, menu_h = 360, 430
    menu_x = (screen_w - menu_w) // 2
    menu_y = (screen_h - menu_h) // 2
    pygame.draw.rect(screen, (40, 40, 40), (menu_x, menu_y, menu_w, menu_h))
    pygame.draw.rect(screen, (100, 100, 100), (menu_x, menu_y, menu_w, menu_h), 2)

    y = menu_y + 20
    font = pygame.font.Font(None, 28)
    title = font.render("Render Settings", True, (220, 220, 220))
    screen.blit(title, (menu_x + (menu_w - title.get_width()) // 2, y))
    y += 50

    label = font.render("Supersample Factor:", True, (200, 200, 200))
    screen.blit(label, (menu_x + 20, y))
    y += 30

    super_button_w = 70
    total_super = len(SUPERSAMPLE_CHOICES)
    if state.supersample_index >= total_super:
        state.supersample_index = 0
        state.supersample_factor = SUPERSAMPLE_CHOICES[0]
    spacing = (menu_w - 40 - total_super * super_button_w) // max(1, (total_super - 1)) if total_super > 1 else 0
    for idx, factor in enumerate(SUPERSAMPLE_CHOICES):
        btn_x = menu_x + 20 + idx * (super_button_w + spacing)
        rect = pygame.Rect(btn_x, y, super_button_w, 35)
        hover = rect.collidepoint(mouse_pos)
        selected = (state.supersample_index == idx)

        if selected:
            color = (100, 200, 150)
        elif hover:
            color = (80, 80, 80)
        else:
            color = (50, 50, 50)

        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        font_btn = pygame.font.Font(None, 24)
        text_surf = font_btn.render(f"{factor:.1f}×", True, (200, 200, 200))
        screen.blit(text_surf, (btn_x + (super_button_w - text_surf.get_width()) // 2, y + (35 - text_surf.get_height()) // 2))

        if hover and mouse_down:
            state.supersample_index = idx
            state.supersample_factor = factor

    y += 50

    # Multiplier buttons
    font = pygame.font.Font(None, 22)
    label = font.render("Render Resolution:", True, (200, 200, 200))
    screen.blit(label, (menu_x + 20, y))
    y += 30

    resolution_choices = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    button_w = 70
    button_h = 35
    columns = 3
    spacing = 15
    rows = (len(resolution_choices) + columns - 1) // columns

    for idx, mult in enumerate(resolution_choices):
        row = idx // columns
        col = idx % columns
        btn_x = menu_x + 20 + col * (button_w + spacing)
        btn_y = y + row * (button_h + 10)
        rect = pygame.Rect(btn_x, btn_y, button_w, button_h)
        hover = rect.collidepoint(mouse_pos)
        is_selected = abs(menu.selected_multiplier - mult) < 1e-6

        if is_selected:
            color = (100, 120, 255)
        elif hover:
            color = (80, 80, 80)
        else:
            color = (50, 50, 50)

        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        font_btn = pygame.font.Font(None, 22)
        text_surf = font_btn.render(f"{mult:g}×", True, (200, 200, 200))
        screen.blit(text_surf, (btn_x + (button_w - text_surf.get_width()) // 2, btn_y + (button_h - text_surf.get_height()) // 2))

        if hover and mouse_down:
            menu.selected_multiplier = mult

    y += rows * (button_h + 10)
    y += 10

    # LIC passes input
    label = font.render("LIC Passes:", True, (200, 200, 200))
    screen.blit(label, (menu_x + 20, y))
    y += 30

    input_rect = pygame.Rect(menu_x + 20, y, 80, 35)
    input_hover = input_rect.collidepoint(mouse_pos)

    if input_hover and mouse_down:
        menu.input_focused = True
        menu.streamlength_input_focused = False
        menu.streamlength_pending_clear = False
        menu.margin_input_focused = False
        menu.seed_input_focused = False

    color = (70, 70, 70) if menu.input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, input_rect)
    pygame.draw.rect(screen, (100, 100, 100), input_rect, 2 if menu.input_focused else 1)

    font_input = pygame.font.Font(None, 24)
    display_text = str(menu.num_passes) if menu.num_passes > 0 else ""
    text_surf = font_input.render(display_text, True, (200, 200, 200))
    screen.blit(text_surf, (menu_x + 30, y + (35 - text_surf.get_height()) // 2))

    # Handle keyboard input for number
    if menu.input_focused and event_key:
        if event_key == pygame.K_BACKSPACE:
            menu.num_passes = menu.num_passes // 10
        elif pygame.K_0 <= event_key <= pygame.K_9:
            digit = event_key - pygame.K_0
            if menu.num_passes == 0:
                menu.num_passes = digit
            else:
                new_val = menu.num_passes * 10 + digit
                if new_val <= 99:
                    menu.num_passes = new_val

    y += 45

    label = font.render("Streamlength factor:", True, (200, 200, 200))
    screen.blit(label, (menu_x + 20, y))
    y += 30

    stream_rect = pygame.Rect(menu_x + 20, y, 120, 35)
    stream_hover = stream_rect.collidepoint(mouse_pos)

    if stream_hover and mouse_down:
        menu.streamlength_input_focused = True
        menu.streamlength_pending_clear = True
        menu.input_focused = False
        menu.margin_input_focused = False
        menu.seed_input_focused = False

    color = (70, 70, 70) if menu.streamlength_input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, stream_rect)
    pygame.draw.rect(
        screen,
        (100, 100, 100),
        stream_rect,
        2 if menu.streamlength_input_focused else 1,
    )

    stream_text = menu.streamlength_text
    stream_display = stream_text if stream_text else ""
    text_surf = font_input.render(stream_display, True, (200, 200, 200))
    screen.blit(
        text_surf,
        (stream_rect.x + 8, stream_rect.y + (35 - text_surf.get_height()) // 2),
    )

    valid_stream = False
    valid_seed = True
    seed_value = state.noise_seed

    if menu.streamlength_input_focused and event_key:
        value = menu.streamlength_text
        if menu.streamlength_pending_clear:
            value = ""
            menu.streamlength_pending_clear = False
        if event_key == pygame.K_BACKSPACE:
            value = value[:-1]
        elif event_key in (pygame.K_PERIOD, pygame.K_KP_PERIOD):
            if '.' not in value:
                value = value + ('.' if value else '0.')
        elif pygame.K_0 <= event_key <= pygame.K_9:
            value += chr(event_key)
        menu.streamlength_text = value

    try:
        parsed_stream = float(menu.streamlength_text)
        valid_stream = parsed_stream > 0.0
        if valid_stream:
            menu.pending_streamlength_factor = parsed_stream
    except ValueError:
        valid_stream = False

    y += 45

    margin_label = font.render("Padding margin:", True, (200, 200, 200))
    screen.blit(margin_label, (menu_x + 20, y))
    y += 30

    margin_rect = pygame.Rect(menu_x + 20, y, 120, 35)
    margin_hover = margin_rect.collidepoint(mouse_pos)

    if margin_hover and mouse_down:
        menu.margin_input_focused = True
        menu.margin_pending_clear = True
        menu.input_focused = False
        menu.streamlength_input_focused = False
        menu.seed_input_focused = False

    color = (70, 70, 70) if menu.margin_input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, margin_rect)
    pygame.draw.rect(
        screen,
        (100, 100, 100),
        margin_rect,
        2 if menu.margin_input_focused else 1,
    )

    margin_text = menu.margin_text
    margin_display = margin_text if margin_text else ""
    margin_surf = font_input.render(margin_display, True, (200, 200, 200))
    screen.blit(
        margin_surf,
        (margin_rect.x + 8, margin_rect.y + (35 - margin_surf.get_height()) // 2),
    )

    valid_margin = True

    if menu.margin_input_focused and event_key:
        value = menu.margin_text
        if menu.margin_pending_clear:
            value = ""
            menu.margin_pending_clear = False
        if event_key == pygame.K_BACKSPACE:
            value = value[:-1]
        elif event_key in (pygame.K_PERIOD, pygame.K_KP_PERIOD):
            if '.' not in value:
                value = value + ('.' if value else '0.')
        elif pygame.K_0 <= event_key <= pygame.K_9:
            value += chr(event_key)
        menu.margin_text = value

    try:
        parsed_margin = float(menu.margin_text)
        valid_margin = parsed_margin >= 0.0
        if valid_margin:
            menu.pending_margin_factor = parsed_margin
    except ValueError:
        valid_margin = False

    y += 45

    seed_label = font.render("Noise Seed:", True, (200, 200, 200))
    screen.blit(seed_label, (menu_x + 20, y))
    y += 30

    seed_rect = pygame.Rect(menu_x + 20, y, 120, 35)
    seed_hover = seed_rect.collidepoint(mouse_pos)

    if seed_hover and mouse_down:
        menu.seed_input_focused = True
        menu.seed_pending_clear = True
        menu.input_focused = False
        menu.streamlength_input_focused = False
        menu.margin_input_focused = False

    color = (70, 70, 70) if menu.seed_input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, seed_rect)
    pygame.draw.rect(
        screen,
        (100, 100, 100),
        seed_rect,
        2 if menu.seed_input_focused else 1,
    )

    seed_text = state.noise_seed_text
    seed_display = seed_text if seed_text else ""
    seed_surf = font_input.render(seed_display, True, (200, 200, 200))
    screen.blit(
        seed_surf,
        (seed_rect.x + 8, seed_rect.y + (35 - seed_surf.get_height()) // 2),
    )

    if menu.seed_input_focused and event_key:
        value = state.noise_seed_text
        if menu.seed_pending_clear:
            value = ""
            menu.seed_pending_clear = False
        if event_key == pygame.K_BACKSPACE:
            value = value[:-1]
        elif event_key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            if value == "":
                value = "-"
        elif pygame.K_0 <= event_key <= pygame.K_9:
            value += chr(event_key)
        state.noise_seed_text = value

    try:
        seed_value = int(state.noise_seed_text) if state.noise_seed_text not in ("", "-") else 0
        valid_seed = True
    except ValueError:
        valid_seed = False

    y += 55

    # Render and Cancel buttons
    action = None
    if (
        button(screen, menu_x + 20, y, 130, 35, "Render", mouse_pos, mouse_down)
        and valid_stream
        and valid_seed
        and valid_margin
    ):
        action = menu.selected_multiplier

    if button(screen, menu_x + 160, y, 130, 35, "Cancel", mouse_pos, mouse_down):
        action = -999

    if valid_seed:
        state.noise_seed = seed_value

    return action


def highpass_menu(screen, state, mouse_pos, mouse_down, event_key):
    """Draw high-pass + CLAHE menu. Returns tuple of params or -999 if cancelled."""
    screen_w, screen_h = screen.get_size()
    menu = state.highpass_menu

    backdrop = pygame.Surface((screen_w, screen_h))
    backdrop.set_alpha(180)
    backdrop.fill((0, 0, 0))
    screen.blit(backdrop, (0, 0))

    menu_w, menu_h = 360, 520
    menu_x = (screen_w - menu_w) // 2
    menu_y = (screen_h - menu_h) // 2
    pygame.draw.rect(screen, (40, 40, 40), (menu_x, menu_y, menu_w, menu_h))
    pygame.draw.rect(screen, (100, 100, 100), (menu_x, menu_y, menu_w, menu_h), 2)

    y = menu_y + 20
    font = pygame.font.Font(None, 28)
    title = font.render("LIC Enhancement", True, (220, 220, 220))
    screen.blit(title, (menu_x + (menu_w - title.get_width()) // 2, y))
    y += 50

    fields = [
        ("Gaussian sigma factor", "sigma_factor_text", True, 0, 140),
        ("Clip limit", "clip_text", True, 1, 140),
        ("Kernel rows", "kernel_rows_text", False, 2, 140),
        ("Kernel cols", "kernel_cols_text", False, 3, 140),
        ("Histogram bins", "num_bins_text", False, 4, 160),
    ]

    font_label = pygame.font.Font(None, 22)
    font_input = pygame.font.Font(None, 24)

    for label_text, attr_name, is_float, field_idx, box_width in fields:
        value_text = getattr(menu, attr_name)
        label = font_label.render(f"{label_text}:", True, (200, 200, 200))
        screen.blit(label, (menu_x + 20, y))
        y += 26

        input_rect = pygame.Rect(menu_x + 20, y, box_width, 36)
        hover = input_rect.collidepoint(mouse_pos)
        if hover and mouse_down:
            menu.focused_field = field_idx
            menu.pending_clear = field_idx

        focused = (menu.focused_field == field_idx)
        color = (70, 70, 70) if focused else (40, 40, 40)
        pygame.draw.rect(screen, color, input_rect)
        pygame.draw.rect(screen, (100, 100, 100), input_rect, 2 if focused else 1)

        text_surf = font_input.render(value_text, True, (200, 200, 200))
        screen.blit(text_surf, (input_rect.x + 8, input_rect.y + (36 - text_surf.get_height()) // 2))

        y += 46

    if menu.focused_field != -1 and event_key:
        attr = fields[menu.focused_field]
        name = attr[1]
        is_float = attr[2]
        value = getattr(menu, name)
        if menu.pending_clear == menu.focused_field:
            value = ""
            menu.pending_clear = -1
        if event_key == pygame.K_BACKSPACE:
            value = value[:-1]
        elif is_float and (event_key == pygame.K_PERIOD or event_key == pygame.K_KP_PERIOD):
            if '.' not in value:
                value = value + ('.' if value else '0.')
        elif pygame.K_0 <= event_key <= pygame.K_9:
            value += chr(event_key)
        setattr(menu, name, value)

    y += 10

    valid_sigma = menu.sigma_factor_text not in ("", ".")
    valid_clip = menu.clip_text not in ("", ".")
    valid_rows = menu.kernel_rows_text.isdigit()
    valid_cols = menu.kernel_cols_text.isdigit()
    valid_bins = menu.num_bins_text.isdigit()
    action = None

    if button(screen, menu_x + 20, y, 120, 35, "Apply", mouse_pos, mouse_down):
        if all([valid_sigma, valid_clip, valid_rows, valid_cols, valid_bins]):
            params = (
                float(menu.sigma_factor_text),
                float(menu.clip_text),
                int(menu.kernel_rows_text),
                int(menu.kernel_cols_text),
                int(menu.num_bins_text),
            )
            action = params
    if button(screen, menu_x + 160, y, 120, 35, "Cancel", mouse_pos, mouse_down):
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
            menu = state.render_menu
            menu.is_open = True
            menu.input_focused = False
            menu.streamlength_input_focused = False
            menu.streamlength_pending_clear = False
            menu.streamlength_text = f"{project.streamlength_factor:.4f}"
            menu.pending_streamlength_factor = project.streamlength_factor
            menu.seed_input_focused = False
            menu.seed_pending_clear = False
            state.noise_seed_text = str(state.noise_seed)
            menu.margin_input_focused = False
            menu.margin_pending_clear = False
            menu.margin_text = f"{state.margin_factor:.3f}"
            menu.pending_margin_factor = state.margin_factor

        y += 50
    else:
        if button(screen, panel_x, y, 180, 35, "Back to Edit", mouse_pos, mouse_down):
            action = -1
        y += 45

        if button(screen, panel_x, y, 180, 35, "Reset to Original", mouse_pos, mouse_down):
            action = -3
        y += 50

        font = pygame.font.Font(None, 22)
        title = font.render("Post-processing:", True, (220, 220, 220))
        screen.blit(title, (panel_x, y))
        y += 35

        down = state.downsample
        new_sigma, dragging = slider(
            screen,
            panel_x,
            y,
            180,
            "Blur σ",
            down.sigma_factor,
            0.1,
            2.0,
            mouse_pos,
            down.dragging,
            mouse_down,
            unit="",
        )
        if new_sigma != down.sigma_factor:
            down.sigma_factor = new_sigma
            down.sigma_text = f"{new_sigma:.2f}"
            down.dirty = True
        down.dragging = dragging
        y += 45

        if button(screen, panel_x, y, 180, 35, "Enhance (HP+CLAHE)...", mouse_pos, mouse_down):
            menu = state.highpass_menu
            menu.is_open = True
            menu.focused_field = 0
            menu.pending_clear = 0
            menu.sigma_factor_text = f"{menu.sigma_factor:.4f}"
            menu.clip_text = f"{menu.clip_limit:.4f}"
            menu.kernel_rows_text = str(menu.kernel_rows)
            menu.kernel_cols_text = str(menu.kernel_cols)
            menu.num_bins_text = str(menu.num_bins)
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
