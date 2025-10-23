import pygame
import numpy as np
import subprocess
from flowcol.types import Conductor
from flowcol.mask_utils import load_conductor_masks
from flowcol import defaults
from flowcol.render import list_color_palettes

MAX_CANVAS_DIM = 8192
SUPERSAMPLE_CHOICES = defaults.SUPERSAMPLE_CHOICES
RESOLUTION_CHOICES = defaults.RENDER_RESOLUTION_CHOICES


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
    menu_w, menu_h = 600, 520
    menu_x = (screen_w - menu_w) // 2
    menu_y = max(40, (screen_h - menu_h) // 2)
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
    font_input = pygame.font.Font(None, 24)
    label = font.render("Render Resolution:", True, (200, 200, 200))
    screen.blit(label, (menu_x + 20, y))
    y += 30

    resolution_choices = RESOLUTION_CHOICES
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
    y += 20

    col1_x = menu_x + 30
    col2_x = menu_x + menu_w // 2 + 20
    row_height = 80
    row_y = y

    input_w = 140
    input_h = 36

    valid_stream = False
    valid_margin = False
    valid_seed = False
    valid_sigma = False
    seed_value = state.noise_seed

    # Row 1: LIC passes (left) and streamlength factor (right)
    label = font.render("LIC Passes:", True, (200, 200, 200))
    screen.blit(label, (col1_x, row_y))
    passes_rect = pygame.Rect(col1_x, row_y + 28, input_w, input_h)
    passes_hover = passes_rect.collidepoint(mouse_pos)

    if passes_hover and mouse_down:
        menu.input_focused = True
        menu.streamlength_input_focused = False
        menu.streamlength_pending_clear = False
        menu.margin_input_focused = False
        menu.seed_input_focused = False
        menu.noise_sigma_input_focused = False

    color = (70, 70, 70) if menu.input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, passes_rect)
    pygame.draw.rect(screen, (100, 100, 100), passes_rect, 2 if menu.input_focused else 1)
    passes_text = str(menu.num_passes) if menu.num_passes > 0 else ""
    passes_surf = font_input.render(passes_text, True, (200, 200, 200))
    screen.blit(passes_surf, (passes_rect.x + 8, passes_rect.y + (input_h - passes_surf.get_height()) // 2))

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

    label = font.render("Streamlength factor:", True, (200, 200, 200))
    screen.blit(label, (col2_x, row_y))
    stream_rect = pygame.Rect(col2_x, row_y + 28, input_w + 20, input_h)
    stream_hover = stream_rect.collidepoint(mouse_pos)

    if stream_hover and mouse_down:
        menu.streamlength_input_focused = True
        menu.streamlength_pending_clear = True
        menu.input_focused = False
        menu.margin_input_focused = False
        menu.seed_input_focused = False
        menu.noise_sigma_input_focused = False

    color = (70, 70, 70) if menu.streamlength_input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, stream_rect)
    pygame.draw.rect(screen, (100, 100, 100), stream_rect, 2 if menu.streamlength_input_focused else 1)
    stream_text = menu.streamlength_text if menu.streamlength_text else ""
    stream_surf = font_input.render(stream_text, True, (200, 200, 200))
    screen.blit(stream_surf, (stream_rect.x + 8, stream_rect.y + (input_h - stream_surf.get_height()) // 2))

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

    row_y += row_height

    # Row 2: Padding margin (left) and noise seed (right)
    margin_label = font.render("Padding margin:", True, (200, 200, 200))
    screen.blit(margin_label, (col1_x, row_y))
    margin_rect = pygame.Rect(col1_x, row_y + 28, input_w + 10, input_h)
    margin_hover = margin_rect.collidepoint(mouse_pos)

    if margin_hover and mouse_down:
        menu.margin_input_focused = True
        menu.margin_pending_clear = True
        menu.input_focused = False
        menu.streamlength_input_focused = False
        menu.seed_input_focused = False
        menu.noise_sigma_input_focused = False

    color = (70, 70, 70) if menu.margin_input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, margin_rect)
    pygame.draw.rect(screen, (100, 100, 100), margin_rect, 2 if menu.margin_input_focused else 1)
    margin_text = menu.margin_text if menu.margin_text else ""
    margin_surf = font_input.render(margin_text, True, (200, 200, 200))
    screen.blit(margin_surf, (margin_rect.x + 8, margin_rect.y + (input_h - margin_surf.get_height()) // 2))

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

    seed_label = font.render("Noise Seed:", True, (200, 200, 200))
    screen.blit(seed_label, (col2_x, row_y))
    seed_rect = pygame.Rect(col2_x, row_y + 28, input_w + 20, input_h)
    seed_hover = seed_rect.collidepoint(mouse_pos)

    if seed_hover and mouse_down:
        menu.seed_input_focused = True
        menu.seed_pending_clear = True
        menu.input_focused = False
        menu.streamlength_input_focused = False
        menu.margin_input_focused = False
        menu.noise_sigma_input_focused = False

    color = (70, 70, 70) if menu.seed_input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, seed_rect)
    pygame.draw.rect(screen, (100, 100, 100), seed_rect, 2 if menu.seed_input_focused else 1)
    seed_text = state.noise_seed_text if state.noise_seed_text else ""
    seed_surf = font_input.render(seed_text, True, (200, 200, 200))
    screen.blit(seed_surf, (seed_rect.x + 8, seed_rect.y + (input_h - seed_surf.get_height()) // 2))

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

    row_y += row_height

    # Row 3: Noise lowpass sigma (left column, spans right for spacing)
    sigma_label = font.render("Noise lowpass sigma:", True, (200, 200, 200))
    screen.blit(sigma_label, (col1_x, row_y))
    sigma_rect = pygame.Rect(col1_x, row_y + 28, input_w + 20, input_h)
    sigma_hover = sigma_rect.collidepoint(mouse_pos)

    if sigma_hover and mouse_down:
        menu.noise_sigma_input_focused = True
        menu.noise_sigma_pending_clear = True
        menu.input_focused = False
        menu.streamlength_input_focused = False
        menu.margin_input_focused = False
        menu.seed_input_focused = False

    color = (70, 70, 70) if menu.noise_sigma_input_focused else (40, 40, 40)
    pygame.draw.rect(screen, color, sigma_rect)
    pygame.draw.rect(screen, (100, 100, 100), sigma_rect, 2 if menu.noise_sigma_input_focused else 1)
    sigma_text = menu.noise_sigma_text if menu.noise_sigma_text else ""
    sigma_surf = font_input.render(sigma_text, True, (200, 200, 200))
    screen.blit(sigma_surf, (sigma_rect.x + 8, sigma_rect.y + (input_h - sigma_surf.get_height()) // 2))

    if menu.noise_sigma_input_focused and event_key:
        value = menu.noise_sigma_text
        if menu.noise_sigma_pending_clear:
            value = ""
            menu.noise_sigma_pending_clear = False
        if event_key == pygame.K_BACKSPACE:
            value = value[:-1]
        elif event_key in (pygame.K_PERIOD, pygame.K_KP_PERIOD):
            if '.' not in value:
                value = value + ('.' if value else '0.')
        elif pygame.K_0 <= event_key <= pygame.K_9:
            value += chr(event_key)
        menu.noise_sigma_text = value

    try:
        parsed_sigma = float(menu.noise_sigma_text)
        valid_sigma = parsed_sigma >= 0.0
        if valid_sigma:
            menu.pending_noise_sigma = parsed_sigma
    except ValueError:
        valid_sigma = False

    row_y += row_height
    y = row_y + 10

    # Render and Cancel buttons
    action = None
    render_btn_x = menu_x + 30
    cancel_btn_x = menu_x + menu_w - 160
    if (
        button(screen, render_btn_x, y, 130, 35, "Render", mouse_pos, mouse_down)
        and valid_stream
        and valid_seed
        and valid_margin
        and valid_sigma
    ):
        action = menu.selected_multiplier

    if button(screen, cancel_btn_x, y, 130, 35, "Cancel", mouse_pos, mouse_down):
        action = -999

    if valid_seed:
        state.noise_seed = seed_value
    if valid_sigma:
        state.noise_sigma = menu.pending_noise_sigma

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

    strength_y = y
    strength_w = menu_w - 40
    new_strength, dragging = slider(
        screen,
        menu_x + 20,
        strength_y,
        strength_w,
        "Strength",
        menu.strength,
        0.0,
        1.0,
        mouse_pos,
        menu.strength_dragging,
        mouse_down,
        unit="",
    )
    if new_strength != menu.strength:
        menu.strength = new_strength
    menu.strength_dragging = dragging
    y += 60

    if menu.focused_field != -1 and event_key and not menu.strength_dragging:
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
                menu.strength,
            )
            action = params
    if button(screen, menu_x + 160, y, 120, 35, "Cancel", mouse_pos, mouse_down):
        action = -999

    return action


def panel(screen, project, state, mouse_pos, mouse_down, event_key=None):
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
                project.conductors.append(Conductor(mask=mask, voltage=0.5, position=pos, interior_mask=interior))
                state.field_dirty = True
                if canvas_changed:
                    state.canvas_width_text = str(project.canvas_resolution[0])
                    state.canvas_height_text = str(project.canvas_resolution[1])
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

        font = pygame.font.Font(None, 22)
        size_label = font.render("Canvas Size (px):", True, (220, 220, 220))
        screen.blit(size_label, (panel_x, y))
        y += 30

        input_w = pygame.Rect(panel_x, y, 80, 35)
        input_h = pygame.Rect(panel_x + 100, y, 80, 35)

        for idx, rect in enumerate([input_w, input_h]):
            hover = rect.collidepoint(mouse_pos)
            if hover and mouse_down:
                state.canvas_focus = idx
                state.canvas_pending_clear = True
            color = (70, 70, 70) if state.canvas_focus == idx else (40, 40, 40)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (100, 100, 100), rect, 2 if state.canvas_focus == idx else 1)

        font_input = pygame.font.Font(None, 24)
        w_text = state.canvas_width_text if state.canvas_width_text else ""
        h_text = state.canvas_height_text if state.canvas_height_text else ""
        w_surf = font_input.render(w_text, True, (200, 200, 200))
        h_surf = font_input.render(h_text, True, (200, 200, 200))
        screen.blit(w_surf, (input_w.x + 8, input_w.y + (35 - w_surf.get_height()) // 2))
        screen.blit(h_surf, (input_h.x + 8, input_h.y + (35 - h_surf.get_height()) // 2))

        if state.canvas_focus != -1 and event_key:
            text = state.canvas_width_text if state.canvas_focus == 0 else state.canvas_height_text
            if state.canvas_pending_clear:
                text = ""
                state.canvas_pending_clear = False
            if event_key == pygame.K_BACKSPACE:
                text = text[:-1]
            elif pygame.K_0 <= event_key <= pygame.K_9:
                text += chr(event_key)
            state.canvas_width_text = text if state.canvas_focus == 0 else state.canvas_width_text
            state.canvas_height_text = text if state.canvas_focus == 1 else state.canvas_height_text

        valid_canvas = False
        try:
            width_val = int(state.canvas_width_text) if state.canvas_width_text else project.canvas_resolution[0]
            height_val = int(state.canvas_height_text) if state.canvas_height_text else project.canvas_resolution[1]
            valid_canvas = width_val > 0 and height_val > 0
        except ValueError:
            valid_canvas = False

        if button(screen, panel_x, y + 45, 80, 30, "Apply", mouse_pos, mouse_down) and valid_canvas:
            old_w, old_h = project.canvas_resolution
            new_w = int(state.canvas_width_text) if state.canvas_width_text else old_w
            new_h = int(state.canvas_height_text) if state.canvas_height_text else old_h
            if new_w > 0 and new_h > 0:
                dx = (old_w - new_w) / 2.0
                dy = (old_h - new_h) / 2.0
                for conductor in project.conductors:
                    conductor.position = (conductor.position[0] - dx, conductor.position[1] - dy)
                project.canvas_resolution = (new_w, new_h)
                state.canvas_width_text = str(new_w)
                state.canvas_height_text = str(new_h)
                state.canvas_focus = -1
                state.canvas_pending_clear = False
                state.field_dirty = True
                return -2

        y += 80
    else:
        if button(screen, panel_x, y, 180, 35, "Back to Edit", mouse_pos, mouse_down):
            action = -1
        y += 45

        if button(screen, panel_x, y, 180, 35, "Reset to Original", mouse_pos, mouse_down):
            action = -3
        y += 50

        color_label = "Color: On" if state.color_enabled else "Color: Off"
        if button(screen, panel_x, y, 180, 35, color_label, mouse_pos, mouse_down):
            state.color_enabled = not state.color_enabled
            state.downsample.dirty = True
        y += 45

        palettes = list_color_palettes()
        if palettes:
            palette_name = palettes[state.color_palette_index % len(palettes)]
            if button(screen, panel_x, y, 180, 35, f"Palette: {palette_name}", mouse_pos, mouse_down):
                state.color_palette_index = (state.color_palette_index + 1) % len(palettes)
                state.downsample.dirty = True
            y += 45

        clip_label = "Clip: Off" if state.color_clip_percent <= 0.0 else f"Clip: {state.color_clip_percent:.1f}%"
        if button(screen, panel_x, y, 180, 35, clip_label, mouse_pos, mouse_down):
            if state.color_clip_percent <= 0.0:
                state.color_clip_percent = 0.5
            elif state.color_clip_percent < 1.0:
                state.color_clip_percent = 1.0
            else:
                state.color_clip_percent = 0.0
            state.downsample.dirty = True
        y += 45

        new_contrast, contrast_dragging = slider(
            screen,
            panel_x,
            y,
            180,
            "Contrast",
            state.color_contrast,
            0.5,
            2.0,
            mouse_pos,
            state.color_contrast_dragging,
            mouse_down,
            unit="",
        )
        if new_contrast != state.color_contrast:
            state.color_contrast = new_contrast
            state.downsample.dirty = True
        state.color_contrast_dragging = contrast_dragging
        y += 45

        new_gamma, gamma_dragging = slider(
            screen,
            panel_x,
            y,
            180,
            "Gamma",
            state.color_gamma,
            0.3,
            3.0,
            mouse_pos,
            state.color_gamma_dragging,
            mouse_down,
            unit="",
        )
        if new_gamma != state.color_gamma:
            state.color_gamma = new_gamma
            state.downsample.dirty = True
        state.color_gamma_dragging = gamma_dragging
        y += 45

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
        detail_label = f"Detail HP: {'On' if state.detail_enabled else 'Off'} ({state.detail_sigma_factor:.3f})"
        if button(screen, panel_x, y, 180, 35, detail_label, mouse_pos, mouse_down):
            state.detail_input_focused = False
            if not state.detail_enabled and state.detail_sigma_factor <= 0.0:
                state.detail_sigma_factor = 0.02
                state.detail_factor_text = f"{state.detail_sigma_factor:.3f}"
            state.detail_enabled = not state.detail_enabled
            state.postprocess_dirty = True
            state.downsample.dirty = True
        y += 45

        font = pygame.font.Font(None, 22)
        label = font.render("Detail σ factor:", True, (220, 220, 220))
        screen.blit(label, (panel_x, y))
        y += 28

        factor_rect = pygame.Rect(panel_x, y, 180, 35)
        hover = factor_rect.collidepoint(mouse_pos)
        if hover and mouse_down:
            state.detail_input_focused = True
            state.detail_pending_clear = True

        color = (70, 70, 70) if state.detail_input_focused else (40, 40, 40)
        pygame.draw.rect(screen, color, factor_rect)
        pygame.draw.rect(
            screen,
            (100, 100, 100),
            factor_rect,
            2 if state.detail_input_focused else 1,
        )

        font_input = pygame.font.Font(None, 24)
        display_text = state.detail_factor_text if state.detail_factor_text else ""
        text_surf = font_input.render(display_text, True, (200, 200, 200))
        screen.blit(
            text_surf,
            (
                factor_rect.x + 8,
                factor_rect.y + (factor_rect.height - text_surf.get_height()) // 2,
            ),
        )

        if state.detail_input_focused and event_key:
            value = state.detail_factor_text
            if state.detail_pending_clear:
                value = ""
                state.detail_pending_clear = False
            if event_key == pygame.K_BACKSPACE:
                value = value[:-1]
            elif event_key in (pygame.K_PERIOD, pygame.K_KP_PERIOD):
                if '.' not in value:
                    value = value + ('.' if value else '0.')
            elif pygame.K_0 <= event_key <= pygame.K_9:
                value += chr(event_key)
            state.detail_factor_text = value

        try:
            parsed_factor = float(state.detail_factor_text) if state.detail_factor_text not in ("", ".", "-") else state.detail_sigma_factor
            parsed_factor = max(parsed_factor, 0.0)
            if not np.isclose(parsed_factor, state.detail_sigma_factor):
                state.detail_sigma_factor = parsed_factor
                state.postprocess_dirty = True
                state.downsample.dirty = True
        except ValueError:
            pass
        y += 45

        if button(screen, panel_x, y, 180, 35, "Equalize...", mouse_pos, mouse_down):
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
