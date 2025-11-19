import pygame
import sys
import os
import json
import random

# Try importing VLM; if unavailable, set vlm to None and fallback will be used
try:
    from vlm import VLM

    vlm = VLM()
except Exception as e:
    print("Warning: VLM import/initialization failed:", e)
    vlm = None

pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 4
DOT_RADIUS = 10
LINE_WIDTH = 10
MARGIN = 100
SPACING = (WIDTH - 2 * MARGIN) // GRID_SIZE

# Colors
BG_COLOR = (255, 255, 255)
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER = (150, 150, 150)
BUTTON_TEXT_COLOR = (255, 255, 255)
DOT_COLOR = (200, 200, 200)
PLAYER_COLORS = [(200, 0, 0), (0, 0, 200)]
PLAYER_BOXES = [(255, 150, 150), (150, 150, 255)]

# Setup window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dots and Boxes")

# Game state
horizontal_lines = [[None] * GRID_SIZE for _ in range(GRID_SIZE + 1)]
vertical_lines = [[None] * (GRID_SIZE + 1) for _ in range(GRID_SIZE)]
boxes = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
scores = [0, 0]
current_player = 0
total_lines = (GRID_SIZE + 1) * GRID_SIZE + GRID_SIZE * (GRID_SIZE + 1)
lines_drawn = 0

# Screenshot tracking
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "raw")

def get_next_game_id():
    """Find the next available game ID."""
    if not os.path.exists(DATA_DIR):
        return 1
    existing_games = [d for d in os.listdir(DATA_DIR) if d.startswith("game_") and os.path.isdir(os.path.join(DATA_DIR, d))]
    if not existing_games:
        return 1
    game_numbers = []
    for game_dir in existing_games:
        try:
            num = int(game_dir.split("_")[1])
            game_numbers.append(num)
        except (IndexError, ValueError):
            continue
    return max(game_numbers) + 1 if game_numbers else 1

game_id = get_next_game_id()
turn = 0

def run_menu():
    clock = pygame.time.Clock() # to make the program not use 100% CPU
    title_font = pygame.font.Font(None, 72)
    btn_font = pygame.font.Font(None, 48)

    btn_width, btn_height = 300, 70
    btn_x = WIDTH // 2 - btn_width // 2 # center horizontally
    spacing_between = 20
    total_height = btn_height * 3 + spacing_between * 2
    start_y = HEIGHT // 2 - total_height // 2

    btn_y1 = start_y
    btn_y2 = start_y + btn_height + spacing_between
    btn_y3 = start_y + 2 * (btn_height + spacing_between)

    rect_1v1 = pygame.Rect(btn_x, btn_y1, btn_width, btn_height)
    rect_1vAI = pygame.Rect(btn_x, btn_y2, btn_width, btn_height)
    rect_exit = pygame.Rect(btn_x, btn_y3, btn_width, btn_height)

    # Text surfaces and positions
    title_surf = title_font.render("Dots and Boxes", True, (0, 0, 0))
    title_pos = (WIDTH // 2 - title_surf.get_width() // 2, 80)

    text_1v1 = btn_font.render("1 vs 1", True, BUTTON_TEXT_COLOR)
    text_1vAI = btn_font.render("1 vs AI", True, BUTTON_TEXT_COLOR)
    text_exit = btn_font.render("Exit", True, BUTTON_TEXT_COLOR)

    text_1v1_rect = text_1v1.get_rect(center=rect_1v1.center)
    text_1vAI_rect = text_1vAI.get_rect(center=rect_1vAI.center)
    text_exit_rect = text_exit.get_rect(center=rect_exit.center)

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # close window
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if rect_1v1.collidepoint(event.pos):
                    return '1v1'
                if rect_1vAI.collidepoint(event.pos):
                    return '1vAI'
                if rect_exit.collidepoint(event.pos):
                    return None

        # Draw menu
        screen.fill(BG_COLOR)
        screen.blit(title_surf, title_pos)

        # draw buttons (change color on hover)
        pygame.draw.rect(screen, BUTTON_HOVER if rect_1v1.collidepoint(mouse_pos) else BUTTON_COLOR, rect_1v1)
        pygame.draw.rect(screen, BUTTON_HOVER if rect_1vAI.collidepoint(mouse_pos) else BUTTON_COLOR, rect_1vAI)
        pygame.draw.rect(screen, BUTTON_HOVER if rect_exit.collidepoint(mouse_pos) else BUTTON_COLOR, rect_exit)

        screen.blit(text_1v1, text_1v1_rect.topleft)
        screen.blit(text_1vAI, text_1vAI_rect.topleft)
        screen.blit(text_exit, text_exit_rect.topleft)

        pygame.display.flip()
        clock.tick(60)

def draw_board():
    screen.fill(BG_COLOR)
    font = pygame.font.Font(None, 36)

    # Draw boxes
    for i in range(GRID_SIZE): # go through rows
        for j in range(GRID_SIZE): # go through columns
            if boxes[i][j] is not None:
                x = MARGIN + j * SPACING
                y = MARGIN + i * SPACING
                pygame.draw.rect(screen, PLAYER_BOXES[boxes[i][j]], (x, y, SPACING, SPACING))

    # Draw horizontal lines
    for i in range(GRID_SIZE + 1):
        for j in range(GRID_SIZE):
            player = horizontal_lines[i][j]
            if player is not None:
                x1 = MARGIN + j * SPACING
                y1 = MARGIN + i * SPACING
                color = PLAYER_COLORS[player]
                pygame.draw.line(screen, color, (x1, y1), (x1 + SPACING, y1), LINE_WIDTH)

    # Draw vertical lines
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE + 1):
            player = vertical_lines[i][j]
            if player is not None:
                x1 = MARGIN + j * SPACING
                y1 = MARGIN + i * SPACING
                color = PLAYER_COLORS[player]
                pygame.draw.line(screen, color, (x1, y1), (x1, y1 + SPACING), LINE_WIDTH)

    # Draw dots
    for i in range(GRID_SIZE + 1):
        for j in range(GRID_SIZE + 1):
            x = MARGIN + j * SPACING
            y = MARGIN + i * SPACING
            pygame.draw.circle(screen, DOT_COLOR, (x, y), DOT_RADIUS)

    # Display scores
    score_text = f"Red: {scores[0]}   Blue: {scores[1]}"
    text_surface = font.render(score_text, True, (0, 0, 0))
    screen.blit(text_surface, text_surface.get_rect(centerx=WIDTH // 2, top=20))

    # Display current turn
    turn_text = f"{'Red' if current_player == 0 else 'Blue'}'s Turn"
    turn_color = PLAYER_COLORS[current_player]
    turn_surface = font.render(turn_text, True, turn_color)
    screen.blit(turn_surface, turn_surface.get_rect(centerx=WIDTH // 2, top=55))

    pygame.display.flip()

def get_line_clicked(pos):
    x, y = pos
    for i in range(GRID_SIZE + 1):
        for j in range(GRID_SIZE):
            x1 = MARGIN + j * SPACING
            y1 = MARGIN + i * SPACING
            if abs(y - y1) < 10 and x1 < x < x1 + SPACING and horizontal_lines[i][j] is None:
                return ('h', i, j)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE + 1):
            x1 = MARGIN + j * SPACING
            y1 = MARGIN + i * SPACING
            if abs(x - x1) < 10 and y1 < y < y1 + SPACING and vertical_lines[i][j] is None:
                return ('v', i, j)
    return None

def apply_move(line_type, i, j, player): # adds a line to the counter
    global lines_drawn
    if line_type == 'h':
        horizontal_lines[i][j] = player
    else:
        vertical_lines[i][j] = player
    lines_drawn += 1

def check_completed_boxes():
    completed = False
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if boxes[i][j] is None:
                if (horizontal_lines[i][j] is not None and
                        horizontal_lines[i + 1][j] is not None and
                        vertical_lines[i][j] is not None and
                        vertical_lines[i][j + 1] is not None):
                    boxes[i][j] = current_player
                    scores[current_player] += 1
                    completed = True
    return completed

def get_available_moves(): # check possible moves

    moves = []
    for i in range(GRID_SIZE + 1):
        for j in range(GRID_SIZE):
            if horizontal_lines[i][j] is None:
                moves.append(('h', i, j))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE + 1):
            if vertical_lines[i][j] is None:
                moves.append(('v', i, j))
    return moves

def count_box_edges(bi, bj): # useful for Ai
    count = 0
    if horizontal_lines[bi][bj] is not None:
        count += 1
    if horizontal_lines[bi + 1][bj] is not None:
        count += 1
    if vertical_lines[bi][bj] is not None:
        count += 1
    if vertical_lines[bi][bj + 1] is not None:
        count += 1
    return count

def will_create_third_edge(line_type, i, j):
    # Temporarily place the line
    if line_type == 'h':
        horizontal_lines[i][j] = current_player
    else:
        vertical_lines[i][j] = current_player

    creates_third = False
    # Check boxes that could be affected by this line
    affected_boxes = []

    if line_type == 'h':
        if i > 0:
            affected_boxes.append((i - 1, j))
        if i < GRID_SIZE:
            affected_boxes.append((i, j))
    else:  # vertical
        if j > 0:
            affected_boxes.append((i, j - 1))
        if j < GRID_SIZE:
            affected_boxes.append((i, j))

    for bi, bj in affected_boxes:
        if 0 <= bi < GRID_SIZE and 0 <= bj < GRID_SIZE:
            if boxes[bi][bj] is None and count_box_edges(bi, bj) == 3:
                creates_third = True
                break

    # Remove the temporary line
    if line_type == 'h':
        horizontal_lines[i][j] = None
    else:
        vertical_lines[i][j] = None

    return creates_third

def ai_move(): # Here the bot choose what to do with simple code
    import random
    available = get_available_moves()
    if not available:
        return None

    # Strategy 1: Complete a box if possible
    for move in available:
        line_type, i, j = move
        # Temporarily place the line
        if line_type == 'h':
            horizontal_lines[i][j] = current_player
        else:
            vertical_lines[i][j] = current_player

        # Check if any box is completed
        completes = False
        for bi in range(GRID_SIZE):
            for bj in range(GRID_SIZE):
                if boxes[bi][bj] is None:
                    if (horizontal_lines[bi][bj] is not None and
                            horizontal_lines[bi + 1][bj] is not None and
                            vertical_lines[bi][bj] is not None and
                            vertical_lines[bi][bj + 1] is not None):
                        completes = True
                        break
            if completes:
                break

        # Remove the temporary line
        if line_type == 'h':
            horizontal_lines[i][j] = None
        else:
            vertical_lines[i][j] = None

        if completes:
            return move

    # Strategy 2: Avoid creating third edges
    safe_moves = []
    for move in available:
        if not will_create_third_edge(move[0], move[1], move[2]):
            safe_moves.append(move)

    if safe_moves:
        return random.choice(safe_moves)

    # Strategy 3: If all moves create third edges, use random one
    # In future maybe we improve this so the AI is more intelligent
    return random.choice(available)

def save_turn_screenshot():
    """Save screenshot of current game state."""
    global turn
    turn += 1
    game_dir = os.path.join(DATA_DIR, f"game_{game_id}")
    os.makedirs(game_dir, exist_ok=True)
    screenshot_path = os.path.join(game_dir, f"turn_{turn:03}.png")
    pygame.image.save(screen, screenshot_path)
    print(f"Screenshot saved: {screenshot_path}")
    return screenshot_path

def save_game_state_json(last_move=None):
    game_dir = os.path.join(DATA_DIR, f"game_{game_id}")
    os.makedirs(game_dir, exist_ok=True)
    state = get_game_state(last_move)
    state_path = os.path.join(game_dir, f"turn_{turn:03}.json")

    def convert(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(convert(state), f, indent=2)
    print(f"Game state saved: {state_path}")
    return state_path


def get_game_state(last_move=None):
    # Return current game state
    horizontal_copy = [row[:] for row in horizontal_lines]
    vertical_copy = [row[:] for row in vertical_lines]
    boxes_copy = [row[:] for row in boxes]

    return {
        "game_id": game_id,
        "turn": turn,
        "current_player": current_player,
        "scores": {"red": scores[0], "blue": scores[1]},
        "boxes": boxes_copy,
        "horizontal_lines": horizontal_copy,
        "vertical_lines": vertical_copy,
        "remaining_lines": total_lines - lines_drawn,
        "available_moves": get_available_moves(),
        "last_move": last_move,
    }


def safe_vlm_predict_move(img_path, candidate_moves):
    if vlm is None:
        return random.choice(candidate_moves) if candidate_moves else None

    try:
        move = vlm.predict_move(img_path, candidate_moves)
    except Exception as e:
        print("VLM predict error:", e)
        move = None

    if is_valid_move(move):
        return move

    if isinstance(move, str):
        parts = [p.strip() for p in move.replace(",", " ").split()]
        if len(parts) == 3 and parts[0] in ("h", "v"):
            try:
                mi, mj = int(parts[1]), int(parts[2])
                parsed = (parts[0], mi, mj)
                if is_valid_move(parsed):
                    return parsed
            except Exception:
                pass

    print("VLM returned invalid move.")
    return random.choice(candidate_moves) if candidate_moves else None

def is_valid_move(move):
    if move is None:
        return False
    try:
        t, i, j = move
    except Exception:
        return False
    if t not in ("h", "v"):
        return False
    if t == "h":
        if 0 <= i <= GRID_SIZE and 0 <= j < GRID_SIZE:
            return horizontal_lines[i][j] is None
        return False
    else:
        if 0 <= i < GRID_SIZE and 0 <= j <= GRID_SIZE:
            return vertical_lines[i][j] is None
        return False


# Show start menu at start
mode = run_menu()
if not mode:
    pygame.quit()
    sys.exit()

# Main game loop
running = True
clock = pygame.time.Clock()

# Draw initial board and save screenshot + JSON
draw_board()
last_screenshot = save_turn_screenshot()
save_game_state_json(last_move=None)

while running:
    draw_board()

    # AI turn
    if mode == '1vAI' and current_player == 1 and lines_drawn < total_lines:
        pygame.time.wait(300)  # delay for better experience
        ai_move_result = ai_move()
        if ai_move_result:
            line_type, i, j = ai_move_result
            apply_move(line_type, i, j, current_player)

            # Check for completed boxes
            completed = check_completed_boxes()
            draw_board()
            save_turn_screenshot()

            # Switch player only if no box was completed
            if not completed:
                current_player = 1 - current_player

    # VLM turn
    if mode == 'BotvVLM' and current_player == 1 and lines_drawn < total_lines:
        pygame.time.wait(300)
        available = get_available_moves()

        vlm_move = safe_vlm_predict_move(last_screenshot, available)

        if vlm_move:
            line_type, i, j = vlm_move
            apply_move(line_type, i, j, current_player)

            completed = check_completed_boxes()
            draw_board()
            last_screenshot = save_turn_screenshot()
            save_game_state_json(last_move=(line_type, i, j))

            if not completed:
                current_player = 1 - current_player
        else:
            fallback = ai_move()
            if fallback:
                line_type, i, j = fallback
                apply_move(line_type, i, j, current_player)
                completed = check_completed_boxes()
                draw_board()
                last_screenshot = save_turn_screenshot()
                save_game_state_json(last_move=(line_type, i, j))
                if not completed:
                    current_player = 1 - current_player

    # PLayer turn
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # In 1vAI only allow clicks when it's player turn
            if mode == '1v1' or (mode == '1vAI' and current_player == 0):
                line = get_line_clicked(event.pos)
                if line:
                    line_type, i, j = line
                    apply_move(line_type, i, j, current_player)

                    # Check for completed boxes FIRST (this fills the boxes)
                    completed = check_completed_boxes()

                    # THEN draw the board with colored boxes and save screenshot
                    draw_board()
                    save_turn_screenshot()

                    # Switch player only if no box was completed
                    if not completed:
                        current_player = 1 - current_player

    # Check for game over
    if lines_drawn == total_lines:
        draw_board()
        pygame.time.wait(1000)
        print("Game Over!")
        print(f"Final Scores → Red: {scores[0]}, Blue: {scores[1]}")
        last_screenshot = save_turn_screenshot()
        save_game_state_json(last_move=None)
        running = False

    clock.tick(30)  # Limit to 30 FPS so its run better

pygame.quit()
sys.exit()