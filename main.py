import pygame
import sys

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
    score_text = "Red: " + str(scores[0]) + "   Blue: " + str(scores[1])
    text_surface = font.render(score_text, True, (0, 0, 0))
    screen.blit(text_surface, (WIDTH // 2 - 100, 20))

    # Display current turn
    turn_text = ("Red" if current_player == 0 else "Blue") + "'s Turn"
    turn_color = PLAYER_COLORS[current_player]
    turn_surface = font.render(turn_text, True, turn_color)
    screen.blit(turn_surface, (WIDTH // 2 - 60, 55))

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

def count_remaining_lines(): # count for game over, useful if we implement different grid sizes
    return sum(line is None for row in horizontal_lines + vertical_lines for line in row)

# Show start menu at start
mode = run_menu()
if not mode:
    pygame.quit()
    sys.exit()

# Main game loop
running = True
while running:
    draw_board()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            line = get_line_clicked(event.pos)
            if line:
                line_type, i, j = line
                if line_type == 'h':
                    horizontal_lines[i][j] = current_player
                else:
                    vertical_lines[i][j] = current_player

                if not check_completed_boxes():
                    current_player = 1 - current_player

    # Check for game over
    if count_remaining_lines() == 0:
        draw_board()
        pygame.time.wait(1000)
        print("Game Over!")
        print(f"Final Scores → Red: {scores[0]}, Blue: {scores[1]}")
        running = False


pygame.quit()
sys.exit()
