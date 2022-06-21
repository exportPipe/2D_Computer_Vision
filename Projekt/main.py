import ctypes
import tkinter.filedialog

import pygame
from utils import *

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Handwriting Recognition")


def init_grid(rows, cols, color):
    grid_ = []
    for i in range(rows):
        grid_.append([])
        for j in range(cols):
            grid_[i].append(color)
    return grid_


def draw_grid(win, grid_):
    for i, row_ in enumerate(grid_):
        for j, pixel in enumerate(row_):
            pygame.draw.rect(win, pixel, (j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

    if DRAW_GRID_LINES:
        for i in range(0, ROWS + 1, 28):
            pygame.draw.line(win, BLACK, (0, i * PIXEL_SIZE), (WIDTH, i * PIXEL_SIZE))

        for i in range(0, COLS + 1, 28):
            pygame.draw.line(win, BLACK, (i * PIXEL_SIZE, 0), (i * PIXEL_SIZE, HEIGHT - TOOLBAR_HEIGHT))


def draw(win, grid_, buttons_):
    win.fill(BG_COLOR)
    draw_grid(win, grid_)

    for button_ in buttons_:
        button_.draw(win)
    pygame.display.update()


def get_row_col_from_pos(position):
    x, y = position
    row_idx = y // PIXEL_SIZE
    col_idx = x // PIXEL_SIZE

    if row_idx >= ROWS:
        raise IndexError

    return row_idx, col_idx


def draw_pixel(row_, col_, size):
    for u in range(row_ - size + 1, row + size):
        for v in range(col_ - size + 1, col + size):
            grid[u][v] = drawing_color


def prompt_file():
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    top = tkinter.Tk()
    top.withdraw()
    file_name = tkinter.filedialog.askopenfilename(parent=top)
    top.destroy()
    return file_name


run = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS, WHITE)
drawing_color = BLACK
drawing_size = 2

button_y = HEIGHT - TOOLBAR_HEIGHT / 2 - 25
buttons = [
    Button(10, button_y, 100, 50, WHITE, "Clear", BLACK),
    Button(120, button_y, 100, 50, WHITE, "Spot"),
    Button(230, button_y, 530, 50, WHITE),
    Button(10, button_y + 60, 100, 50, WHITE, "Open...", BLACK)
]


while run:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        # LEFT CLICK
        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            try:
                row, col = get_row_col_from_pos(pos)
                draw_pixel(row, col, drawing_size)
            except IndexError:
                for idx, button in enumerate(buttons):
                    if not button.clicked(pos):
                        continue
                    if button.text == "Clear":
                        grid = init_grid(ROWS, COLS, BG_COLOR)
                    if button.text == "Spot":
                        buttons[idx + 1].text = get_text(grid)
                    if button.text == "Open...":
                        buttons[idx - 1].text = get_text(prompt_file(), True)

    draw(win=WIN, grid_=grid, buttons_=buttons)

pygame.quit()
exit(0)
