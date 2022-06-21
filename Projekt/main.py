import tkinter
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
        for i in range(0, ROWS + 1, 7):
            pygame.draw.line(win, BLACK, (0, i * PIXEL_SIZE), (WIDTH, i * PIXEL_SIZE))

        for i in range(0, COLS + 1, 7):
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

button_y_1 = HEIGHT - TOOLBAR_HEIGHT / 2 - 85
button_y_2 = HEIGHT - TOOLBAR_HEIGHT / 2 - 25
button_y_3 = HEIGHT - TOOLBAR_HEIGHT / 2 + 50
buttons = [
    Button(10, button_y_1, 50, 50, BLACK),
    Button(70, button_y_1, 50, 50, WHITE),
    Button(130, button_y_1, 80, 50, WHITE, "Clear", BLACK),
    Button(340, button_y_1, 100, 50, WHITE, "Spot", BLUE),
    Button(10, button_y_2, 430, 50, WHITE),
    Button(10, button_y_3, 80, 50, WHITE, "Open...", BLACK),
    Button(340, button_y_3, 100, 50, WHITE, "Save", BLACK)
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
                        buttons[idx + 2].text = ''
                        drawing_color = BLACK
                        break
                    if button.text == "Spot":
                        buttons[idx + 1].text = get_text(grid)
                        break
                    if button.text == "Open...":
                        try:
                            buttons[idx - 1].text = get_text(prompt_file(), True)
                        except ValueError:
                            print('invalid file')
                        break
                    if button.text == "Save":
                        file = open('recognized.txt', 'w')
                        file.write(buttons[idx - 2].text)
                        file.close()
                    drawing_color = button.color

    draw(win=WIN, grid_=grid, buttons_=buttons)

pygame.quit()
exit(0)
