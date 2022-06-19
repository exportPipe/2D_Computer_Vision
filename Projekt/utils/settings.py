import pygame

pygame.init()
pygame.font.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FPS = 90

ROWS = COLS = 112
WIDTH, HEIGHT = ROWS * 7, COLS * 9

TOOLBAR_HEIGHT = HEIGHT - WIDTH

PIXEL_SIZE = WIDTH // COLS

BG_COLOR = WHITE

DRAW_GRID_LINES = True


def get_font(size):
    return pygame.font.SysFont("comicsans", size)
