import pygame
import sys
import numpy as np

WINDOW_SIZE = 700
GRID_SIZE = 28
CELL_SIZE = WINDOW_SIZE // GRID_SIZE

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_COLOR = (200, 200, 200)

grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

created_img_num = 0

def draw_grid(screen):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pygame.draw.rect(screen, GRID_COLOR, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
            color = WHITE if grid[row][col] == 0 else BLACK
            pygame.draw.rect(screen, color, (col * CELL_SIZE + 1, row * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2))

def print_grid():
    for row in grid:
        print(" ".join(map(str, row)))

def main():

    global created_img_num

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("MNIST")

    clock = pygame.time.Clock()

    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    col = min(max(0, mouse_x // CELL_SIZE), GRID_SIZE - 1)
                    row = min(max(0, mouse_y // CELL_SIZE), GRID_SIZE - 1)
                    grid[row][col] = 1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    for row in range(GRID_SIZE):
                        for col in range(GRID_SIZE):
                            grid[row][col] = 0
                elif event.key == pygame.K_p:
                    print("Prediction:", nn.predict(np.array(grid).flatten()))

                elif event.key == pygame.K_s:
                    pygame.image.save(screen, f"img/img{created_img_num}.png")
                    print("img created.")
                    created_img_num = created_img_num + 1

        screen.fill(WHITE)
        draw_grid(screen)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()