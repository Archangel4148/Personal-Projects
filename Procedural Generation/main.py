import random
from pathlib import Path
import pygame

from helpers import Cell, GridManager, Texture, WFCManager, load_tileset
from math_helpers import snap_to_factor

WINDOW_SIZE = (720, 720)
FPS = 40

# Tile information
ALLOWED_ROTATIONS = [0, 90, 180, 270]
TILESET_PATH = Path("tilesets/Circuit")
TILESET = load_tileset(TILESET_PATH, ALLOWED_ROTATIONS)

WAIT_FOR_KEY = False

def render_frame(screen: pygame.Surface, cells: list[Cell]):
    # Fill the background (wipes previous frame)
    screen.fill("black")

    # Render all cells
    for cell in cells:
        cell.draw(screen)

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    running = True

    w, h = WINDOW_SIZE

    # Snap dimension to a factor of the width (to avoid artifacts)
    square = 20
    dim=(square, square)
    snapped_dim = (
        snap_to_factor(w, dim[0]),
        snap_to_factor(h, dim[1])
    )
    print(f"Snapped dimension from {dim} to {snapped_dim}")
    grid = GridManager.build_grid(c1=(0, 0), c2=(w, h), dim=snapped_dim, tile_names=list(TILESET.keys()))
    solver = WFCManager(grid, TILESET)

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if WAIT_FOR_KEY:
                        # When space is pressed, step the solver once
                        solver.step()
                elif event.key == pygame.K_r:
                    solver.reset()
        if not WAIT_FOR_KEY:
            result = solver.step()
        render_frame(screen, grid.cells)

        # Update the display
        pygame.display.flip()

        # Limit FPS
        clock.tick(FPS)

if __name__ == "__main__":
    main()