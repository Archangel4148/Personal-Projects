import pygame

from rendering import PointSequence

pygame.init()

def main():
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True

    center = (screen.width // 2, screen.height // 2)

    # Create a circular sequence of points about the center of the screen
    sequence = PointSequence.build_circle(center, 200, 16)

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Background/clear screen
        screen.fill("white")

        # Render the points sequence
        sequence.draw(screen)

        pygame.display.flip()
        clock.tick(60)  # FPS limit

if __name__ == "__main__":
    main()