import pygame

from rendering import PointSequence, Point
from algorithm import get_processed_spells

pygame.init()

def build_rune(x, y, r, spell: dict) -> PointSequence:
    sequence = PointSequence.build_circle((x, y), r, 13)
    if spell["concentration"]:
        sequence.points.append(Point(x, y, special_point_type=0))
    elif spell["ritual"]:
        sequence.points.append(Point(x, y, special_point_type=1))
    return sequence

def main():
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True

    center = (screen.width // 2, screen.height // 2)

    # Create a circular sequence of points about the center of the screen
    sequence = PointSequence.build_circle(center, 200, 13)

    print(list(get_processed_spells().iterrows())[0].to_dict())

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

# AOE, Damage Type, and Duration can be zero, everything else is not

# Dot in the center for concentration, circled for ritual