import pygame

from rendering import PointSequence, Point
from algorithm import get_processed_spells

pygame.init()

def build_rune(x, y, r, spell: dict) -> PointSequence:
    sequence = PointSequence.build_circle((x, y), r, 13)
    point_type = None
    conc, rit = spell["concentration"], spell["ritual"]
    if conc and rit:
        point_type = 2
    elif conc:
        point_type = 1
    elif rit:
        point_type = 0

    if point_type is not None:
        sequence.points.append(Point(x, y, special_point_type=point_type))
    return sequence

def main():
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True

    center = (screen.width // 2, screen.height // 2)

    # Render a spell!
    spell_name = "Alarm"

    spell = list(get_processed_spells().iterrows())[spell_idx][1].to_dict()
    print("Rendering spell:", spell["name"])
    sequence = build_rune(center[0], center[1], 200, spell)

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