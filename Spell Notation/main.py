import pygame

from rendering import PointSequence, Point, int_color
from algorithm import get_processed_spells, get_required_connection_indices

pygame.init()

def build_rune(x, y, r, spell: dict) -> PointSequence:
    """Build the rune sequence from a provided spell, center point, and radius"""
    num_points = 13
    sequence = PointSequence.build_circle((x, y), r, num_points)
    conc = spell.get("concentration", False)
    rit = spell.get("ritual", False)
    if conc and rit:
        point_type = 2
    elif conc:
        point_type = 1
    elif rit:
        point_type = 0
    else:
        point_type = None
    if point_type is not None:
        # Add the special center point
        sequence.points.append(Point(x, y, special_point_type=point_type))
    # Get required connections and add them
    connections = get_required_connection_indices(spell, num_points)
    for i, (k, conn_list) in enumerate(connections.items()):
        color = int_color(k)
        for connection in conn_list:
            sequence.connect_points(*connection, line_color=color)
    return sequence

def main():
    screen = pygame.display.set_mode((700, 700))
    clock = pygame.time.Clock()
    running = True

    center = (screen.width // 2, screen.height // 2)

    # Render a spell!
    spell_name = "Fireball"
    df = get_processed_spells()
    try:
        spell = df[df["name"] == spell_name].iloc[0].to_dict()
        print("Rendering spell:", spell["name"])
    except IndexError:
        print(f"Invalid spell name, '{spell_name}'")
        return
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

