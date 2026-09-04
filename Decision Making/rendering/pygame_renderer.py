import pygame

from rendering.appearance import EntityShape
from rendering.renderer import Renderer
from simulation.world import World


class PygameRenderer(Renderer):
    
    def __init__(self, window_size: tuple[int, int], title="Simulation", bg_color = (0, 0, 0)) -> None:
        self.width, self.height = window_size
        self.bg_color = bg_color

        self._closed = False

        # Set up PyGame window
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)

    def draw(self, world: World) -> None:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True
                return

        # Draw the current state (World + Entities)
        self.screen.fill(self.bg_color)

        # TODO: Draw world border

        # Draw entities
        # TODO: Have custom "DrawMethod" information for each entity
        for entity in world.entities:
            x, y = entity.position
            look = entity.appearance

            if look.shape == EntityShape.CIRCLE:
                pygame.draw.circle(
                    self.screen,
                    look.color,
                    (x, y),
                    look.size,
                )

        # Show the completed frame
        pygame.display.flip()

    def requests_stop(self) -> bool:
        return self._closed
    
    def keep_alive(self, world: World) -> None:
        if self._closed:
            pygame.quit()
            return

        clock = pygame.time.Clock()
        while not self._closed:
            self.draw(world)
            clock.tick(30)  # Limit to 30FPS, since nothing is happening
        pygame.quit()
