import pygame

from rendering.appearance import EntityShape
from rendering.overlay import CircleOverlay, OverlayPrimitive, OverlayStyle, PathOverlay
from rendering.renderer import Renderer
from simulation.world import World


class PygameRenderer(Renderer):
    
    def __init__(
        self,
        window_size: tuple[int, int],
        title="Simulation",
        bg_color=(0, 0, 0),
        show_overlays: bool = True,
    ) -> None:
        self.width, self.height = window_size
        self.bg_color = bg_color
        self.show_overlays = show_overlays

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

        self.screen.fill(self.bg_color)

        # TODO: Draw world border

        if self.show_overlays:
            self._draw_overlays(world)
        self._draw_entities(world)

        pygame.display.flip()

    def _draw_overlays(self, world: World) -> None:
        for entity in world.entities:
            for primitive in entity.overlays():
                self._draw_overlay_primitive(primitive)

    def _draw_overlay_primitive(self, primitive: OverlayPrimitive) -> None:
        if isinstance(primitive, CircleOverlay):
            self._draw_circle_overlay(primitive)
        elif isinstance(primitive, PathOverlay):
            self._draw_path_overlay(primitive)

    def _draw_circle_overlay(self, primitive: CircleOverlay) -> None:
        look = primitive.look
        if look.style in (OverlayStyle.FILL, OverlayStyle.FILL_AND_RING):
            self._draw_filled_circle(look.color, primitive.center, primitive.radius, look.fill_alpha)
        if look.style in (OverlayStyle.RING, OverlayStyle.FILL_AND_RING):
            pygame.draw.circle(
                self.screen,
                look.color,
                primitive.center,
                primitive.radius,
                width=look.ring_width,
            )

    def _draw_path_overlay(self, primitive: PathOverlay) -> None:
        if len(primitive.points) < 2:
            return
        pygame.draw.lines(
            self.screen,
            primitive.color,
            False,
            primitive.points,
            primitive.width,
        )

    def _draw_filled_circle(
        self,
        color: tuple[int, int, int],
        center: tuple[float, float],
        radius: float,
        alpha: int,
    ) -> None:
        r = round(radius)
        if r <= 0:
            return
        surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, max(0, min(alpha, 255))), (r, r), r)
        x, y = center
        self.screen.blit(surf, (x - r, y - r))

    def _draw_entities(self, world: World) -> None:
        for entity in world.entities:
            look = entity.appearance
            if look.shape == EntityShape.CIRCLE:
                pygame.draw.circle(
                    self.screen,
                    look.color,
                    entity.position,
                    look.size,
                )

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
