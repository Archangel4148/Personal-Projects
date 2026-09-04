from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from collections.abc import Sequence

from tools.math_helpers import distance
from rendering.overlay import CircleOverlay

if TYPE_CHECKING:
    from rendering.overlay import Overlay, OverlayPrimitive
    from simulation.agent import Agent
    from simulation.world import World


class Sense(ABC):
    """A mechanism through which an agent can read information from the world"""

    def __init__(self, overlay: Overlay | None = None) -> None:
        self.overlay = overlay

    @abstractmethod
    def perceive(self, world: World, agent: Agent) -> Sequence[object]:
        """Return information currently available to the agent."""
        ...

    def overlay_primitives(self, agent: Agent) -> Sequence[OverlayPrimitive]:
        return ()

class EntitySense(Sense):
    """An sense that simply observes all entities within range (range=None for unlimited)"""
    def __init__(self, range: float | None = None, overlay: Overlay | None = None) -> None:
        super().__init__(overlay=overlay)
        self.range = range

    def perceive(self, world: World, agent: Agent):
        if self.range is None:
            return world.entities

        return [entity for entity in world.entities if distance(agent.position, entity.position) <= self.range]

    def overlay_primitives(self, agent: Agent) -> Sequence[OverlayPrimitive]:
        if self.overlay is None or self.range is None:
            return ()
        return (
            CircleOverlay(
                center=agent.position,
                radius=self.range,
                look=self.overlay,
            ),
        )
