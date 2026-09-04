from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from collections.abc import Sequence

from tools.math_helpers import distance

if TYPE_CHECKING:
    from simulation.agent import Agent
    from simulation.world import World


class Sense(ABC):
    """A mechanism through which an agent can read information from the world"""

    @abstractmethod
    def perceive(self, world: World, agent: Agent) -> Sequence[object]:
        """Return information currently available to the agent."""
        ...

class EntitySense(Sense):
    """An sense that simply observes all entities within range (range=None for unlimited)"""
    def __init__(self, range: float | None = None) -> None:
        super().__init__()
        self.range = range

    def perceive(self, world: World, agent: Agent):
        if self.range is None:
            return world.entities

        return [entity for entity in world.entities if distance(agent.position, entity.position) <= self.range]
