
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from collections.abc import Sequence

if TYPE_CHECKING:
    from simulation.agent import Agent
    from simulation.world import World


class Sense(ABC):
    """A mechanism through which an agent can read information from the world"""

    @abstractmethod
    def perceive(self, world: World, agent: Agent) -> Sequence[object]:
        """Return information currently available to the agent."""
        ...
