from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from simulation.world import World

if TYPE_CHECKING:
    from simulation.world import World


class EndCondition(ABC):

    @abstractmethod
    def should_end(self, world: World) -> bool:
        ...


class TimeLimitCondition(EndCondition):

    def __init__(self, time_limit: int) -> None:
        self.time_limit = time_limit

    def should_end(self, world: World) -> bool:
        return world.time >= self.time_limit