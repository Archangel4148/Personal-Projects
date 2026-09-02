from dataclasses import dataclass, field
from collections.abc import Sequence

from simulation.end_conditions import EndCondition
from simulation.entity import Entity
from simulation.world import World

@dataclass
class Scenario:
    name: str
    bounds: tuple[int, int]
    tps: float = 20
    end_conditions: Sequence[EndCondition] = field(default_factory=list)

    def build_entities(self) -> Sequence[Entity]:
        raise NotImplementedError

    def build_world(self) -> World:
        return World(
            entities=self.build_entities(),
            name=self.name,
            bounds=self.bounds,
        )
