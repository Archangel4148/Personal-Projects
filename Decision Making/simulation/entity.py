from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NewType
import uuid

if TYPE_CHECKING:
    from simulation.world import World

EntityID = NewType("EntityID", uuid.UUID)

class Entity:
    """An object that exists in the world."""
    def __init__(self, name: str = "Unnamed Entity", position: tuple[float, float] = (0.0, 0.0)) -> None:
        self.name = name
        self.position = position

        # Give each Entity a unique ID
        self.id = EntityID(uuid.uuid4())

    @property
    def pos_x(self) -> float:
        return self.position[0]

    @property
    def pos_y(self) -> float:
        return self.position[1]


class UpdatableEntity(Entity, ABC):

    @abstractmethod
    def update(self, world: World) -> None:
        ...

class MovableEntity(Entity, ABC):
    @property
    @abstractmethod
    def max_speed(self) -> float:
        ...
