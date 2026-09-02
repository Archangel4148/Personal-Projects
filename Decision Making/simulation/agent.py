from __future__ import annotations
from abc import ABC, abstractmethod
import math
import random
from typing import TYPE_CHECKING

from simulation.actions import Action, MoveAction, NoOpAction
from simulation.combat import Attack, AttackEntityAction, CombatEntity, DamageableEntity
from simulation.entity import Entity, EntityID, MovableEntity
from tools.math import distance, step_towards

if TYPE_CHECKING:
    from simulation.world import World


class Agent(Entity, ABC):

    def __init__(self, name: str = "Unnamed Agent", position: tuple[float, float] = (0.0, 0.0)) -> None:
        super().__init__(name, position)

    @abstractmethod
    def observe(self, world: World) -> None:
        """Observe the world, and update memory/state (read-only, this should not touch the world)"""
        # TODO: Eventually, this should evolve into the agent having senses that each can observe the world, and the agent relies on those, not the world itself
        ...

    @abstractmethod
    def choose_action(self) -> Action:
        """Based on the current state/observations, choose an action to take"""
        ...


class LazyAgent(Agent):
    """A boring agent that does nothing"""

    def observe(self, world: World) -> None:
        pass

    def choose_action(self) -> Action:
        """Just takes a blank no-op action"""
        return NoOpAction()


class RandomMoveAgent(Agent, MovableEntity):
    """An agent that takes a MoveAction in a random direction each tick"""

    def __init__(self, step_distance: float, name: str = "Unnamed Random Move Agent", position: tuple[float, float] = (0, 0)) -> None:
        super().__init__(name, position)

        self._step = step_distance

    @property
    def max_speed(self) -> float:
        return self._step

    def observe(self, world: World) -> None:
        pass

    def choose_action(self) -> Action:
        # Choose a random direction to move in
        angle = random.random() * 2 * math.pi
        return MoveAction(dx=self._step * math.cos(angle), dy=self._step * math.sin(angle))
