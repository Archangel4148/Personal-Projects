from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING

from brains.brain import Brain, RandomMovementBrain
from rendering.appearance import Appearance
from simulation.actions import Action
from simulation.entity import Entity, MovableEntity

if TYPE_CHECKING:
    from simulation.world import World

class Agent(Entity):

    def __init__(self, brain: Brain, name: str = "Unnamed Agent", position: tuple[float, float] = (0.0, 0.0), appearance: Appearance | None = None) -> None:
        super().__init__(name=name, position=position, appearance=appearance)

        self.brain = brain

    def observe(self, world: World) -> None:
        """Observe the world, and update memory/state (read-only, this should not touch the world)"""
        self.brain.observe(world, self)

    def choose_action(self) -> Action:
        """Based on the current state/observations, choose an action to take"""
        return self.brain.choose_action(self)


class RandomMoveAgent(Agent, MovableEntity):
    """An agent that takes a MoveAction in a random direction each tick"""

    def __init__(self, step_distance: float, name: str = "Unnamed Random Move Agent", position: tuple[float, float] = (0.0, 0.0), appearance: Appearance | None = None) -> None:
        super().__init__(
            brain=RandomMovementBrain(senses=[]), 
            name=name, 
            position=position,
            appearance=appearance
        )

        self._step = step_distance

    @property
    def max_speed(self) -> float:
        return self._step
