from __future__ import annotations

from abc import ABC
import math
import random
from typing import TYPE_CHECKING
from collections.abc import Sequence

from simulation.actions import Action, MoveAction, NoOpAction
from simulation.entity import MovableEntity
from brains.senses.sense import Sense

if TYPE_CHECKING:
    from rendering.overlay import OverlayPrimitive
    from simulation.world import World
    from simulation.agent import Agent


class Brain(ABC):

    def __init__(self, senses: Sequence[Sense]) -> None:
        self.senses = senses
        self.knowledge = []

    def observe(self, world: World, agent: Agent) -> None:
        self.knowledge = []

        for sense in self.senses:
            self.knowledge.extend(
                sense.perceive(world, agent)
            )

    def choose_action(self, agent: Agent) -> Action:
        return NoOpAction()

    def overlays(self, agent: Agent) -> Sequence[OverlayPrimitive]:
        primitives: list[OverlayPrimitive] = []
        for sense in self.senses:
            primitives.extend(sense.overlay_primitives(agent))
        return primitives


def wander(agent: MovableEntity) -> MoveAction:
    """Take a max_speed step in a random direction"""
    angle = random.random() * 2 * math.pi
    return MoveAction(
        dx=agent.max_speed * math.cos(angle),
        dy=agent.max_speed * math.sin(angle),
    )


class RandomMovementBrain(Brain):

    def choose_action(self, agent: Agent):
        assert isinstance(agent, MovableEntity)
        return wander(agent)
