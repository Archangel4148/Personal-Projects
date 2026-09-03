
from abc import ABC
import math
import random
from typing import TYPE_CHECKING, Generic, TypeVar
from collections.abc import Sequence

from simulation.actions import Action, MoveAction, NoOpAction

if TYPE_CHECKING:
    from brains.senses.sense import Sense
    from simulation.agent import RandomMoveAgent
    from simulation.agent import Agent
    from simulation.world import World

# Define a template type for agents
AgentT = TypeVar("AgentT", bound="Agent")

class Brain(ABC, Generic[AgentT]):

    def __init__(self, senses: Sequence[Sense]) -> None:
        self.senses = senses
        self.knowledge = []

    def observe(self, world: World, agent: AgentT) -> None:
        self.knowledge = []

        for sense in self.senses:
            self.knowledge.extend(
                sense.perceive(world, agent)
            )

    def choose_action(self, agent: AgentT) -> Action:
        return NoOpAction()


class RandomMovementBrain(Brain[RandomMoveAgent]):

    def choose_action(self, agent):
        angle = random.random() * 2 * math.pi

        return MoveAction(
            dx=agent.max_speed * math.cos(angle),
            dy=agent.max_speed * math.sin(angle),
        )
