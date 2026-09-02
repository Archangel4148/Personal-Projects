

from collections.abc import Sequence

from simulation.actions import Action
from simulation.agent import Agent


class World:

    def __init__(self, agents: Sequence[Agent], name: str = "Unnamed World") -> None:
        self.agents = agents
        self.name = name

        self.time = 0

    def update_environment(self) -> None:
        pass

    def advance_time(self):
        self.time += 1

    def resolve_actions(self, actions: dict[Agent, Action]):
        pass
