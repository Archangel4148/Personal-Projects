

from simulation.agent import Agent


class World:

    def __init__(self, agents: list[Agent], name: str = "Unnamed World", start_time:int = 0) -> None:
        self.agents = agents
        self.name = name
        self.time = start_time

    def update(self) -> None:
        self.time += 1
