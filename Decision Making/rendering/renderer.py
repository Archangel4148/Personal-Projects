

from abc import ABC, abstractmethod

from simulation.world import World


class Renderer(ABC):

    @abstractmethod
    def draw(self, world: World) -> None:
        ...


class PrintRenderer(Renderer):

    def draw(self, world: World) -> None:
        print(f"Rendering {world.name}, a world at time {world.time}")