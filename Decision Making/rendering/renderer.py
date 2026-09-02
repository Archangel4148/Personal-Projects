from abc import ABC, abstractmethod

from simulation.world import World


class Renderer(ABC):

    @abstractmethod
    def draw(self, world: World) -> None:
        ...

    def requests_stop(self) -> bool:
        """True if the renderer wants the simulation to end"""
        return False

    def keep_alive(self, world: World) -> None:
        """After the simulation ends, hold the final frame until dismissed"""
        return

class PrintRenderer(Renderer):

    def draw(self, world: World) -> None:
        print(f"Rendering {world.name}, a world at time {world.time}")
