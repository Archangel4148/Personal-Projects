from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.world import World


class Agent:

    def __init__(self, name="Unnamed Agent") -> None:
        self.name = name

    def update(self, world: "World"):
        pass