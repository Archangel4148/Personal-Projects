from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from simulation.actions import Action

if TYPE_CHECKING:
    from simulation.world import World


class Agent(ABC):

    def __init__(self, name="Unnamed Agent") -> None:
        self.name = name

    @abstractmethod
    def observe(self, world: "World") -> None:
        """Observe the world, and update memory/state (read-only, this should not touch the world)"""
        ...

    @abstractmethod
    def choose_action(self) -> Action:
        """Based on the current state/observations, choose an action to take"""
        ...


class LazyAgent(Agent):
    """A boring agent that does nothing"""

    def observe(self, world: World) -> None:
        """"""
        pass

    def choose_action(self) -> Action:
        """Just takes a blank no-op action"""
        return Action()