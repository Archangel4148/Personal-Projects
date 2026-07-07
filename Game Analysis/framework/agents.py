import random
from abc import ABC, abstractmethod

from framework.base import GameState, Action


class Agent(ABC):
    """The standard representation of a game agent"""

    @abstractmethod
    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Choose an action for the given state"""
        pass


class RandomAgent(Agent):
    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Randomly choose a legal action"""
        return random.choice(legal_actions)
