from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Hashable, NewType

import numpy as np

GameState = NewType("GameState", dict[str, Any])


@dataclass(kw_only=True, frozen=True)
class Action:
    """A single action to be applied to a game state"""
    type: str
    payload: Any = None


class GameModule(ABC):
    """The standard representation of a game"""

    @abstractmethod
    def setup_initial_state(self, config: dict) -> GameState:
        """Initialize the module to the starting state"""
        pass

    @abstractmethod
    def get_current_player_idx(self, state: GameState) -> int:
        """Return the integer index of the player required to choose the next action"""
        pass

    @abstractmethod
    def get_legal_actions(self, state: GameState) -> list[Action]:
        """Get the legal actions for the given state"""
        pass

    @abstractmethod
    def apply_action(self, state: GameState, action: Action) -> GameState:
        """Apply the given action to the given state"""
        pass

    @abstractmethod
    def is_game_over(self, state: GameState) -> tuple[bool, list[int]]:
        """Determine if the game is over, and return a tuple of winning player indices (if any)"""
        pass

    @abstractmethod
    def vectorize_state(self, state: GameState) -> np.ndarray:
        """Dump the state into a numerical vector"""
        pass

    @abstractmethod
    def hash_state(self, state: GameState) -> Hashable:
        pass
