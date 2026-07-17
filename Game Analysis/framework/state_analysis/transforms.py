from __future__ import annotations

from abc import ABC, abstractmethod

from framework.base import GameState, GameModule


class StateTransform(ABC):
    @abstractmethod
    def transform(self, state: GameState, game: GameModule) -> GameState:
        ...

class IdentityTransform(StateTransform):
    def transform(self, state: GameState, game: GameModule) -> GameState:
        return state
