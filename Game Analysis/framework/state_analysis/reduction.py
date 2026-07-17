from __future__ import annotations

from abc import ABC, abstractmethod

from framework.base import GameState, GameModule
from framework.state_analysis.transforms import StateTransform, IdentityTransform


class StateEquivalence(ABC):

    @abstractmethod
    def canonical(self, state: GameState, game: GameModule) -> GameState:
        """Return the canonical state of the provided game state"""
        ...


class SymmetryEquivalence(StateEquivalence):
    def __init__(self, *transforms: StateTransform):
        self.transforms = (
            IdentityTransform(),
            *transforms,
        )

    def canonical(
            self,
            state: GameState,
            game: GameModule,
    ) -> GameState:
        variants = [
            transform.transform(state, game)
            for transform in self.transforms
        ]
        # Note: It doesn't matter which state variant is picked, as long as it's consistent
        return min(variants, key=game.state_key)
