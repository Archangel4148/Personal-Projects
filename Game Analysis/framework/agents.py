import random
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from framework.base import GameState, Action
from framework.state_analysis.reduction import StateEquivalence, SymmetryEquivalence
if TYPE_CHECKING:
    from framework.state_analysis.solver import StateSolver


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


class SolvedGraphAgent(Agent):
    def __init__(self, solver: "StateSolver", equivalence: StateEquivalence) -> None:
        self.solver = solver
        self.equivalence = equivalence
        self.game = solver.game
        
        # Pre-cache the edge lookup for fast action retrieval during live games
        self.edge_lookup = {}
        for edge in self.solver.graph.edges:
            self.edge_lookup[(edge.source, edge.target)] = edge

    def choose_action(self, state: Any, legal_actions: list[Any]) -> Any:
        """Plays perfectly by looking up the optimal path in the solved graph."""
        canonical_state = self.equivalence.canonical(state, self.game)
        canonical_key = self.game.state_key(canonical_state)
        
        # Look up the pre-calculated optimal next state
        next_state_key = self.solver.optimal_moves.get(canonical_key)
        
        # If the state exists in the database, find the winning action
        if next_state_key is not None:
            for action in legal_actions:
                next_live_state = self.game.apply_action(state, action) 
                
                # Get the canonical key of where this live action lands
                next_live_canonical = self.equivalence.canonical(next_live_state, self.game)
                next_live_key = self.game.state_key(next_live_canonical)
                
                # If this action leads to our optimal canonical state, take it!
                if next_live_key == next_state_key:
                    return action
        

        # Fallback: play randomly
        return random.choice(legal_actions)