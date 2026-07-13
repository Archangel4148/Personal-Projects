import copy
from abc import ABC
from typing import Any

from framework.base import GameState, Action


class GameObserver(ABC):
    """Base class for observing game execution.

    Observers may collect statistics, log games, build datasets,
    visualize play, or terminate games early.
    """

    def on_game_start(self, initial_state: GameState) -> None:
        """Called once before the first action."""
        pass

    def before_action(
            self,
            state: GameState,
            current_player: int,
            legal_actions: list[Action]
    ) -> None:
        """Called immediately before an agent chooses an action."""
        pass

    def after_action(
            self,
            previous_state: GameState,
            action: Action,
            new_state: GameState
    ) -> None:
        """Called immediately after an action is applied."""
        pass

    def on_game_end(
            self,
            final_state: GameState,
            winners: list[int]
    ) -> None:
        """Called once when the game ends."""
        pass

    def get_results(self) -> dict[str, Any]:
        """Return any collected data."""
        return {}


class GameSummaryObserver(GameObserver):
    """Tracks standard game completion information."""

    def __init__(self):
        self.actions_taken = 0
        self.final_state = None
        self.winners = []

    def on_game_start(self, initial_state):
        self.actions_taken = 0
        self.final_state = None
        self.winners = []

    def after_action(self, previous_state, action, new_state):
        self.actions_taken += 1

    def on_game_end(self, final_state, winners):
        self.final_state = final_state
        self.winners = winners

    def get_results(self):
        return {
            "actions_taken": self.actions_taken,
            "final_state": self.final_state,
            "winner_indices": self.winners,
        }


class StateRecorder(GameObserver):
    """Stores every state visited during a game."""

    def __init__(self):
        self.states = []

    def on_game_start(self, initial_state):
        self.states.clear()
        self.states.append(copy.deepcopy(initial_state))

    def after_action(self, previous_state, action, new_state):
        self.states.append(copy.deepcopy(new_state))

    def get_results(self):
        return {
            "state_history": self.states
        }


class ActionRecorder(GameObserver):
    """Stores every action taken."""

    def __init__(self):
        self.actions = []
        self.current_player = None

    def on_game_start(self, initial_state: GameState) -> None:
        self.actions.clear()

    def before_action(self, state: GameState, current_player: int, legal_actions: list[Action]) -> None:
        self.current_player = current_player

    def after_action(self, previous_state, action, new_state):
        self.actions.append({"player": self.current_player, "action": action})

    def get_results(self):
        return {
            "action_history": self.actions
        }
