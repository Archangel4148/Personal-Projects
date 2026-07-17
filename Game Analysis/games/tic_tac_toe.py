from typing import Hashable

import numpy as np

from framework.agents import RandomAgent
from framework.base import GameState, Action, GridGameModule
from framework.components import GridBoardComponent
from framework.runner import GameRunner


class TicTacToeModule(GridGameModule):
    WIN_COMBINATIONS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontal rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Vertical columns
        [0, 4, 8], [2, 4, 6]  # Diagonals
    ]

    def setup_initial_state(self, config: dict | None = None) -> GameState:
        """Set up the empty board, and set it to turn 1 (player 1)"""
        return GameState({
            "board": GridBoardComponent.create_board(3, 3, 0),
            "turn_flag": 1,
        })

    def get_current_player_idx(self, state: GameState) -> int:
        return 0 if state["turn_flag"] == 1 else 1

    def get_legal_actions(self, state: GameState) -> list[Action]:
        """Return placement actions for each remaining empty square"""
        actions = []
        for idx, val in enumerate(state["board"]):
            if val == 0:
                actions.append(Action(type="PLACE_MARKER", payload={"idx": idx}))
        return actions

    def apply_action(self, state: GameState, action: Action) -> GameState:
        new_state = GameState(state.copy())
        new_state["board"] = state["board"].copy()
        # Handle placement
        if action.type == "PLACE_MARKER":
            # Depending on turn, place a -1 or a 1
            new_state["board"][action.payload["idx"]] = state["turn_flag"]

        # Flip the turn
        new_state["turn_flag"] *= -1
        return new_state

    def is_game_over(self, state: GameState) -> tuple[bool, list[int]]:
        winner = GridBoardComponent.check_lines(state["board"], self.WIN_COMBINATIONS, 0)
        # There is a winner
        if winner != 0:
            winner_idx = 0 if winner == 1 else 1
            return True, [winner_idx]
        # Tie game (no available spaces)
        if 0 not in state["board"]:
            return True, []
        # No winner, continue play
        return False, []

    def vectorize_state(self, state: GameState) -> np.ndarray:
        # Dump the turn and board into a 10x1 vector (first value is the turn)
        return np.insert(state["board"], 0, state["turn_flag"])

    def state_key(self, state: GameState) -> Hashable:
        return tuple([tuple(list(map(int, state["board"]))), state["turn_flag"]])

    def board_dimensions(self) -> tuple[int, int]:
        """rows, columns"""
        return 3, 3

    def board_display_map(self) -> dict[int, str]:
        return {-1: "X", 1: "O", 0: "-"}


if __name__ == '__main__':
    # Build and initialize a game module
    module = TicTacToeModule()
    players = [RandomAgent(), RandomAgent()]

    runner = GameRunner(module, players)

    results = runner.run_game()
    print("Results :", results)

    # Show the final board
    print("\nBoard State:")
    display = module.render_state(results["final_state"])
    print(display)
