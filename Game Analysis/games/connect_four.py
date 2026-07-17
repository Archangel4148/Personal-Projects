from typing import Hashable

import numpy as np

from framework.agents import RandomAgent
from framework.base import GameModule, GameState, Action, GridGameModule
from framework.components import GridBoardComponent
from framework.runner import GameRunner


def generate_connect_four_lines(width: int = 7, height: int = 6) -> list[list[int]]:
    lines = []
    # For each cell, append lines in every direction (if they fit on the board)
    for r in range(height):
        for c in range(width):
            # Horizontal Right
            if c + 3 < width:
                lines.append([
                    r * width + c,
                    r * width + (c + 1),
                    r * width + (c + 2),
                    r * width + (c + 3)
                ])

            # Vertical Up
            if r + 3 < height:
                lines.append([
                    r * width + c,
                    (r + 1) * width + c,
                    (r + 2) * width + c,
                    (r + 3) * width + c
                ])

            # Diagonal Up-Right
            if c + 3 < width and r + 3 < height:
                lines.append([
                    r * width + c,
                    (r + 1) * width + (c + 1),
                    (r + 2) * width + (c + 2),
                    (r + 3) * width + (c + 3)
                ])

            # Diagonal Up-Left
            if c - 3 >= 0 and r + 3 < height:
                lines.append([
                    r * width + c,
                    (r + 1) * width + (c - 1),
                    (r + 2) * width + (c - 2),
                    (r + 3) * width + (c - 3)
                ])

    return lines


class ConnectFourModule(GridGameModule):
    def __init__(self, board_width: int = 7, board_height: int = 6):
        self.board_width = board_width
        self.board_height = board_height

        # Calculate combinations once on creation
        self.win_combinations = generate_connect_four_lines(self.board_width, self.board_height)

    def setup_initial_state(self, config: dict | None = None) -> GameState:
        """Set up the empty board, and set it to turn 1 (player 1)"""
        return GameState({
            "board": GridBoardComponent.create_board(self.board_width, self.board_height, 0),
            "turn_flag": 1,
        })

    def get_current_player_idx(self, state: GameState) -> int:
        return 0 if state["turn_flag"] == 1 else 1

    def get_legal_actions(self, state: GameState) -> list[Action]:
        """Return placement actions for each non-full column"""
        actions = []
        for i, col in enumerate(GridBoardComponent.iter_columns(state["board"], self.board_width)):
            # If there is an available space, it's a valid column
            if 0 in col:
                actions.append(Action(type="PLACE_MARKER", payload={"col": i}))
        return actions

    def apply_action(self, state: GameState, action: Action) -> GameState:
        new_state = GameState(state.copy())
        new_state["board"] = state["board"].copy()
        board = new_state["board"]
        # Handle placement
        if action.type == "PLACE_MARKER":
            target_col = GridBoardComponent.iter_columns(board, self.board_width)[action.payload["col"]]
            target_idx = np.where(target_col == 0)[0][-1] * self.board_width + action.payload["col"]
            # Depending on turn, place a -1 or a 1
            board[target_idx] = state["turn_flag"]

        # Flip the turn
        new_state["turn_flag"] *= -1
        return new_state

    def is_game_over(self, state: GameState) -> tuple[bool, list[int]]:
        winner = GridBoardComponent.check_lines(state["board"], self.win_combinations, 0)
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
        return 6, 7


if __name__ == '__main__':
    # Set up a game
    width, height = 7, 6
    module = ConnectFourModule(width, height)
    players = [RandomAgent(), RandomAgent()]

    runner = GameRunner(module, players)

    # Run the game
    for i in range(3):
        print(f"\n\n======GAME {i+1}======")
        results = runner.run_game(config={"board_width": width, "board_height": height})

        # Display the results
        print("Results:", results)
        print("\nFinal Board:")
        GridBoardComponent.print_board(results["final_state"]["board"], width, display_map={1: "X", -1: "O", 0: "-"})
