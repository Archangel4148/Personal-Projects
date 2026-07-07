import numpy as np

from framework.base import GameModule, GameState, Action
from framework.components import GridBoardComponent


class TicTacToeModule(GameModule):
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


if __name__ == '__main__':
    # Build and initialize a game module
    module = TicTacToeModule()
    state = module.setup_initial_state()

    # Show the available actions, make some moves, and show the reduced list of actions
    print(module.get_legal_actions(state))
    state = module.apply_action(state, Action(type="PLACE_MARKER", payload={"idx": 3}))
    state = module.apply_action(state, Action(type="PLACE_MARKER", payload={"idx": 1}))
    state = module.apply_action(state, Action(type="PLACE_MARKER", payload={"idx": 4}))
    state = module.apply_action(state, Action(type="PLACE_MARKER", payload={"idx": 2}))
    state = module.apply_action(state, Action(type="PLACE_MARKER", payload={"idx": 5}))
    print(module.get_legal_actions(state))

    # Check if the game has ended
    print("Game Over? :", module.is_game_over(state))
    print("Vectorized State :", module.vectorize_state(state))

    # Show the final board
    print("\nBoard State:")
    GridBoardComponent.print_board(state["board"], width=3)
