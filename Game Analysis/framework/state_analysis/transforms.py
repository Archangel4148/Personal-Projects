from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from framework.base import GameState, GameModule, GridGameModule


class StateTransform(ABC):
    @abstractmethod
    def transform(self, state: GameState, game: GameModule) -> GameState:
        ...


class IdentityTransform(StateTransform):
    def transform(self, state: GameState, game: GameModule) -> GameState:
        return state


class GridBoardTransform(StateTransform, ABC):

    @abstractmethod
    def transform_board(self, board: list[int], rows: int, cols: int) -> np.ndarray:
        ...

    def transform(self, state: GameState, game: GridGameModule) -> GameState:
        new_state = deepcopy(state)

        rows, cols = game.board_dimensions()

        new_state["board"] = self.transform_board(
            state["board"],
            rows,
            cols,
        )

        return GameState(new_state)


class FlipOverHorizontalAxis(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        board = np.asarray(board).reshape(rows, cols)
        return board[::-1, :].ravel()


class FlipOverVerticalAxis(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        board = np.asarray(board).reshape(rows, cols)
        return board[:, ::-1].ravel()


class Rotate90(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        if rows != cols:
            raise ValueError("Rotate90 requires square board")

        board = np.asarray(board).reshape(rows, cols)
        return np.rot90(board, k=-1).ravel()  # clockwise


class Rotate180(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        board = np.asarray(board).reshape(rows, cols)
        return np.rot90(board, k=2).ravel()
        # or simply: return board[::-1].ravel()


class Rotate270(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        if rows != cols:
            raise ValueError("Rotate270 requires square board")

        board = np.asarray(board).reshape(rows, cols)
        return np.rot90(board, k=1).ravel()  # counter-clockwise


class Transpose(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        if rows != cols:
            raise ValueError("Transpose requires square board")

        board = np.asarray(board).reshape(rows, cols)
        return board.T.ravel()


class FlipOverAntiDiagonal(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        if rows != cols:
            raise ValueError("Anti-diagonal flip requires square board")

        board = np.asarray(board).reshape(rows, cols)
        return np.fliplr(np.flipud(board)).T.ravel()


class SwapPlayersTransform(StateTransform):
    def transform(self, state, game):
        new_state = deepcopy(state)

        new_state["board"] = [
            -x if x != 0 else 0
            for x in state["board"]
        ]

        new_state["turn_flag"] = 1 - state["turn_flag"]

        return new_state

class PermuteGroupsTransform(StateTransform):
    def transform(self, state, game):
        new_state = deepcopy(state)

        new_state["groups"] = tuple(sorted(list(new_state["groups"])))

        return new_state