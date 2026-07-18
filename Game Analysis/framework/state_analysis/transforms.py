from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

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
    def transform_board(self, board: list[int], rows: int, cols: int) -> list[int]:
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
        result = []

        for r in reversed(range(rows)):
            start = r * cols
            result.extend(board[start:start + cols])

        return result


class FlipOverVerticalAxis(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        result = []

        for r in range(rows):
            row = board[r * cols:(r + 1) * cols]
            result.extend(reversed(row))

        return result


class Rotate90(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        if rows != cols:
            raise ValueError("Rotate90 requires square board")

        result = []

        for c in range(cols):
            for r in reversed(range(rows)):
                result.append(board[r * cols + c])

        return result


class Rotate180(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        return list(reversed(board))


class Rotate270(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        if rows != cols:
            raise ValueError("Rotate270 requires square board")

        result = []

        for c in reversed(range(cols)):
            for r in range(rows):
                result.append(board[r * cols + c])

        return result


class Transpose(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        if rows != cols:
            raise ValueError("Transpose requires square board")

        result = []

        for c in range(cols):
            for r in range(rows):
                result.append(board[r * cols + c])

        return result


class FlipOverAntiDiagonal(GridBoardTransform):
    def transform_board(self, board, rows, cols):
        if rows != cols:
            raise ValueError("Anti-diagonal flip requires square board")

        result = []

        for r in range(rows):
            for c in range(cols):
                result.append(
                    board[(cols - 1 - c) * cols + (rows - 1 - r)]
                )

        return result


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