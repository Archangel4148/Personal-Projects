from abc import ABC

import numpy as np


class GameComponent(ABC):
    """Base class for atomic game components representing unique mechanics and specifics
    for implementing different GameModules"""
    pass


class GridBoardComponent(GameComponent):
    """Handles low-level grid indexing and line checks"""

    @staticmethod
    def create_board(width: int, height: int, empty_val: int) -> np.ndarray:
        """Returns a 1D {width}x{height} numpy array of {empty_val}"""
        return np.full((width * height), empty_val)

    @staticmethod
    def check_lines(board: np.ndarray, lines: list[list[int]], empty_val: int) -> int:
        """Checks each of the provided lines. If all grid values along a line match
        (excluding empty value), returns the grid value"""
        for line in lines:
            first_val = board[line[0]]
            if first_val != empty_val and np.all(board[line] == first_val):
                return first_val
        return empty_val

    @staticmethod
    def iter_rows(board: np.ndarray, width: int) -> list[np.ndarray]:
        reshaped = board.reshape(-1, width)
        return [row for row in reshaped]

    @staticmethod
    def iter_columns(board: np.ndarray, width: int) -> list[np.ndarray]:
        reshaped = board.reshape(-1, width)
        return [col for col in reshaped.T]

    @classmethod
    def print_board(cls, board: np.ndarray, width: int, display_map: dict[int, str] | None = None) -> None:
        """Prints each row of the provided board"""
        print_map = (lambda val: display_map.get(val, str(val))) if display_map else str
        for row in cls.iter_rows(board, width):
            print(" ".join(map(print_map, row)))