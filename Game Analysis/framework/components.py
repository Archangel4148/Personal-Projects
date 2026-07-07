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
    def print_board(board: np.ndarray, width: int) -> None:
        """Prints each row of the provided board"""
        for row in range(0, board.size, width):
            print(" ".join(map(str, board[row: row + width])))