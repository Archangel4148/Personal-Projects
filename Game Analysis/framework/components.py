from abc import ABC
from random import shuffle

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


class CardStackComponent(GameComponent):
    """Handles low-level data structure management for decks, hands, and discard piles"""

    @staticmethod
    def create_deck(size: int=52):
        """Build a flat numpy array representing a deck of the given size"""
        return np.arange(size, dtype=np.int8)

    @staticmethod
    def shuffle(deck: np.ndarray) -> np.ndarray:
        """Shuffle the provided deck"""
        shuffle(deck)
        return deck

    @staticmethod
    def transfer_card(from_stack: list, to_stack: list, index: int = 0):
        """Transfer card {index} from one stack to another"""
        if from_stack:
            card = from_stack.pop(index)
            to_stack.append(card)
