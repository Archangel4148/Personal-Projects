from abc import ABC


class Action(ABC):
    ...

class NoOpAction(Action):
    pass

class MoveAction(Action):
    def __init__(self, dx: float, dy: float) -> None:
        self.dx = dx
        self.dy = dy
