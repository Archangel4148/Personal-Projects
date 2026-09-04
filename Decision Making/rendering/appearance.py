
from dataclasses import dataclass
from enum import StrEnum

class EntityShape(StrEnum):
    CIRCLE = "circle"

@dataclass(frozen=True)
class Appearance:
    color: tuple[int, int, int]
    size: float = 5.0
    shape: EntityShape = EntityShape.CIRCLE
