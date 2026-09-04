from dataclasses import dataclass
from enum import StrEnum


class OverlayStyle(StrEnum):
    RING = "ring"
    FILL = "fill"
    FILL_AND_RING = "fill_and_ring"


@dataclass(frozen=True)
class Overlay:
    """How an overlay should look. Attach to a producer (or None to hide)."""
    color: tuple[int, int, int]
    style: OverlayStyle = OverlayStyle.RING
    fill_alpha: int = 40
    ring_width: int = 1


@dataclass(frozen=True)
class CircleOverlay:
    center: tuple[float, float]
    radius: float
    look: Overlay


@dataclass(frozen=True)
class PathOverlay:
    points: tuple[tuple[float, float], ...]
    color: tuple[int, int, int]
    width: int = 2


OverlayPrimitive = CircleOverlay | PathOverlay
