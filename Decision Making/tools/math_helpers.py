
import math


def distance(pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
    return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

def clamp_magnitude(dx: float, dy: float, max_len: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length <= max_len or length == 0:
        return dx, dy
    scale = max_len / length
    return dx * scale, dy * scale

def step_towards(from_pos: tuple[float, float], to_pos: tuple[float, float], max_step: float) -> tuple[float, float]:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return clamp_magnitude(dx, dy, max_step)