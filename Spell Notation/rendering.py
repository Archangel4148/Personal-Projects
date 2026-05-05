
from dataclasses import dataclass
import numpy as np

import pygame


@dataclass
class Point:
    x: float
    y: float
    point_index: int | None = None
    color: str = "black"
    radius: int = 10
    special_point_type: int | None = None

    @property
    def is_start_point(self) -> bool:
        return self.point_index == 0
    
    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y
    
    def draw(self, surface: pygame.Surface):
        # Draw the point as a circle
        pygame.draw.circle(surface, self.color, self.position, self.radius)

        # Draw the "special point"
        if self.special_point_type is not None:
            if self.special_point_type in (0, 2):
                # Type 0: Hollow circle
                bg_color = surface.get_at((0, 0))
                pygame.draw.circle(surface, bg_color, self.position, self.radius * 0.75)

            if self.special_point_type in (1, 2):
                # Type 1: Small dot
                pygame.draw.circle(surface, self.color, self.position, self.radius / 2.5)

        # Draw the start point as a hollow circle
        if self.is_start_point:
            bg_color = surface.get_at((0, 0))
            pygame.draw.circle(surface, bg_color, self.position, self.radius * 0.75)

@dataclass
class Connection:
    point_1: Point
    point_2: Point
    line_thickness: int = 4
    color: str = "black"

    def draw(self, surface: pygame.Surface):
        pygame.draw.line(surface, self.color, self.point_1.position, self.point_2.position, self.line_thickness)

class PointSequence:
    def __init__(self, points: list[Point]):
        self.points: list[Point] = points
        self.connections: list[Connection] = []

    def get_point(self, index: int) -> Point | None:
        try:
            return next(point for point in self.points if point.point_index == index)
        except StopIteration:
            return None

    def draw(self, surface: pygame.Surface):
        """Render all points and connections to the provided surface"""
        for connection in self.connections:
            connection.draw(surface)
        for point in self.points:
            point.draw(surface)

    @classmethod
    def build_circle(cls, center: tuple[float, float], radius: float, num_points: int):
        """Create a circle of num_points points around the provided center with the provided radius"""
        cx, cy = center
        # Get evenly spaced angles (starting from the top)
        angles = np.linspace(-(np.pi / 2), (2*np.pi) - (np.pi / 2), num_points + 1)
        # Convert angles and radius to cartesian coordinates
        positions = [((radius * np.cos(angle)) + cx, (radius * np.sin(angle)) + cy) for angle in angles[:-1]]
        # Create points, and assign indices in order
        points = [Point(*pos, i) for i, pos in enumerate(positions)]
        return cls(points)
    
    def connect_points(self, idx_1: int, idx_2: int, line_width=4, line_color="black"):
        """Create a connection between two points with the provided indices"""
        p1 = self.get_point(idx_1)
        if p1 is None:
            raise ValueError(f"Point with id {idx_1} does not exist.")
        p2 = self.get_point(idx_2)
        if p2 is None:
            raise ValueError(f"Point with id {idx_2} does not exist.")
        self.connections.append(Connection(p1, p2, line_thickness=line_width, color=line_color))

if __name__ == "__main__":
    sequence = PointSequence.build_circle((100, 100), 100, 5)