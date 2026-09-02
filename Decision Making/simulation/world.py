

from collections.abc import Sequence

from simulation.actions import Action, MoveAction
from simulation.agent import Agent
from simulation.entity import Entity, UpdatableEntity


class World:

    def __init__(self, entities: Sequence[Entity], name: str = "Unnamed World", bounds: tuple[float | None, float | None] = (None, None)) -> None:
        self.entities = entities
        self.name = name
        self.bounds = bounds  # (width, height), None means unbounded

        self.time = 0

    @property
    def agents(self) -> list[Agent]:
        return [entity for entity in self.entities if isinstance(entity, Agent)]

    @property
    def width(self) -> float | None:
        return self.bounds[0]

    @property
    def height(self) -> float | None:
        return self.bounds[1]

    def update_environment(self) -> None:
        """Update all entities/states that are updatable"""
        for entity in self.entities:
            if isinstance(entity, UpdatableEntity):
                entity.update(self)

    def advance_time(self):
        self.time += 1

    def resolve_actions(self, actions: dict[Agent, Action]):

        for agent, action in actions.items():
            # Handle all supported action types
            if isinstance(action, MoveAction):
                # Get agent position (current and target)
                new_x, new_y = agent.position
                target_x, target_y = new_x + action.dx, new_y + action.dy

                # Ensure the agent is within the bounds
                if self.width is None or target_x < self.width:
                    new_x = target_x
                if self.height is None or target_y < self.height:
                    new_y = target_y

                # Update agent position
                agent.position = (new_x, new_y)
