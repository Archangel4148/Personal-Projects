from collections.abc import Sequence
import random

from simulation.actions import Action, MoveAction
from simulation.agent import Agent
from simulation.combat import CombatEntity, AttackEntityAction, DamageableEntity, resolve_combat
from simulation.entity import Entity, EntityID, MovableEntity, UpdatableEntity
from tools.math_helpers import clamp_magnitude

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

    def _get_entity(self, entity_id: EntityID) -> Entity | None:
        try:
            return next(entity for entity in self.entities if entity.id == entity_id)
        except StopIteration:
            # Entity not found
            return None

    def _remove_entity(self, entity: Entity):
        self.entities = [e for e in self.entities if e.id != entity.id]

    def update_environment(self) -> None:
        """Update all entities/states that are updatable"""
        for entity in self.entities:
            if isinstance(entity, UpdatableEntity):
                entity.update(self)

    def advance_time(self):
        self.time += 1

    def resolve_actions(self, actions: dict[Agent, Action]):

        # TODO: Find a better way to handle action resolution order, for now it's random
        resolution_order = list(actions.items())
        random.shuffle(resolution_order)

        for agent, action in resolution_order:
            # Handle all supported action types
            if isinstance(action, MoveAction):
                if not isinstance(agent, MovableEntity):
                    continue
                # Get agent position (current and target)
                new_x, new_y = agent.position
                dx, dy = clamp_magnitude(action.dx, action.dy, agent.max_speed)
                target_x, target_y = new_x + dx, new_y + dy

                # Ensure the agent is within the bounds
                if self.width is None or 0 <= target_x < self.width:
                    new_x = target_x
                if self.height is None or 0 <= target_y < self.height:
                    new_y = target_y

                # Update agent position
                agent.position = (new_x, new_y)

            elif isinstance(action, AttackEntityAction):
                if isinstance(agent, CombatEntity):
                    # Find the target
                    target = self._get_entity(action.target_id)
                    if not isinstance(target, DamageableEntity) or not target.is_alive:
                        continue

                    # Resolve the combat
                    resolve_combat(attacker=agent, target=target, attack_id=action.attack_id)

                    if not target.is_alive:
                        self._handle_death(target)

    def _handle_death(self, entity: DamageableEntity):
        self._remove_entity(entity)
