from collections.abc import Sequence
import math
import random

from brains.brain import Brain
from scenarios.scenario_base import Scenario
from simulation.actions import Action, MoveAction, NoOpAction
from simulation.agent import Agent, RandomMoveAgent
from simulation.combat import Attack, AttackEntityAction, CombatEntity, DamageableEntity
from simulation.entity import Entity, EntityID, MovableEntity
from simulation.world import World
from tools.math_helpers import distance, step_towards


class HunterBrain(Brain[HunterAgent]):


    def choose_action(self, agent: Agent ) -> Action:
        # Get all valid entities that are known
        valid_targets = [
            entity
            for entity in self.knowledge
            if isinstance(entity, DamageableEntity)
            and entity.id != agent.id
        ]

        # If there are no valid targets, do nothing
        if not valid_targets:
            return NoOpAction()

        # Find the nearest valid target
        prey = min(
            valid_targets,
            key=lambda target: distance(agent.position, target.position)
        )

        distance_to_prey = distance(agent.position, prey.position)

        if distance_to_prey < agent.attack.range:
            return AttackEntityAction(
                prey.id,
                agent.attack.id,
            )

        dx, dy = step_towards(
            from_pos=agent.position,
            to_pos=prey.position,
            max_step=agent.max_speed,
        )

        return MoveAction(dx, dy)

class HunterAgent(Agent, CombatEntity, MovableEntity):

    def __init__(self, max_speed: float, attack: Attack, name: str = "Unnamed Hunter Agent", position: tuple[float, float] = (0, 0)) -> None:
        super().__init__(name=name, position=position)
        self._max_speed = max_speed
        self.attack = attack

        self.prey_id: EntityID | None = None
        self.prey_pos: tuple[float, float] | None = None

    @property
    def max_speed(self) -> float:
        return self._max_speed

    def observe(self, world: World) -> None:
        # Find the nearest valid target
        valid_targets = [e for e in world.entities if isinstance(e, DamageableEntity) and e.id != self.id]

        if not valid_targets:
            self.prey_id = None
            self.prey_pos = None
        else:
            prey = min(valid_targets, key=lambda t: distance(self.position, t.position))
            self.prey_id = prey.id
            self.prey_pos = prey.position
    
    def choose_action(self) -> Action:
        if self.prey_pos and self.prey_id is not None:
            distance_to_prey = distance(self.position, self.prey_pos)

            # Within range, attack!
            if distance_to_prey < self.attack.range:
                return AttackEntityAction(self.prey_id, self.attack.id)

            # Out of range, move towards prey
            return MoveAction(*step_towards(from_pos=self.position, to_pos=self.prey_pos, max_step=self.max_speed))

        # If there is no valid prey, move randomly:
        angle = random.random() * 2 * math.pi
        return MoveAction(dx=self._max_speed * math.cos(angle), dy=self._max_speed * math.sin(angle))

    def get_attack(self, attack_id: str) -> Attack | None:
        return self.attack if self.attack.id == attack_id else None

class HelplessPreyAgent(RandomMoveAgent, DamageableEntity):

    def __init__(self, hp: float, max_hp: float, step_distance: float, name: str = "Helpless Prey", position: tuple[float, float] = (0, 0)) -> None:
        super().__init__(step_distance, name, position)
        self.hp = hp
        self.max_hp = max_hp

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def take_damage(self, amount: float) -> None:
        self.hp -= amount



class PredatorVsPrey(Scenario):
    name: str = "Predator vs Prey"
    bounds: tuple[int, int] = (800, 600)
    prey_count: int = 5
    attack_range: float = 7.0

    def build_entities(self) -> Sequence[Entity]:
        width, height = self.bounds
        hunter = [
            HunterAgent(
                max_speed=10,
                attack=Attack(id="Bite", damage=1, range=self.attack_range),
                name="Wolf"
            )
        ]
        prey = [HelplessPreyAgent(hp=1, max_hp=1, step_distance=8, name=f"Agent {i+1}", position=(random.randint(0, width), random.randint(0, height))) for i in range(self.prey_count)]
        return hunter + prey
