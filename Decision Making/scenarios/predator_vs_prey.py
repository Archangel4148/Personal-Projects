from collections.abc import Sequence
import math
import random

from scenarios.scenario_base import Scenario
from simulation.actions import Action, MoveAction
from simulation.agent import Agent, RandomMoveAgent
from simulation.combat import Attack, AttackEntityAction, CombatEntity, DamageableEntity
from simulation.entity import Entity, EntityID, MovableEntity
from simulation.world import World
from tools.math import distance, step_towards


class HunterAgent(Agent, CombatEntity, MovableEntity):

    def __init__(self, max_speed: float, attack: Attack, name: str = "Unnamed Hunter Agent", position: tuple[float, float] = (0, 0)) -> None:
        super().__init__(name, position)
        self._max_speed = max_speed
        self.attack = attack

        self.prey_id: EntityID | None = None
        self.prey_pos: tuple[float, float] | None = None

    @property
    def max_speed(self) -> float:
        return self._max_speed

    def observe(self, world: World) -> None:
        # Find the nearest valid target
        valid_targets = [e for e in world.entities if isinstance(e, DamageableEntity) and not e.id == self.id]

        if not valid_targets:
            self.prey_id = None
            self.prey_pos = None
        else:
            prey = min(valid_targets, key=lambda t: distance(self.position, t.position))
            self.prey_id = prey.id
            self.prey_pos = prey.position
    
    def choose_action(self) -> Action:
        if self.prey_pos and self.prey_id:
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
        hunter = [
            HunterAgent(
                max_speed=10,
                attack=Attack(id="Bite", damage=1, range=6.0),
                name="Wolf"
            )
        ]
        prey = [HelplessPreyAgent(hp=1, max_hp=1, step_distance=5, name=f"Agent {i+1}", position=(random.randint(0, 800), random.randint(0, 600))) for i in range(5)]
        return hunter + prey
