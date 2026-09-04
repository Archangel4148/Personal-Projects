from collections.abc import Sequence
import random

from brains.brain import Brain, wander
from brains.senses.sense import EntitySense
from scenarios.scenario_base import Scenario
from simulation.actions import Action, MoveAction
from simulation.agent import Agent, RandomMoveAgent
from simulation.combat import Attack, AttackEntityAction, CombatEntity, DamageableEntity
from simulation.entity import Entity, MovableEntity
from tools.math_helpers import distance, step_towards


class HunterBrain(Brain):

    def choose_action(self, agent: Agent) -> Action:
        assert isinstance(agent, HunterAgent)
        # Get all valid entities that are known
        valid_targets = [
            entity
            for entity in self.knowledge
            if isinstance(entity, DamageableEntity)
            and entity.id != agent.id
        ]

        if not valid_targets:
            return wander(agent)

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
        super().__init__(
            brain=HunterBrain(senses=[EntitySense(range=200)]),  # This hunter can perfectly sense every Entity in the world
            name=name,
            position=position
        )
        self._max_speed = max_speed
        self.attack = attack

    @property
    def max_speed(self) -> float:
        return self._max_speed

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
        prey = [HelplessPreyAgent(hp=1, max_hp=1, step_distance=6, name=f"Agent {i+1}", position=(random.randint(0, width), random.randint(0, height))) for i in range(self.prey_count)]
        return hunter + prey
