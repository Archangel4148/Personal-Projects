from abc import ABC, abstractmethod
from dataclasses import dataclass

from simulation.actions import Action
from simulation.entity import Entity, EntityID
from tools.math_helpers import distance

@dataclass(frozen=True)
class Attack:
    id: str
    damage: float
    range: float


class DamageableEntity(Entity, ABC):
    hp: float
    max_hp: float

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        ...

    @abstractmethod
    def take_damage(self, amount: float) -> None:
        ...


class CombatEntity(Entity, ABC):
    @abstractmethod
    def get_attack(self, attack_id: str) -> Attack | None:
        ...


class AttackEntityAction(Action):
    def __init__(self, target_id: EntityID, attack_id: str) -> None:
        self.target_id = target_id
        self.attack_id = attack_id


def resolve_combat(attacker: CombatEntity, target: DamageableEntity, attack_id: str):
    # Get the incoming attack
    attack = attacker.get_attack(attack_id)
    if attack is None:
        return

    # Handle attack range
    if distance(attacker.position, target.position) > attack.range:
        return

    # If the target is in range, they take damage
    target.take_damage(attack.damage)