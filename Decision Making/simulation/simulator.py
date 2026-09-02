from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

from rendering.renderer import Renderer
from simulation.end_conditions import EndCondition
from simulation.world import World

if TYPE_CHECKING:
    from simulation.agent import Agent


class Simulator(ABC):

    def __init__(self, world: World, end_conditions: Sequence[EndCondition], renderer: Renderer | None = None) -> None:
        self.world = world
        self.end_conditions = end_conditions
        self.renderer = renderer

    def should_end(self) -> bool:
        return any(
            condition.should_end(self.world)
            for condition in self.end_conditions
        )
    
    def step(self) -> None:
        # Update the world state
        self.world.update_environment()

        # All agents observe the world
        for agent in self.world.agents:
            agent.observe(self.world)

        # All agents choose actions
        actions = {agent: agent.choose_action() for agent in self.world.agents}

        # Resolve actions
        self.world.resolve_actions(actions)

        # Increment time
        self.world.advance_time()

        if self.renderer:
            # Render the new simulation state
            self.renderer.draw(self.world)

    @abstractmethod
    def run(self) -> None:
        ...


class InstantSimulator(Simulator):

    def run(self) -> None:
        # Step until the world ends (no delays)
        while not self.should_end():
            self.step()


class TpsSimulator(Simulator):
    def __init__(self, world: World, end_conditions: Sequence[EndCondition], tps: float, renderer: Renderer | None = None) -> None:
        super().__init__(world, end_conditions, renderer)

        if tps <= 0:
            raise ValueError("TPS must be greater than zero")

        self.tps = tps

    def run(self) -> None:
            from time import perf_counter, sleep

            tick_duration = 1 / self.tps
            while not self.should_end():
                # Track start time
                start = perf_counter()

                # Perform the step
                self.step()

                # Sleep for the remaining time to ensure accurate TPS
                elapsed = perf_counter() - start
                remaining = tick_duration - elapsed
                if remaining > 0:
                    sleep(remaining)
