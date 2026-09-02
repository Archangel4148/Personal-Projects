from rendering.renderer import PrintRenderer
from simulation.agent import LazyAgent
from simulation.end_conditions import TimeLimitCondition
from simulation.simulator import InstantSimulator, TpsSimulator
from simulation.world import World


def main():
    # Create and populate the world
    agents = [LazyAgent(name=f"Agent {i+1}", position=(20 * i, 20 * i)) for i in range(3)]
    world = World(entities=agents, name="Titan", bounds=(100, 100))

    # Create a renderer to draw the state
    renderer = PrintRenderer()

    # Add end conditions (this world will end after 10 time steps)
    end_conditions = [TimeLimitCondition(time_limit=10)]

    # Create a simulator to run the simulation
    simulator = TpsSimulator(world=world, end_conditions=end_conditions, tps=2, renderer=renderer)

    # Run the simulation
    simulator.run()

if __name__ == "__main__":
    main()
