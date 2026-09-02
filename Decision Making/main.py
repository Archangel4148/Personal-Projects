from rendering.renderer import PrintRenderer
from simulation.agent import Agent
from simulation.end_conditions import TimeLimitCondition
from simulation.simulator import InstantSimulator, Simulator
from simulation.world import World


def main():
    # Create and populate the world
    agents = [Agent(name=f"Agent {i+1}") for i in range(3)]
    world = World(agents=agents, name="Titan")

    # Create a renderer to draw the state
    renderer = PrintRenderer()

    # Add end conditions (this world will end after 10 time steps)
    end_conditions = [TimeLimitCondition(time_limit=10)]

    # Create a simulator to run the simulation
    simulator = InstantSimulator(world=world, end_conditions=end_conditions, renderer=renderer)

    # Run the simulation
    simulator.run()

if __name__ == "__main__":
    main()
