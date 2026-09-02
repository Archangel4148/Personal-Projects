from scenarios.predator_vs_prey import PredatorVsPrey
from rendering.pygame import PygameRenderer
from simulation.simulator import InstantSimulator, TpsSimulator


def main():
    # Choose a scenario
    scenario = PredatorVsPrey("Predator vs. Prey", bounds=(800, 600))

    # Create and populate the world from the scenario
    world = scenario.build_world()

    # Create a renderer to draw the state
    renderer = PygameRenderer(window_size=scenario.bounds, bg_color=(255, 255, 255))


    # Create a simulator to run the simulation
    simulator = TpsSimulator(world=world, end_conditions=scenario.end_conditions, tps=scenario.tps, renderer=renderer)

    # Run the simulation
    simulator.run()

if __name__ == "__main__":
    main()
