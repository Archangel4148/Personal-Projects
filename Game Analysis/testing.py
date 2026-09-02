# Build and initialize a game module
from collections import defaultdict

from framework.agents import Agent, RandomAgent, SolvedGraphAgent
from framework.runner import GameRunner
from framework.state_analysis.reduction import SymmetryEquivalence
from framework.state_analysis.solver import StateSolver
from framework.state_analysis.state_analysis import StateGraphBuilder
from framework.state_analysis.transforms import FlipOverAntiDiagonal, FlipOverHorizontalAxis, FlipOverVerticalAxis, PermuteGroupsTransform, Rotate180, Rotate270, Rotate90, Transpose
from games.chopsticks import ChopsticksModule, SortHandsTransform
from games.nim import NimModule
from games.tic_tac_toe import TicTacToeModule

# Create the game, and define symmetries to prune
game = ChopsticksModule()
# game = TicTacToeModule()
equivalence = SymmetryEquivalence(
        # FlipOverHorizontalAxis(),
        # FlipOverVerticalAxis(),
        # Rotate90(),
        # Rotate180(),
        # Rotate270(),
        # Transpose(),
        # FlipOverAntiDiagonal(),
        # PermuteGroupsTransform()
        # SortHandsTransform()
    )

# Build the state graph
builder = StateGraphBuilder(game=game, equivalence=equivalence)
graph = builder.traverse_states(max_depth=1000, include_module=True)

# Build and run the solver
solver = StateSolver(graph, game)
solver.solve()

# Create some agents: one random player, and one solved player
players: list = [SolvedGraphAgent(solver, equivalence), RandomAgent()]
# players: list = [SolvedGraphAgent(solver, equivalence), SolvedGraphAgent(solver, equivalence)]

# Run the game!
runner = GameRunner(game, players)
win_counts = defaultdict(int)

iterations = 10000
actions = 0
for i in range(iterations):
    if i % 1000 == 0:
        print(i, "/", iterations)
    results = runner.run_game()
    win_counts[tuple(results["winner_indices"])] += 1
    actions += results["actions_taken"]

print("Win Counts:", win_counts)
print("Average Action Count:", actions / iterations)

# print("Results :", results)
# print("\nBoard State:")
# display = game.render_state(results["final_state"])
# print(display)