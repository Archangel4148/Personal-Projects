from framework.agents import RandomAgent, SolvedGraphAgent
from framework.runner import GameRunner
from framework.state_analysis.reduction import SymmetryEquivalence
from framework.state_analysis.solver import StateSolver
from framework.state_analysis.state_analysis import StateGraphBuilder
from framework.state_analysis.transforms import PermuteGroupsTransform, FlipOverHorizontalAxis, FlipOverVerticalAxis, \
    Rotate90, Rotate180, Rotate270, Transpose, FlipOverAntiDiagonal
from games.chopsticks import SortHandsTransform
from games.nim import NimModule
from games.tic_tac_toe import TicTacToeModule

game = TicTacToeModule()
equivalence = SymmetryEquivalence(
    FlipOverHorizontalAxis(),
    FlipOverVerticalAxis(),
    Rotate90(),
    Rotate180(),
    Rotate270(),
    Transpose(),
    FlipOverAntiDiagonal(),
    # PermuteGroupsTransform(),
    # SortHandsTransform(),
)

# Build the state graph
builder = StateGraphBuilder(game=game, equivalence=equivalence)
graph = builder.traverse_states(max_depth=1000, include_module=True)

# Run the solver
solver = StateSolver(graph=graph, game=game)
solver.solve()

# Create players
players = [RandomAgent(), SolvedGraphAgent(solver=solver, equivalence=equivalence)]

# Run games
runner = GameRunner(game_module=game, agents=players)
iterations = 1000

wins = {}
for _ in range(iterations):
    results = runner.run_game(config={})
    winner = str(results["winner_indices"])
    if winner in wins:
        wins[winner] += 1
    else:
        wins[winner] = 1

# Show results
print(wins)
