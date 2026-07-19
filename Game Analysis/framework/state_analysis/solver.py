

from collections import defaultdict, deque
from enum import Enum, IntEnum
from typing import Hashable

from framework.base import GameModule
from framework.state_analysis.reduction import SymmetryEquivalence
from framework.state_analysis.state_analysis import StateGraph, StateGraphBuilder
from framework.state_analysis.transforms import FlipOverAntiDiagonal, FlipOverHorizontalAxis, FlipOverVerticalAxis, PermuteGroupsTransform, Rotate180, Rotate270, Rotate90, Transpose
from games.chopsticks import ChopsticksModule, SortHandsTransform
from games.nim import NimModule
from games.tic_tac_toe import TicTacToeModule


class StateValue(IntEnum):
    WIN = 1
    LOSS = -1
    DRAW = 0

class StateSolver:
    def __init__(self, graph: StateGraph, game: GameModule) -> None:
        self.graph = graph
        self.game = game
        
        self.values: dict[Hashable, tuple[StateValue]] = {}
        self.num_players = self._detect_player_count()

    def _detect_player_count(self) -> int:
        """Dynamically find the total number of players."""
        max_player_idx = 1
        for node in self.graph.nodes.values():
            if hasattr(node, "current_player") and node.current_player is not None:
                max_player_idx = max(max_player_idx, node.current_player)
            else:
                try:
                    p_idx = self.game.get_current_player_idx(node.state)
                    max_player_idx = max(max_player_idx, p_idx)
                except AttributeError:
                    pass
        return max_player_idx + 1

    def build_graph_links(self) -> tuple[defaultdict, defaultdict]:
        """Build both directional adjacency matrices for the graph"""
        children = defaultdict(list)
        parents = defaultdict(list)

        for edge in self.graph.edges:
            children[edge.source].append(edge.target)
            parents[edge.target].append(edge.source)
        
        return children, parents

    def solve(self):
        children, parents = self.build_graph_links()

        queue = deque()
        unresolved_children = {}

        draw_vector: tuple[StateValue] = tuple([StateValue.DRAW] * self.num_players)

        # Initialize all nodes and add base cases (terminals and dead ends)
        for key, node in self.graph.nodes.items():
            if node.is_terminal:
                winner = node.winner

                # If nobody won, it's a draw
                if winner is None:
                    self.values[key] = draw_vector
                else:
                    winner_idx = int(winner.strip("[]"))

                    current_player = node.current_player

                    # Winner gets 1.0, everyone else gets -1.0
                    payoff = [StateValue.LOSS] * self.num_players
                    if 0 <= winner_idx < self.num_players:
                        payoff[winner_idx] = StateValue.WIN
                    self.values[key] = tuple(payoff)

                queue.append(key)
                unresolved_children[key] = 0

            else:
                unresolved_children[key] = len(children[key])

                # If a node is not terminal but has no children, treat it as a draw (to avoid errors)
                if unresolved_children[key] == 0:
                    self.values[key] = draw_vector
                    queue.append(key)
        
        # Propagate values upwards (retrograde analysis)
        while queue:
            current_key = queue.popleft()
            current_vector = self.values[current_key]

            for parent in parents[current_key]:
                # Skip parents that have already found a path
                if parent in self.values:
                    continue

                parent_node = self.graph.nodes[parent]

                if hasattr(parent_node, 'current_player') and parent_node.current_player is not None:
                    active_player = parent_node.current_player
                else:
                    active_player = self.game.get_current_player_idx(parent_node.state)

                # If this child gives the current player a win, they will pick it
                if current_vector[active_player] == StateValue.WIN:
                    self.values[parent] = current_vector
                    queue.append(parent)
                    continue

                unresolved_children[parent] -= 1

                # If all children are evaluated and none were a loss
                if unresolved_children[parent] == 0:
                    def get_child_vector(c):
                        return self.values[c] if c in self.values else draw_vector
                    
                    # Select the child vector that maximizes the active player's payoff (max-n rule)
                    best_vector = max(
                        [get_child_vector(c) for c in children[parent]],
                        key=lambda vector: vector[active_player]
                    )
                    
                    self.values[parent] = best_vector
                    queue.append(parent)
        
        # Anything left over is stuck in an infinite loop without a forced win
        for key in self.graph.nodes:
            if key not in self.values:
                self.values[key] = draw_vector

        return self.values

if __name__ == "__main__":
    
    game = TicTacToeModule()
    equivalence = SymmetryEquivalence(
        FlipOverHorizontalAxis(),
        FlipOverVerticalAxis(),
        Rotate90(),
        Rotate180(),
        Rotate270(),
        Transpose(),
        FlipOverAntiDiagonal(),
        # PermuteGroupsTransform()
        # SortHandsTransform()
    )
    builder = StateGraphBuilder(game=game, equivalence=equivalence)
    graph = builder.traverse_states(max_depth=1000, include_module=True)

    solver = StateSolver(graph, game)
    values = solver.solve()

    root = next(
        key for key, node in graph.nodes.items()
        if node.is_root
    )

    print([(f"Player {idx}", value.name) for idx, value in enumerate(values[root])])