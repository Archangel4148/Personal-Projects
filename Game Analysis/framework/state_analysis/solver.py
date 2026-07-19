

from collections import defaultdict, deque
from enum import Enum, IntEnum
from typing import Any, Hashable

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
        self.distances: dict[Hashable, int | float] = {}
        self.optimal_moves: dict[Hashable, Hashable | None] = {}
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

                self.distances[key] = 0
                self.optimal_moves[key] = None
                queue.append(key)
                unresolved_children[key] = 0

            else:
                unresolved_children[key] = len(children[key])

                # If a node is not terminal but has no children, treat it as a draw (to avoid errors)
                if unresolved_children[key] == 0:
                    self.values[key] = draw_vector
                    self.distances[key] = 0
                    self.optimal_moves[key] = None
                    queue.append(key)
        
        # Propagate values upwards (retrograde analysis)
        while queue:
            current_key = queue.popleft()
            current_vector = self.values[current_key]
            current_dist = self.distances[current_key]

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
                    self.distances[parent] = current_dist + 1
                    self.optimal_moves[parent] = current_key
                    queue.append(parent)
                    continue

                unresolved_children[parent] -= 1

                if unresolved_children[parent] == 0:
                    def max_n_key(c_key):
                        vec = self.values[c_key]
                        d = self.distances[c_key]
                        payoff = vec[active_player]
                        
                        # Tie-breaking logic:
                        # WIN -> Minimize distance (higher -d is better)
                        # LOSS -> Maximize distance (higher d is better)
                        # DRAW -> Minimize distance (higher -d is better)
                        if payoff == StateValue.WIN:
                            return (payoff, -d)
                        elif payoff == StateValue.LOSS:
                            return (payoff, d)
                        else:
                            return (payoff, -d)
                    
                    # Select the child vector that maximizes the active player's payoff (max-n rule)
                    best_child = max(children[parent], key=max_n_key)
                    
                    self.values[parent] = self.values[best_child]
                    self.distances[parent] = self.distances[best_child] + 1
                    self.optimal_moves[parent] = best_child
                    queue.append(parent)
        
        # Anything left over is stuck in an infinite loop without a forced win
        for key in self.graph.nodes:
            if key not in self.values:
                parent_node = self.graph.nodes[key]
                if hasattr(parent_node, 'current_player') and parent_node.current_player is not None:
                    active_player = parent_node.current_player
                else:
                    active_player = self.game.get_current_player_idx(parent_node.state)

                def post_solve_key(c_key):
                    # If child is in a cycle too, treat it as an infinite-length draw loop
                    vec = self.values[c_key] if c_key in self.values else draw_vector
                    d = self.distances[c_key] if c_key in self.values else float('inf')
                    
                    payoff = vec[active_player]
                    if payoff == StateValue.WIN:
                        return (payoff, -d)
                    elif payoff == StateValue.LOSS:
                        return (payoff, d)
                    else:
                        return (payoff, -d)

                if children[key]:
                    best_child = max(children[key], key=post_solve_key)
                    self.optimal_moves[key] = best_child
                    
                    if best_child in self.values:
                        self.values[key] = self.values[best_child]
                        self.distances[key] = self.distances[best_child] + 1
                    else:
                        self.values[key] = draw_vector
                        self.distances[key] = float('inf')
                else:
                    self.values[key] = draw_vector
                    self.distances[key] = 0
                    self.optimal_moves[key] = None

        return self.values

    def get_readable_summary(self, key: Hashable, scan_action: bool = False) -> dict[str, Any]:
        """Generates a clean human-readable analysis profile for a specific state."""
        if key not in self.values:
            return {"Error": "State key not found."}
            
        vector = self.values[key]
        readable_vector = [(f"Player {idx}", val.name) for idx, val in enumerate(vector)]
        next_state = self.optimal_moves[key]

        # Find the action on the edge connecting key -> next_state
        optimal_action = None
        if scan_action:
            if next_state is not None:
                for edge in self.graph.edges:
                    if edge.source == key and edge.target == next_state:
                        optimal_action = edge.action
                        break


        return {
            "payoffs": readable_vector,
            "distance_to_outcome": self.distances[key],
            "has_optimal_move": self.optimal_moves[key] is not None,
            "next_state_key": next_state,
            "optimal_action": optimal_action
        }


if __name__ == "__main__":
    
    game = NimModule()
    equivalence = SymmetryEquivalence(
        # FlipOverHorizontalAxis(),
        # FlipOverVerticalAxis(),
        # Rotate90(),
        # Rotate180(),
        # Rotate270(),
        # Transpose(),
        # FlipOverAntiDiagonal(),
        PermuteGroupsTransform()
        # SortHandsTransform()
    )
    builder = StateGraphBuilder(game=game, equivalence=equivalence)
    graph = builder.traverse_states(max_depth=1000, include_module=True)

    # Run the solver
    solver = StateSolver(graph, game)
    solver.solve()

    root = next(key for key, node in graph.nodes.items() if node.is_root)

    summary = solver.get_readable_summary(root, scan_action=True)
    print("--- Root State Analysis ---")
    print(f"Results:         {summary['payoffs']}")
    print(f"Plies to End:    {summary['distance_to_outcome']}")
    print(f"Next State Key:  {summary['next_state_key']}")
    print(f"Optimal Action:  {summary['optimal_action']}")