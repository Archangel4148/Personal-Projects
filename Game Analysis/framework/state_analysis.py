from __future__ import annotations

import math
import sys
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass

from PyQt6.QtWidgets import QApplication
from pyvis.network import Network

from framework.base import GameModule, GameState, Action
from framework.state_analysis.reduction import StateEquivalence, SymmetryEquivalence
from framework.state_analysis.rendering import GameDisplayData, StateVisualizationWindow
from framework.state_analysis.transforms import FlipOverHorizontalAxis, FlipOverVerticalAxis, PermuteGroupsTransform, \
    Rotate180, Rotate90, Rotate270, Transpose, FlipOverAntiDiagonal
from games.connect_four import ConnectFourModule
from games.nim import NimModule
from games.tic_tac_toe import TicTacToeModule


@dataclass
class NodeData:
    state: GameState
    depth: int
    is_terminal: bool
    is_root: bool = False
    winner: str | None = None


@dataclass
class EdgeData:
    source: Hashable
    target: Hashable
    action: Action


@dataclass
class StateGraph:
    nodes: dict[Hashable, NodeData]
    edges: list[EdgeData]
    game: GameModule | None = None  # Optional: allows nice rendering!


@dataclass
class StateGraphStatistics:
    # Existing Metrics
    total_states: int
    average_branching_factor: float
    state_density: float
    graph_diameter: float
    dead_end_count: int
    unreachable_state_count: int

    # Decision Space Metrics
    min_branching_factor: float
    max_branching_factor: float
    branching_factor_std_dev: float
    max_depth: int

    # Topological Metrics
    reconvergent_node_count: int
    has_cycles: bool

    # Game Balance Metrics
    terminal_state_distribution: dict[str, int]
    average_depth_to_terminal: dict[str, float]

    def print_summary(self) -> None:
        """Prints a structured dashboard of the state graph statistics."""
        border = "=" * 50
        section_border = "-" * 50

        print(border)
        print(f"{'STATE GRAPH ANALYSIS REPORT':^50}")
        print(border)

        print("\n[ General Graph Metrics ]")
        print(f"  Total States (Nodes):      {self.total_states:,}")
        print(f"  State Density (Nodes/Edge): {self.state_density:.4f}")
        print(f"  Graph Diameter (Proxy):     {self.graph_diameter:.1f}")
        print(f"  Dead Ends (Non-terminal):   {self.dead_end_count:,}")
        print(f"  Unreachable States:         {self.unreachable_state_count:,}")

        print(section_border)

        print("\n[ Decision Space & Complexity ]")
        print(f"  Max Depth Reached:          {self.max_depth:,}")
        print(f"  Average Branching Factor:   {self.average_branching_factor:.2f}")
        print(f"  Min Branching Factor:       {self.min_branching_factor:.1f}")
        print(f"  Max Branching Factor:       {self.max_branching_factor:.1f}")
        print(f"  Branching Std Deviation:    {self.branching_factor_std_dev:.2f}")

        print(section_border)

        print("\n[ Topology & Structure ]")
        print(f"  Reconvergent Nodes (In>1):  {self.reconvergent_node_count:,}")
        print(f"  Has Cycles:                 {'Yes (Looping Detected)' if self.has_cycles else 'No (DAG / Tree)'}")

        print(section_border)

        print("\n[ Game Balance & Outcomes ]")
        print("  Terminal State Distribution:")
        if self.terminal_state_distribution:
            for winner, count in self.terminal_state_distribution.items():
                pct = (count / sum(self.terminal_state_distribution.values())) * 100
                print(f"    - Winner '{winner}': {count:,} ({pct:.1f}%)")
        else:
            print("    No terminal states recorded.")

        print("\n  Average Depth to Terminal:")
        if self.average_depth_to_terminal:
            for winner, avg_depth in self.average_depth_to_terminal.items():
                print(f"    - Winner '{winner}': {avg_depth:.2f} plies")
        else:
            print("    N/A")

        print(border)


class StateGraphBuilder:
    def __init__(
            self,
            game: GameModule,
            equivalence: StateEquivalence,
    ):
        self.game_module = game
        self.equivalence = equivalence

    def traverse_states(self, max_depth: int, include_module: bool = False) -> StateGraph:
        print(f"Beginning state traversal with maximum depth {max_depth}")
        game = self.game_module

        # Start with only the initial state
        initial_state = game.setup_initial_state(config={})
        canonical = self.equivalence.canonical(initial_state, game)
        initial_key = game.state_key(canonical)

        nodes: dict[Hashable, NodeData] = {
            initial_key: NodeData(
                state=canonical,
                depth=0,
                is_terminal=game.is_game_over(initial_state)[0],
                is_root=True,
            )
        }
        edges: list[EdgeData] = []

        current_depth_states = {initial_key: initial_state}

        # Search up to maximum depth
        for depth in range(max_depth):
            if current_depth_states:
                print(f"Processing depth {depth + 1}")
            next_depth_states = {}

            for state_key, state in current_depth_states.items():
                # Do not search ended games
                if game.is_game_over(state)[0]:
                    continue

                # Take each available action, adding any new states encountered to the list
                for action in game.get_legal_actions(state):
                    new_state = game.apply_action(state, action)
                    canonical = self.equivalence.canonical(new_state, game)
                    new_key = game.state_key(canonical)

                    # Record the connection
                    edges.append(
                        EdgeData(
                            source=state_key,
                            target=new_key,
                            action=action,
                        )
                    )

                    if new_key not in nodes:
                        is_terminal, winner_idx = game.is_game_over(new_state)
                        new_node = NodeData(
                            state=self.equivalence.canonical(new_state, game),
                            depth=depth + 1,
                            is_terminal=is_terminal,
                        )
                        if winner_idx:
                            new_node.winner = str(winner_idx)

                        nodes[new_key] = new_node

                        next_depth_states[new_key] = new_state

            current_depth_states = next_depth_states

        state_graph = StateGraph(nodes=nodes, edges=edges, game=self.game_module if include_module else None)
        return state_graph


class StateGraphVisualizer:
    def __init__(self, graph: StateGraph):
        self.graph = graph

    def render(self, window_title: str = "Game State Visualization", width: int = 800, height: int = 800, hpad: int = 0,
               vpad: int = 0, use_physics: bool = True):
        """Create and update the visualization window"""

        if self.graph is None:
            raise ValueError("State graph must be initialized.")

        network_html = self.build_network_html(height=height, width=width, use_physics=use_physics)

        # Build display data
        display_data = GameDisplayData(
            window_title=window_title,
            html_data=network_html,
        )

        # Create the application/window
        app = QApplication(sys.argv)
        window = StateVisualizationWindow(display_data=display_data)
        window.resize(width + hpad, height + vpad)
        window.update_visualization()
        window.show()
        sys.exit(app.exec())

    @staticmethod
    def build_graph_features(
            nodes: list[Hashable],
            edges: list[tuple[Hashable, Action, Hashable]],
    ) -> tuple[dict[Hashable, int], list[tuple[int, Action, int]]]:
        node_ids = {h: i for i, h in enumerate(nodes)}
        clean_edges = [
            (node_ids[src], action, node_ids[dst])
            for src, action, dst in edges
        ]
        return node_ids, clean_edges

    def build_network_html(self, height: int = 800, width: int = 800, use_physics: bool = True) -> str:
        network = Network(height=height, width=width, directed=True)

        node_ids = {
            state_key: i
            for i, state_key in enumerate(self.graph.nodes)
        }

        terminal_palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
        outcome_colors: dict[str, str] = {}
        palette_index = 0

        # Nodes
        for state_key, data in self.graph.nodes.items():
            if data.is_root:
                color = "lightgreen"
            elif data.is_terminal:
                # Fallback label if data.winner is None (indicating a draw)
                outcome = data.winner if data.winner is not None else "Draw"
                
                # Dynamically assign a color from the palette if it's the first time seeing this outcome
                if outcome not in outcome_colors:
                    if outcome == "Draw":
                        outcome_colors[outcome] = "#7f7f7f"  # Give draws a neutral gray color
                    else:
                        outcome_colors[outcome] = terminal_palette[palette_index % len(terminal_palette)]
                        palette_index += 1
                
                color = outcome_colors[outcome]
            else:
                color = "lightblue"

            if self.graph.game is not None:
                title_str = f"{self.graph.game.render_state(data.state)}\n"
            else:
                title_str = ""
            title_str += f"Depth: {data.depth}\nTerminal: {data.is_terminal}\nWinner: {data.winner if data.winner else 'None'}"

            network.add_node(
                node_ids[state_key],
                title=title_str,
                size=30 if data.is_terminal else 10,
                color=color,
            )

        # Edges
        for edge in self.graph.edges:
            network.add_edge(
                node_ids[edge.source],
                node_ids[edge.target],
                title=str(edge.action),
            )

        network.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=100,
        )
        network.toggle_physics(use_physics)

        return network.generate_html()


class StateGraphAnalyzer:
    def __init__(self, graph: StateGraph):
        self.graph = graph

    def analyze(self) -> StateGraphStatistics:
        if not self.graph:
            raise ValueError("No graph has been traversed yet. Run traverse_states first.")
        print("Beginning state analysis")
        graph = self.graph
        nodes = graph.nodes
        edges = graph.edges

        total_states = len(nodes)
        if total_states == 0:  # Handle empty graphs
            return StateGraphStatistics()

        out_degrees = defaultdict(int)
        in_degrees = defaultdict(int)

        # Find node in/out degrees (reconvergence and branching)
        for node_hash in nodes:
            out_degrees[node_hash] = 0
            in_degrees[node_hash] = 0

        for edge in edges:
            out_degrees[edge.source] += 1
            in_degrees[edge.target] += 1

        # Only count branching factor for non-terminal nodes
        non_terminal_branching = [
            count for node_hash, count in out_degrees.items()
            if not nodes[node_hash].is_terminal
        ]

        if non_terminal_branching:
            min_branching = min(non_terminal_branching)
            max_branching = max(non_terminal_branching)
            avg_branching = sum(non_terminal_branching) / len(non_terminal_branching)

            # Standard deviation
            variance = sum((x - avg_branching) ** 2 for x in non_terminal_branching) / len(non_terminal_branching)
            branching_std_dev = math.sqrt(variance)
        else:
            min_branching = max_branching = avg_branching = branching_std_dev = 0.0

        state_density = total_states / len(edges) if len(edges) > 0 else 0.0

        dead_end_count = sum(
            1 for node_hash, count in out_degrees.items()
            if count == 0 and not nodes[node_hash].is_terminal
        )

        # Find any unreachable nodes (besides the root)
        unreachable_count = sum(
            1 for node_hash, count in in_degrees.items()
            if count == 0 and not nodes[node_hash].is_root
        )

        # Count transpositions (states with multiple paths to them)
        reconvergent_node_count = sum(1 for count in in_degrees.values() if count > 1)

        max_depth = 0
        terminal_distribution = defaultdict(int)
        depths_per_winner = defaultdict(list)
        for node in nodes.values():
            if node.depth > max_depth:
                max_depth = node.depth

            if node.is_terminal:
                winner_key = node.winner if node.winner is not None else "Draw/None"
                terminal_distribution[winner_key] += 1
                depths_per_winner[winner_key].append(node.depth)

        # Average depth to win/loss
        avg_depth_to_terminal = {
            winner: sum(depths) / len(depths)
            for winner, depths in depths_per_winner.items()
        }

        # This is a simplified estimation! (Full cycle checking would be expensive)
        has_cycles = any(edge.source == edge.target for edge in edges)

        return StateGraphStatistics(
            total_states=total_states,
            average_branching_factor=avg_branching,
            state_density=state_density,
            graph_diameter=float(max_depth),  # Using max depth reached as tree-diameter proxy
            dead_end_count=dead_end_count,
            unreachable_state_count=unreachable_count,
            min_branching_factor=min_branching,
            max_branching_factor=max_branching,
            branching_factor_std_dev=branching_std_dev,
            max_depth=max_depth,
            reconvergent_node_count=reconvergent_node_count,
            has_cycles=has_cycles,
            terminal_state_distribution=dict(terminal_distribution),
            average_depth_to_terminal=avg_depth_to_terminal
        )


def main():
    # Build the state graph
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
    )
    builder = StateGraphBuilder(game=game, equivalence=equivalence)
    graph = builder.traverse_states(max_depth=20, include_module=True)

    # Analyze the state graph
    analyzer = StateGraphAnalyzer(graph)
    stats = analyzer.analyze()
    stats.print_summary()

    # Visualize the state graph
    visualizer = StateGraphVisualizer(graph)
    visualizer.render(width=1920, height=1080, hpad=35, vpad=40, use_physics=True)


if __name__ == '__main__':
    main()
