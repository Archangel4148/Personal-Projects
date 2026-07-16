from __future__ import annotations

import sys
from collections.abc import Hashable
from dataclasses import dataclass

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
from pyvis.network import Network

from framework.base import GameModule, GameState, Action
from games.tic_tac_toe import TicTacToeModule


@dataclass(frozen=True, kw_only=True)
class GameDisplayData:
    game_name: str
    html_data: str


@dataclass
class NodeData:
    state: GameState
    depth: int
    is_terminal: bool
    is_root: bool = False
    winner: object | None = None


@dataclass
class EdgeData:
    source: Hashable
    target: Hashable
    action: Action


@dataclass
class StateGraph:
    nodes: dict[Hashable, NodeData]
    edges: list[EdgeData]


class StateVisualizationWindow(QWidget):
    def __init__(self, display_data: GameDisplayData):
        super().__init__()

        self.display_data = display_data

        self.setWindowTitle("Game State Visualization")
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)

    def update_visualization(self):
        """Update the visualization to the provided game"""
        self.setWindowTitle(f"{self.display_data.game_name} Visualization")
        html_content = self.display_data.html_data
        self.browser.setHtml(html_content)


class StateAnalyzer:
    def __init__(self, game_module: GameModule):
        self.game_module = game_module

    def visualize_states(self, traversal_depth: int = 1):
        """Create and update the visualization window"""
        width, height = 800, 800
        hpad, vpad = 25, 40

        # Build network
        graph = self.traverse_states(max_depth=traversal_depth)

        network_html = self.build_network_html(state_graph=graph, height=height)

        # Build display data
        display_data = GameDisplayData(
            game_name=self.game_module.name,
            html_data=network_html,
        )

        # Create the application/window
        app = QApplication(sys.argv)
        window = StateVisualizationWindow(display_data=display_data)
        window.resize(width + hpad, height + vpad)
        window.update_visualization()
        window.show()
        sys.exit(app.exec())

    def traverse_states(self, max_depth: int) -> StateGraph:
        game = self.game_module

        # Start with only the initial state
        initial_state = game.setup_initial_state(config={})
        initial_hash = game.hash_state(initial_state)

        nodes: dict[Hashable, NodeData] = {
            initial_hash: NodeData(
                state=initial_state,
                depth=0,
                is_terminal=game.is_game_over(initial_state)[0],
                is_root=True,
            )
        }
        edges: list[EdgeData] = []

        current_depth_states = {initial_hash: initial_state}

        # Search up to maximum depth
        for depth in range(max_depth):
            next_depth_states = {}

            for hashed_state, state in current_depth_states.items():
                # Do not search ended games
                if game.is_game_over(state)[0]:
                    continue

                # Take each available action, adding any new states encountered to the list
                for action in game.get_legal_actions(state):
                    new_state = game.apply_action(state, action)
                    new_hash = game.hash_state(new_state)

                    # Record the connection
                    edges.append(
                        EdgeData(
                            source=hashed_state,
                            target=new_hash,
                            action=action,
                        )
                    )

                    if new_hash not in nodes:
                        nodes[new_hash] = NodeData(
                            state=new_state,
                            depth=depth + 1,
                            is_terminal=game.is_game_over(new_state)[0],
                        )

                        next_depth_states[new_hash] = new_state

            current_depth_states = next_depth_states

        return StateGraph(nodes=nodes, edges=edges)

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

    @staticmethod
    def build_network_html(state_graph: StateGraph, height: int = 800,
                           width: int = 800) -> str:
        network = Network(height=height, width=width, directed=True)

        node_ids = {
            state_hash: i
            for i, state_hash in enumerate(state_graph.nodes)
        }

        # Nodes
        for state_hash, data in state_graph.nodes.items():
            if data.is_root:
                color = "lightgreen"
            elif data.is_terminal:
                color = "red"
            else:
                color = "lightblue"

            network.add_node(
                node_ids[state_hash],
                # label=str(data.state),
                title=(
                    f"Depth: {data.depth}\n"
                    f"Terminal: {data.is_terminal}"
                ),
                size=30 if data.is_terminal else 10,
                color=color,
            )

        # Edges
        for edge in state_graph.edges:
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
        network.toggle_physics(False)

        return network.generate_html()


def main():
    game = TicTacToeModule()
    analyzer = StateAnalyzer(game)

    analyzer.visualize_states(traversal_depth=5)


if __name__ == '__main__':
    main()
