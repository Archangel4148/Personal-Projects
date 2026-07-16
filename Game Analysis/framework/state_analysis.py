from __future__ import annotations

import sys
from dataclasses import dataclass

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
from pyvis.network import Network

from framework.base import GameModule
from games.tic_tac_toe import TicTacToeModule


@dataclass(frozen=True, kw_only=True)
class GameDisplayData:
    game_name: str
    html_data: str

    @classmethod
    def from_module(cls, module: GameModule, html: str) -> GameDisplayData:
        return cls(
            game_name=module.name,
            html_data=html,
        )


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

    def visualize_states(self):
        """Create and update the visualization window"""
        # Build display data
        display_data = GameDisplayData.from_module(self.game_module, self.build_network_html())

        # Create the application/window
        app = QApplication(sys.argv)
        window = StateVisualizationWindow(display_data=display_data)
        window.update_visualization()
        window.show()
        sys.exit(app.exec())

    def build_network_html(self) -> str:
        network = Network(height=800, width=800)
        network.add_nodes(nodes=[1, 2, 3, 4])
        network.add_edges(edges=[(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
        network.barnes_hut()
        return network.generate_html()


def main():
    game = TicTacToeModule()
    analyzer = StateAnalyzer(game)

    analyzer.visualize_states()


if __name__ == '__main__':
    main()
