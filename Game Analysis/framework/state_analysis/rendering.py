from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QWidget, QVBoxLayout


@dataclass(frozen=True, kw_only=True)
class GameDisplayData:
    window_title: str
    html_data: str


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
        self.setWindowTitle(self.display_data.window_title)
        html_content = self.display_data.html_data
        self.browser.setHtml(html_content)
