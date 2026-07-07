from typing import Any, NewType

from framework.agents import Agent
from framework.base import GameModule

GameResult = NewType("GameResult", dict[str, Any])


class GameRunner:
    def __init__(self, game_module: GameModule, agents: list[Agent]):
        """Args:
            game_module: The game module containing a rules engine for the chosen game
            agents: A list of players
        """
        self.game = game_module
        self.agents = agents

    def run_game(self, config: dict[str, Any] | None = None) -> GameResult:
        """Execute a single complete game, returning a summary of the results"""
        state = self.game.setup_initial_state(config)
        actions_taken = 0

        # Main game loop
        while True:
            # Check if the game is over
            game_over, winners = self.game.is_game_over(state)
            if game_over:
                break

            # Get the current player (Agent)
            current_agent_idx = self.game.get_current_player_idx(state)
            current_agent = self.agents[current_agent_idx]

            # Have the current agent choose an action
            legal_actions = self.game.get_legal_actions(state)
            chosen_action = current_agent.choose_action(state, legal_actions)

            # Perform the chosen action
            state = self.game.apply_action(state, chosen_action)
            actions_taken += 1

        # Return the results
        return GameResult({
            "actions_taken": actions_taken,
            "winner_indices": winners,
            "final_state": state,
        })
