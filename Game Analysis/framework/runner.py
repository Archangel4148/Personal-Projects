from typing import Any, NewType

from framework.agents import Agent
from framework.base import GameModule
from framework.observers import GameObserver, GameSummaryObserver, ObserverDirective

GameResult = NewType("GameResult", dict[str, Any])


class GameRunner:
    def __init__(self, game_module: GameModule, agents: list[Agent], observers: list[GameObserver] | None = None):
        """Args:
            game_module: The game module containing a rules engine for the chosen game
            agents: A list of players
        """
        self.game = game_module
        self.agents = agents
        self.observers = [GameSummaryObserver(), *(observers or [])]

    def run_game(self, config: dict[str, Any] | None = None, action_limit: int | None = None) -> GameResult:
        """Execute a single complete game, returning a summary of the results"""
        if config:
            state = self.game.setup_initial_state(config)
        else:
            state = self.game.setup_initial_state({})
        actions_taken = 0
        action_limit_reached = False

        # Notify observers of game start
        for observer in self.observers:
            observer.on_game_start(initial_state=state)

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

            # Notify observers before each action
            for observer in self.observers:
                observer.before_action(state=state, current_player=current_agent_idx, legal_actions=legal_actions)

            # Perform the chosen action
            new_state = self.game.apply_action(state, chosen_action)
            actions_taken += 1

            # Notify observers after each action, and handle any directives
            should_stop = False
            for observer in self.observers:
                signal = observer.after_action(previous_state=state, action=chosen_action, action_number=actions_taken, new_state=new_state)
                if signal == ObserverDirective.TERMINATE:
                    should_stop = True

            state = new_state

            if should_stop:
                break

            if action_limit and actions_taken >= action_limit:
                action_limit_reached = True
                break

        # Notify observers on game end
        for observer in self.observers:
            observer.on_game_end(final_state=state, winners=winners)

        # Compile the observer results into a final object
        results = {}
        for observer in self.observers:
            results.update(observer.get_results())

        return GameResult(results)
