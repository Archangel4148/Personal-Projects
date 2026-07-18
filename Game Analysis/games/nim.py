


from typing import Hashable

import numpy as np

from framework.agents import Agent, RandomAgent
from framework.base import Action, GameModule, GameState
from framework.runner import GameRunner


class NimModule(GameModule):
    def setup_initial_state(self, config: dict) -> GameState:
        """Initialize the module to the starting state"""
        return GameState({
            "groups": (1, 3, 5, 7),
            "turn_flag": 1,
        })

    def get_current_player_idx(self, state: GameState) -> int:
        """Return the integer index of the player required to choose the next action"""
        return 0 if state["turn_flag"] == 1 else 1

    def get_legal_actions(self, state: GameState) -> list[Action]:
        """Return TAKE actions for each pile and available amount"""
        actions = []
        for pile_idx, pile_size in enumerate(state["groups"]):
            for amount in range(1, pile_size + 1):
                actions.append(Action(type="TAKE", payload={"pile_idx": pile_idx, "amount": amount}))                
        return actions

    def apply_action(self, state: GameState, action: Action) -> GameState:
        """Apply the given action to the given state"""
        new_state = GameState(state.copy())
        
        if action.type == "TAKE":
            new_piles = list(state["groups"])
            new_piles[action.payload["pile_idx"]] -= action.payload["amount"]
            new_state["groups"] = tuple(new_piles)

        # Flip the turn
        new_state["turn_flag"] *= -1
        return new_state

    def is_game_over(self, state: GameState) -> tuple[bool, list[int]]:
        """Determine if the game is over, and return a tuple of winning player indices (if any)"""
        
        if all(v == 0 for v in state["groups"]):
            # The game is over, the previous player lost!
            return (True, [1 if state["turn_flag"] == 1 else 0])
        
        # No winner, continue play
        return False, []

    def vectorize_state(self, state: GameState) -> np.ndarray:
        """Dump the state into a numerical vector"""
        return np.array(list(state["groups"]) + [state["turn_flag"]])
    
    def state_key(self, state: GameState) -> Hashable:
        return (state["groups"], state["turn_flag"],)

    def render_state(self, state: GameState) -> str:
        render = "\n".join(["|" * count for count in state["groups"]])
        return render


if __name__ == "__main__":
    
    game = NimModule()
    agents: list[Agent] = [RandomAgent(), RandomAgent()]
    runner = GameRunner(game, agents)

    results = runner.run_game()

    print(results)
