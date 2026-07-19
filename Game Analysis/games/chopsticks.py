


from copy import deepcopy
from typing import Hashable

import numpy as np

from framework.agents import Agent, RandomAgent
from framework.base import Action, GameModule, GameState
from framework.runner import GameRunner
from framework.state_analysis.transforms import StateTransform


class ChopsticksModule(GameModule):
    def setup_initial_state(self, config: dict) -> GameState:
        """Initialize the module to the starting state"""

        self.num_players = config.get("num_players", 2)
        groups_per_player = config.get("groups_per_player", 2)
        points_per_group = config.get("points_per_group", 1)
        self.knockout_score = config.get("knockout_score", 5)

        return GameState({
            "groups": [[points_per_group] * groups_per_player for _ in range(self.num_players)],
            "turn_idx": 0,
        })

    def get_current_player_idx(self, state: GameState) -> int:
        """Return the integer index of the player required to choose the next action"""
        return state["turn_idx"] % self.num_players

    def get_legal_actions(self, state: GameState) -> list[Action]:
        """Return available ATTACK and SPLIT actions"""
        player_idx = self.get_current_player_idx(state)

        player_groups = state["groups"][player_idx]
        opponent_indices = [idx for idx in range(self.num_players) if idx != player_idx]

        actions = []
        # Add attack actions for each hand on each opponent hand
        for opponent_idx in opponent_indices:
            opponent_hands = state["groups"][opponent_idx]
            for target_idx, target_value in enumerate(opponent_hands): 
                # Cannot target dead hands
                if target_value == 0:
                    continue
                
                for source_idx, source_value in enumerate(player_groups):
                    # Dead hands cannot attack
                    if source_value == 0:
                        continue
                    actions.append(Action(type="ATTACK", payload={"source": (player_idx, source_idx), "target": (opponent_idx, target_idx)}))

        # Add all possible split actions (if players have >2 hands, they pick two to split between)
        hands = player_groups
        for i in range(len(hands)):
            for j in range(i + 1, len(hands)):
                total = hands[i] + hands[j]
                original = tuple(sorted((hands[i], hands[j])))
                seen = set()

                for a in range(total + 1):
                    b = total - a

                    candidate = (a, b)
                    canonical = tuple(sorted(candidate))

                    # New split must be different than original hand
                    if canonical == original:
                        continue

                    # Don't include duplicates
                    if candidate in seen:
                        continue

                    # "Suicide" splits are not allowed
                    if max(candidate) >= 5:
                        continue

                    seen.add(candidate)

                    actions.append(Action(type="SPLIT", payload={"player": player_idx, "hands": (i, j), "values": candidate}))
        
        return actions

    def apply_action(self, state: GameState, action: Action) -> GameState:
        """Apply the given action to the given state"""
        new_state = GameState(deepcopy(state))
        
        if action.type == "ATTACK":
            source_idx, group_s_idx = action.payload["source"]
            target_idx, group_t_idx = action.payload["target"]

            new_value = (new_state["groups"][target_idx][group_t_idx] + new_state["groups"][source_idx][group_s_idx])

            # Eliminate any hands that reach the knockout score
            new_state["groups"][target_idx][group_t_idx] = (0 if new_value >= self.knockout_score else new_value)
        
        elif action.type == "SPLIT":
            player_idx = action.payload["player"]
            hands = action.payload["hands"]
            values = action.payload["values"]

            for hand, value in zip(hands, values):
                new_state["groups"][player_idx][hand] = value

        # Increment the turn
        new_state["turn_idx"] += 1
        return new_state

    def is_game_over(self, state: GameState) -> tuple[bool, list[int]]:
        """Determine if the game is over, and return a tuple of winning player indices (if any)"""
        
        living_players = []
        for player_idx, hands in enumerate(state["groups"]):
            # If a player has any points on any hand, they're alive
            if any(v > 0 for v in hands):
                living_players.append(player_idx)
        
        if len(living_players) == 1:
            # Only one player remains. They win!
            return True, [living_players[0]]
        
        # All players eliminated is a tie (this should never happen!)
        if len(living_players) == 0:
            return True, []

        # No winner, continue play
        return False, []

    def vectorize_state(self, state: GameState) -> np.ndarray:
        """Dump the state into a numerical vector"""
        values = []
        for hands in state["groups"]:
            values.extend(hands)
        values.append(state["turn_idx"])
        return np.array(values)
    
    def state_key(self, state: GameState) -> Hashable:
        return (tuple(tuple(hands) for hands in state["groups"]), self.get_current_player_idx,)

    def render_state(self, state: GameState) -> str:
        lines = []
        for player_idx, hands in enumerate(state["groups"]):
            marker = "*" if player_idx == state["turn_idx"] else " "
            hand_values = " ".join(str(value) for value in hands)

            lines.append(
                f"{marker}P{player_idx}: {hand_values}"
            )
        return "\n".join(lines)


class SortHandsTransform(StateTransform):
    def transform(self, state, game):
        new_state = deepcopy(state)

        # Sort each player's hands
        new_state["groups"] = [
            sorted(player)
            for player in new_state["groups"]
        ]
        return new_state


if __name__ == "__main__":
    
    game = ChopsticksModule()

    state = game.setup_initial_state(config={
        "num_players": 2,
        # "groups_per_player": 3,
        "points_per_group": 1
    })
    # print(state, "\n")
    # actions = game.get_legal_actions(state)
    # print(actions, "\n")

    agents: list[Agent] = [RandomAgent(), RandomAgent()]
    runner = GameRunner(game, agents)
    results = runner.run_game()
    print(results)
