import random

import numpy as np

from framework.agents import RandomAgent
from framework.base import GameModule, GameState, Action
from framework.components import CardStackComponent
from framework.runner import GameRunner


class WarModule(GameModule):
    WAR_BURN_COUNT = 3
    def setup_initial_state(self, config: dict | None = None) -> GameState:
        """Set up the deck and deal to each player"""
        num_players = config.get("num_players", 2)

        self._seen_states = set()

        # Create and shuffle the deck
        deck = CardStackComponent.create_deck()
        CardStackComponent.shuffle(deck)

        # Deal to each player
        cards_per_player = 52 // num_players
        player_hands = {}
        for i in range(num_players):
            start_idx = i * cards_per_player
            end_idx = start_idx + cards_per_player
            player_hands[f"player_{i}_hand"] = deck[start_idx:end_idx].tolist()
            player_hands[f"player_{i}_stake"] = []

        all_players = list(range(num_players))

        # Pack everything into the game state; all players are ready and waiting to play
        return GameState({
            **player_hands,
            "num_players": num_players,
            "eligible_battlers": all_players.copy(),
            "waiting_to_flip": all_players.copy(),
        })

    def get_current_player_idx(self, state: GameState) -> int:
        return state["waiting_to_flip"][0]

    def get_legal_actions(self, state: GameState) -> list[Action]:
        """Each player can only take one action"""
        p_idx = self.get_current_player_idx(state)
        # If they ran out of cards , they have to forfeit
        if not state[f"player_{p_idx}_hand"]:
            return [Action(type="FORFEIT", payload={})]
        return [Action(type="FLIP_CARD", payload={})]

    def apply_action(self, state: GameState, action: Action) -> GameState:
        new_state = GameState({k: (v.copy() if isinstance(v, list) else v) for k, v in state.items()})
        p_idx = self.get_current_player_idx(state)

        new_state["waiting_to_flip"].pop(0)

        if action.type == "FORFEIT":
            # If a player forfeits, they cannot battle
            if p_idx in new_state["eligible_battlers"]:
                new_state["eligible_battlers"].remove(p_idx)
        elif action.type == "FLIP_CARD":
            # The player flips a card from their deck to their stake pile
            self._play_required_card(new_state, p_idx)

        # If there are no players waiting to flip, a battle begins
        if len(new_state["waiting_to_flip"]) == 0:
            self._resolve_battle(new_state)

        # Check for cycles (to detect infinite games)
        self._check_for_cycle(new_state)

        return new_state

    def _resolve_battle(self, state: GameState):
        battlers = state["eligible_battlers"]

        if not battlers:
            # Everyone failed a required play; game should already be over
            state["waiting_to_flip"] = []
            return

        # If a player ran out of cards mid-war, the other player wins!
        if len(battlers) == 1:
            winner = battlers[0]
            for i in range(state["num_players"]):
                while state[f"player_{i}_stake"]:
                    CardStackComponent.transfer_card(state[f"player_{i}_stake"], state[f"player_{winner}_hand"])

            # Reset round data so healthy players can start a fresh clean round
            state["eligible_battlers"] = [i for i in range(state["num_players"]) if state[f"player_{i}_hand"]]
            state["waiting_to_flip"] = state["eligible_battlers"].copy()
            return

        # Map each player to the value of their stake
        played_values = {p: state[f"player_{p}_stake"][-1] % 13 for p in battlers if state[f"player_{p}_stake"]}
        if not played_values:
            raise RuntimeError("Battle resolved with no played cards")

        # Find the player(s) with the highest value played
        max_val = max(played_values.values())
        highest_rollers = [p for p, val in played_values.items() if val == max_val]

        # Single Winner:
        if len(highest_rollers) == 1:
            winner = highest_rollers[0]
            # Collect everyone's stake pile and give it all to the winner
            for i in range(state["num_players"]):
                while state[f"player_{i}_stake"]:
                    CardStackComponent.transfer_card(state[f"player_{i}_stake"], state[f"player_{winner}_hand"])

            # Reset the round data for the next turn
            state["eligible_battlers"] = [i for i in range(state["num_players"]) if state[f"player_{i}_hand"]]
            state["waiting_to_flip"] = state["eligible_battlers"].copy()

        # Tied for Winner
        else:
            state["eligible_battlers"] = highest_rollers

            # Every player in the war burns WAR_BURN_COUNT cards face-down
            still_alive = []
            for p in highest_rollers:
                alive = True
                for _ in range(self.WAR_BURN_COUNT):
                    if not self._play_required_card(state, p):
                        alive = False
                if alive:
                    still_alive.append(p)

            # Set the state so any remaining players will resolve the war on the next turn
            state["eligible_battlers"] = still_alive

            if len(still_alive) <= 1:
                self._resolve_battle(state)
            else:
                state["waiting_to_flip"] = still_alive.copy()

    @staticmethod
    def _play_required_card(state: GameState, player: int) -> bool:
        hand = state[f"player_{player}_hand"]

        if not hand:
            return False

        CardStackComponent.transfer_card(
            hand,
            state[f"player_{player}_stake"]
        )
        return True

    def _check_for_cycle(self, state: GameState, verbose: bool = False):
        key = (
            tuple(state["player_0_hand"]),
            tuple(state["player_1_hand"]),
            tuple(state["player_0_stake"]),
            tuple(state["player_1_stake"]),
            tuple(state["eligible_battlers"]),
            tuple(state["waiting_to_flip"]),
        )

        if key in self._seen_states:
            if verbose:
                print(f"Cycle detected!")
                print(f"P0: {state['player_0_hand']}")
                print(f"P1: {state['player_1_hand']}")
            raise RuntimeError("Cycle Detected")

        self._seen_states.add(key)

    def is_game_over(self, state: GameState) -> tuple[bool, list[int]]:
        """The game is over if there is only one player remaining with any cards (hand or stake)"""
        active_players = [i for i in range(state["num_players"]) if
                          state[f"player_{i}_hand"] or state[f"player_{i}_stake"]]

        # If 1 player remains, they win!
        if len(active_players) == 1:
            return True, [active_players[0]]

        return False, []

    def vectorize_state(self, state: GameState) -> np.ndarray:
        """Flatten the player hand counts and stakes"""
        vector = []
        for i in range(state["num_players"]):
            vector.append(len(state[f"player_{i}_hand"]))
        return np.array(vector)


if __name__ == '__main__':
    random.seed(1)
    # Build and initialize a game module
    module = WarModule()
    players = [RandomAgent(), RandomAgent()]

    runner = GameRunner(module, players)

    iterations = 1000
    cycle_count = 0
    action_count = [0, 0]
    for _ in range(iterations):
        try:
            results = runner.run_game()
            action_count[0] += results["actions_taken"]
            action_count[1] += 1
        except RuntimeError:
            # Cycle found!
            cycle_count += 1


    print(f"{cycle_count}/{iterations} ({cycle_count / iterations * 100}%)")
    if action_count[1]:
        print(f"Average Action Count (for valid games): {action_count[0] / action_count[1]:.3f}")
