import numpy as np

from framework.agents import RandomAgent
from framework.base import GameModule, GameState, Action
from framework.components import CardStackComponent
from framework.runner import GameRunner


class WarModule(GameModule):
    def setup_initial_state(self, config: dict | None = None) -> GameState:
        """Set up the deck and deal to each player"""
        num_players = config.get("num_players", 2)

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
            "current_player": 0,
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
            new_state["eligible_battlers"].remove(p_idx)
        elif action.type == "FLIP_CARD":
            # The player flips a card from their deck to their stake pile
            CardStackComponent.transfer_card(new_state[f"player_{p_idx}_hand"], new_state[f"player_{p_idx}_stake"])

        # If there are no players waiting to flip, a battle begins
        if len(new_state["waiting_to_flip"]) == 0:
            self._resolve_battle(new_state)

        return new_state

    def _resolve_battle(self, state: GameState):
        battlers = state["eligible_battlers"]

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
            return

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

            # Every player in the war burns one card face-down
            for p in highest_rollers:
                CardStackComponent.transfer_card(state[f"player_{p}_hand"], state[f"player_{p}_stake"])

            # Set the state so the high-rollers will resolve the war on the next turn
            state["waiting_to_flip"] = highest_rollers.copy()

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
            vector.append(len(state[f"player_{i}_hand"]))  # Since there's no strategy, specific hand doesn't matter
        return np.array(vector)


if __name__ == '__main__':
    # Build and initialize a game module
    module = WarModule()
    players = [RandomAgent(), RandomAgent()]

    runner = GameRunner(module, players)

    results = runner.run_game(action_limit=1000)
    print("Results :", results)

