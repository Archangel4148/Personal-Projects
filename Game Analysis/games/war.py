from typing import Hashable

import numpy as np

from framework.agents import Agent, RandomAgent
from framework.base import GameModule, GameState, Action
from framework.components import CardStackComponent
from framework.observers import CycleObserver, GameObserver
from framework.runner import GameRunner


class WarModule(GameModule):

    def setup_initial_state(self, config: dict | None = None) -> GameState:
        """Set up the deck and deal to each player"""
        if config is None:
            raise ValueError("No config provided; require arguments 'num_players' and 'burn_count'.")

        num_players = config.get("num_players", 2)
        self._burn_count = config.get("burn_count", 3)

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
            "rounds_played": 0,
            "war_count": 0,
            "cycle_found": False,
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

        return new_state

    def _resolve_battle(self, state: GameState):
        battlers = [p for p in state["eligible_battlers"] if state[f"player_{p}_stake"]]
        if not battlers:
            # Give everything to whoever has cards left in their hand
            survivors = [i for i in range(state["num_players"]) if state[f"player_{i}_hand"]]
            winner = survivors[0] if survivors else 0
            for i in range(state["num_players"]):
                while state[f"player_{i}_stake"]:
                    CardStackComponent.transfer_card(state[f"player_{i}_stake"], state[f"player_{winner}_hand"])

            state["eligible_battlers"] = [i for i in range(state["num_players"]) if state[f"player_{i}_hand"]]
            state["waiting_to_flip"] = state["eligible_battlers"].copy()
            state["rounds_played"] += 1
            return

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
            # Collect everyone's stake and give it all to the winner
            cards_left = True
            while cards_left:
                cards_left = False
                for i in range(state["num_players"]):
                    if state[f"player_{i}_stake"]:
                        CardStackComponent.transfer_card(state[f"player_{i}_stake"], state[f"player_{winner}_hand"])
                        cards_left = True

            # Reset the round data for the next turn
            state["eligible_battlers"] = [i for i in range(state["num_players"]) if state[f"player_{i}_hand"]]
            state["waiting_to_flip"] = state["eligible_battlers"].copy()

        # Tied for Winner
        else:
            # Every player in the war burns cards face-down
            still_alive = set(highest_rollers)
            for _ in range(self._burn_count):
                for p in range(state["num_players"]):
                    if p in still_alive:
                        # If they can't play a burn card, they forfeit mid-war
                        if not self._play_required_card(state, p):
                            still_alive.discard(p)

            state["eligible_battlers"] = list(still_alive)

            # If 0 or 1 players survived the burn phase, resolve it immediately
            if len(still_alive) <= 1:
                # Find a winner to give stakes to
                survivors = list(still_alive) if still_alive else [i for i in range(state["num_players"]) if
                                                                   state[f"player_{i}_hand"]]
                winner = survivors[0] if survivors else 0

                for i in range(state["num_players"]):
                    while state[f"player_{i}_stake"]:
                        CardStackComponent.transfer_card(state[f"player_{i}_stake"], state[f"player_{winner}_hand"])

                state["eligible_battlers"] = [i for i in range(state["num_players"]) if state[f"player_{i}_hand"]]
                state["waiting_to_flip"] = state["eligible_battlers"].copy()
            else:
                # The surviving players must flip their next card to resolve the tie.
                state["waiting_to_flip"] = list(still_alive).copy()

            state["war_count"] += 1

        state["rounds_played"] += 1

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

    def state_key(self, state: GameState) -> Hashable:
        result = []

        for i in range(state["num_players"]):
            result.append(tuple(state[f"player_{i}_hand"]))
            result.append(tuple(state[f"player_{i}_stake"]))

        result.append(tuple(state["eligible_battlers"]))
        result.append(tuple(state["waiting_to_flip"]))

        return tuple(result)

    def is_game_over(self, state: GameState) -> tuple[bool, list[int]]:
        """The game is over if there is only one player remaining with any cards (hand or stake)"""
        if state["cycle_found"]:
            # If the game has entered a cycle, there is no winner
            return True, []
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
    # Run games of War for each player count, and show stats
    module = WarModule()
    for p in range(3,4):
        num_players = p
        war_burn_count = 3
        players: list[Agent] = [RandomAgent() for _ in range(num_players)]
        observers: list[GameObserver] = [CycleObserver(module.state_key)]

        runner = GameRunner(game_module=module, agents=players, observers=observers)

        # Run a bunch of games, and track statistics
        iterations = 1000
        cycles, cycle_len, rounds, wars = 0, 0, 0, 0
        for _ in range(iterations):
            results = runner.run_game(config={"num_players": num_players, "burn_count": war_burn_count})
            if results["cycle_found"]:
                cycles += 1
                cycle_len += results["cycle_length"]
            else:
                rounds += results["final_state"]["rounds_played"]
                wars += results["final_state"]["war_count"]

        # Display statistics
        print("\n\n=== PLAYER COUNT:", p, "===")
        print("Total iterations:", iterations)
        print(f"Average game length: {rounds / (iterations - cycles):.2f}")
        print(f"Average number of wars: {wars / (iterations - cycles):.2f}")
        print(f"Infinite cycle games: {cycles} ({cycles / iterations * 100:.2f}%)")
        if cycles > 0:
            print(f"Average cycle length: {cycle_len / cycles:.2f}")
