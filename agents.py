"""
agents.py - The AI agents (Parts 8, 9 and 16).

An Agent bundles three configurable things:
    * a name,
    * a search depth,
    * a heuristic (board evaluation function).
It always searches with Minimax; Alpha-Beta pruning can be switched on/off.

Two agents are provided:
    NEXUS : Minimax + Alpha-Beta + heuristic H1 (line potential)
    TITAN : Minimax + Alpha-Beta + heuristic H2 (position + threats)
Both are genuine searchers - neither plays randomly.
"""

import random
import time
from typing import List, Optional

from game import Move, TicTacToe
from heuristic import HEURISTICS, Heuristic
from minimax import SearchResult, SearchStats, search_best_move


class Agent:
    """A named Minimax / Alpha-Beta player."""

    def __init__(
        self,
        name: str,
        depth: int,
        heuristic: Heuristic,
        use_pruning: bool = True,
        seed: Optional[int] = None,
        description: str = "",
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.name = name
        self.depth = depth
        self.heuristic = heuristic
        self.use_pruning = use_pruning
        self.description = description
        self.rng = random.Random(seed)   # used only to break exact ties
        self.stats = SearchStats()       # cumulative since reset_stats()
        self.think_time: float = 0.0     # seconds spent searching
        self.moves_made: int = 0
        self.last_result: Optional[SearchResult] = None

    # ------------------------------------------------------------------ #
    def evaluate(self, board: List[List[str]], player: str) -> int:
        """Score a position with this agent's heuristic (player's view)."""
        return self.heuristic(board, player)

    def choose_move(self, game: TicTacToe, player: str,
                    depth: Optional[int] = None) -> Move:
        """Pick a move for `player`. `depth` overrides self.depth if given."""
        search_depth = self.depth if depth is None else depth
        started = time.perf_counter()
        result = search_best_move(game, player, search_depth, self.heuristic,
                                  self.use_pruning, self.rng)
        self.think_time += time.perf_counter() - started
        self.stats.add(result.stats)
        self.moves_made += 1
        self.last_result = result
        return result.best_move

    def reset_stats(self) -> None:
        """Zero the counters (called before every game)."""
        self.stats.reset()
        self.think_time = 0.0
        self.moves_made = 0
        self.last_result = None

    def __repr__(self) -> str:
        mode = "Alpha-Beta" if self.use_pruning else "Minimax"
        return f"Agent({self.name}, depth={self.depth}, {mode})"


def create_nexus(depth: int = 3, seed: Optional[int] = None,
                 use_pruning: bool = True) -> Agent:
    """NEXUS: Minimax + Alpha-Beta with heuristic H1 (line potential)."""
    return Agent("NEXUS", depth, HEURISTICS["H1"], use_pruning, seed,
                 "Minimax + Alpha-Beta, heuristic H1 (line potential)")


def create_titan(depth: int = 3, seed: Optional[int] = None,
                 use_pruning: bool = True) -> Agent:
    """TITAN: Minimax + Alpha-Beta with heuristic H2 (position + threats)."""
    return Agent("TITAN", depth, HEURISTICS["H2"], use_pruning, seed,
                 "Minimax + Alpha-Beta, heuristic H2 (position + threats)")
