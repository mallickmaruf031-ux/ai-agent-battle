"""
minimax.py - Minimax search with optional Alpha-Beta pruning (Parts 3, 4, 5, 7, 8).

Terms
-----
root player  : the player the AI is choosing a move for (MAX).
MAX node     : it is the root player's turn  -> take the HIGHEST child value.
MIN node     : it is the opponent's turn     -> take the LOWEST child value.
depth        : how many moves ahead to look.
                 depth 1 = "AI move"
                 depth 2 = "AI move" -> "opponent move"
                 depth 3 = "AI move" -> "opponent move" -> "AI move" ...
               When the depth budget runs out and the game is not over, the
               position is scored by the heuristic.

Scoring of finished games
-------------------------
  root player wins  : +100  (minus the ply at which it happens)
  root player loses : -100  (plus  the ply at which it happens)
  draw              :    0
The small ply adjustment makes the AI prefer quicker wins and slower losses.
It never changes whether a move wins, draws or loses - only which winning move
is preferred. Win scores stay within [91, 99]; heuristic scores stay within
+/-90, so an estimate can never outrank a real result.

Alpha-Beta
----------
alpha = best value MAX is already guaranteed on the path to the root.
beta  = best value MIN is already guaranteed on the path to the root.
As soon as alpha >= beta the remaining children of the node cannot change the
decision at the root, so they are skipped ("pruned"). The same function runs
plain Minimax when use_pruning=False, which lets us compare the two fairly.

Statistics
----------
nodes_evaluated : number of positions the search visited (each call to
                  minimax() is one node; the root position is not counted).
nodes_pruned    : number of child moves that were skipped because of a cut-off.
                  (Each skipped child stands for a whole subtree.)
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from game import Move, TicTacToe, opponent
from heuristic import WIN_SCORE, Heuristic

INF = 10 ** 9


@dataclass
class SearchStats:
    """Counters collected while searching."""

    nodes_evaluated: int = 0
    nodes_pruned: int = 0

    def add(self, other: "SearchStats") -> None:
        self.nodes_evaluated += other.nodes_evaluated
        self.nodes_pruned += other.nodes_pruned

    def reset(self) -> None:
        self.nodes_evaluated = 0
        self.nodes_pruned = 0


@dataclass
class SearchResult:
    """What a root search returns."""

    best_move: Move
    best_moves: List[Move]          # every move tied for the best score
    score: int                      # value of the best move(s)
    move_scores: Dict[Move, int]    # see note in search_best_move()
    stats: SearchStats = field(default_factory=SearchStats)


def terminal_score(winner: Optional[str], root_player: str, ply: int) -> int:
    """Score of a finished game from the root player's point of view."""
    if winner is None:
        return 0
    if winner == root_player:
        return WIN_SCORE - ply
    return -WIN_SCORE + ply


def minimax(
    game: TicTacToe,
    depth: int,
    alpha: int,
    beta: int,
    maximizing: bool,
    root_player: str,
    heuristic: Heuristic,
    stats: SearchStats,
    ply: int = 1,
    use_pruning: bool = True,
) -> int:
    """Value of the current position for `root_player`.

    depth       remaining moves to look ahead (0 -> use the heuristic)
    maximizing  True if it is root_player's turn in this position
    ply         moves already made since the root (used for win/loss scores)
    """
    stats.nodes_evaluated += 1

    # 1. Finished game: exact result.
    winner = game.check_winner()
    if winner is not None:
        return terminal_score(winner, root_player, ply)
    moves = game.get_valid_moves()
    if not moves:
        return 0  # draw

    # 2. Out of depth: estimate with the heuristic.
    if depth == 0:
        return heuristic(game.board, root_player)

    # 3. Otherwise look one move further.
    mover = root_player if maximizing else opponent(root_player)
    if maximizing:
        best = -INF
        for index, move in enumerate(moves):
            game.make_move(move, mover)
            value = minimax(game, depth - 1, alpha, beta, False, root_player,
                            heuristic, stats, ply + 1, use_pruning)
            game.undo_move(move)
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if use_pruning and alpha >= beta:
                stats.nodes_pruned += len(moves) - index - 1
                break
        return best

    best = INF
    for index, move in enumerate(moves):
        game.make_move(move, mover)
        value = minimax(game, depth - 1, alpha, beta, True, root_player,
                        heuristic, stats, ply + 1, use_pruning)
        game.undo_move(move)
        if value < best:
            best = value
        if best < beta:
            beta = best
        if use_pruning and alpha >= beta:
            stats.nodes_pruned += len(moves) - index - 1
            break
    return best


def search_best_move(
    game: TicTacToe,
    player: str,
    depth: int,
    heuristic: Heuristic,
    use_pruning: bool = True,
    rng: Optional[random.Random] = None,
) -> SearchResult:
    """Choose a move for `player` by searching `depth` moves ahead.

    For every legal move: apply it temporarily, run Minimax (+ Alpha-Beta) on
    the resulting position, undo the move, and keep the best value. If several
    moves share exactly the best value, one of them is picked at random
    (using `rng`) so that different games can follow different lines without
    weakening the agent.

    Note on `move_scores`: with pruning on, the root searches each move inside
    the window (best_so_far - 1, +inf). That keeps every move that ties the
    best score EXACT (scores are integers), while moves that are clearly worse
    only get an upper bound. Pruning therefore never changes best_moves or
    score - tests verify this against plain Minimax.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    moves = game.get_valid_moves()
    if not moves or game.check_winner() is not None:
        raise ValueError("no move can be chosen: the game is already over")

    stats = SearchStats()
    move_scores: Dict[Move, int] = {}
    best_score = -INF
    best_moves: List[Move] = []

    for move in moves:
        game.make_move(move, player)
        alpha = best_score - 1 if use_pruning else -INF
        value = minimax(game, depth - 1, alpha, INF, False, player,
                        heuristic, stats, 1, use_pruning)
        game.undo_move(move)
        move_scores[move] = value
        if value > best_score:
            best_score = value
            best_moves = [move]
        elif value == best_score:
            best_moves.append(move)

    chooser = rng if rng is not None else random
    return SearchResult(
        best_move=chooser.choice(best_moves),
        best_moves=best_moves,
        score=best_score,
        move_scores=move_scores,
        stats=stats,
    )
