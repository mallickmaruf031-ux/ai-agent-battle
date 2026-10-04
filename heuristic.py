"""
heuristic.py - Board evaluation functions (Assignment Part 6).

A heuristic estimates how good a NON-FINISHED position is for a given player.
Every heuristic here has the same signature:

    heuristic(board, player) -> int

and returns a score from the point of view of `player`:
    positive  -> good for `player`
    negative  -> good for the opponent
    0         -> balanced

Both heuristics are zero-sum: heuristic(board, X) == -heuristic(board, O).

Scale
-----
Finished games are scored by minimax.py (win = +100, loss = -100, draw = 0,
adjusted slightly by depth). Heuristic scores are clamped to +/-MAX_HEURISTIC
so that an estimate can never be mistaken for (or outrank) a real win/loss.

The two heuristics used by the agents:

  H1  "Line potential"       (used by NEXUS)
      Looks at the 8 winning lines only.
        open two-in-a-row   : +/-10   (a strong threat)
        open one-in-a-row   : +/-1    (a line that is still alive)
      A line containing both X and O is dead and counts for nothing.

  H2  "Position + threats"   (used by TITAN)
      Combines a square-value table with threat detection.
        centre +3, corners +2, edges +1  (own squares minus opponent squares)
        open two-in-a-row   : +/-10   (a strong threat)
      Lines with a single mark are ignored; only square values matter there.
"""

from typing import Callable, Dict, List

from game import EMPTY, WIN_LINES, opponent

WIN_SCORE = 100
MAX_HEURISTIC = 90  # strictly below the smallest real win score (100 - 9 = 91)

THREAT_SCORE = 10   # open two-in-a-row
LINE_SCORE = 1      # open one-in-a-row

# Square values for H2: centre is worth most, then corners, then edges.
POSITION_WEIGHTS: List[List[int]] = [
    [2, 1, 2],
    [1, 3, 1],
    [2, 1, 2],
]


def _clamp(score: int) -> int:
    return max(-MAX_HEURISTIC, min(MAX_HEURISTIC, score))


def _winner_score(board: List[List[str]], player: str):
    """Return +/-WIN_SCORE if the game is already won, else None."""
    foe = opponent(player)
    for (r1, c1), (r2, c2), (r3, c3) in WIN_LINES:
        mark = board[r1][c1]
        if mark != EMPTY and mark == board[r2][c2] == board[r3][c3]:
            return WIN_SCORE if mark == player else -WIN_SCORE
    return None


def _line_contents(board: List[List[str]], line, player: str):
    """Return (my_marks, opponent_marks) on one winning line."""
    foe = opponent(player)
    mine = theirs = 0
    for row, col in line:
        cell = board[row][col]
        if cell == player:
            mine += 1
        elif cell == foe:
            theirs += 1
    return mine, theirs


def h1_line_potential(board: List[List[str]], player: str) -> int:
    """H1: score every winning line that is still alive for exactly one side."""
    won = _winner_score(board, player)
    if won is not None:
        return won
    score = 0
    for line in WIN_LINES:
        mine, theirs = _line_contents(board, line, player)
        if mine and theirs:
            continue  # dead line: nobody can complete it
        if theirs == 0:
            if mine == 2:
                score += THREAT_SCORE
            elif mine == 1:
                score += LINE_SCORE
        else:  # mine == 0
            if theirs == 2:
                score -= THREAT_SCORE
            elif theirs == 1:
                score -= LINE_SCORE
    return _clamp(score)


def h2_positional_threat(board: List[List[str]], player: str) -> int:
    """H2: square values (centre > corner > edge) plus open two-in-a-row threats."""
    won = _winner_score(board, player)
    if won is not None:
        return won
    foe = opponent(player)
    score = 0
    for row in range(3):
        for col in range(3):
            if board[row][col] == player:
                score += POSITION_WEIGHTS[row][col]
            elif board[row][col] == foe:
                score -= POSITION_WEIGHTS[row][col]
    for line in WIN_LINES:
        mine, theirs = _line_contents(board, line, player)
        if mine == 2 and theirs == 0:
            score += THREAT_SCORE
        elif theirs == 2 and mine == 0:
            score -= THREAT_SCORE
    return _clamp(score)


Heuristic = Callable[[List[List[str]], str], int]

HEURISTICS: Dict[str, Heuristic] = {
    "H1": h1_line_potential,
    "H2": h2_positional_threat,
}

HEURISTIC_DESCRIPTIONS: Dict[str, str] = {
    "H1": "Line potential: open two-in-a-row = +/-10, open one-in-a-row = +/-1",
    "H2": "Position + threats: centre +3, corner +2, edge +1, open two-in-a-row = +/-10",
}
