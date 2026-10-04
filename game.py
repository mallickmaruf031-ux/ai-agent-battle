"""
game.py - Tic-Tac-Toe game engine (Assignment Part 2 and Part 16).

This module knows nothing about AI. It only implements the rules:
  * the 3x3 board,
  * listing empty cells,
  * making / undoing a move,
  * detecting a winner, a draw, and the end of the game,
  * printing the board.

A move is a (row, col) tuple with 0 <= row, col <= 2.
"""

from typing import List, Optional, Tuple

Move = Tuple[int, int]

SIZE = 3
EMPTY = ""
PLAYER_X = "X"
PLAYER_O = "O"

# The 8 winning lines: 3 rows, 3 columns, 2 diagonals.
WIN_LINES: Tuple[Tuple[Move, Move, Move], ...] = (
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0)),
)


def opponent(player: str) -> str:
    """Return the other player's mark."""
    return PLAYER_O if player == PLAYER_X else PLAYER_X


def move_to_index(move: Move) -> int:
    """(row, col) -> cell number 0..8 (row-major). Used in logs and CSV files."""
    return move[0] * SIZE + move[1]


def index_to_move(index: int) -> Move:
    """Cell number 0..8 -> (row, col)."""
    return divmod(index, SIZE)


class TicTacToe:
    """A 3x3 Tic-Tac-Toe board with the rules of the game.

    Attributes
    ----------
    board : list[list[str]]
        3x3 grid. Each cell is 'X', 'O' or '' (empty).
    move_count : int
        Number of marks currently on the board.
    """

    def __init__(self) -> None:
        self.board: List[List[str]] = [[EMPTY] * SIZE for _ in range(SIZE)]
        self.move_count: int = 0

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear the board."""
        self.board = [[EMPTY] * SIZE for _ in range(SIZE)]
        self.move_count = 0

    def copy(self) -> "TicTacToe":
        """Return an independent copy of this game."""
        clone = TicTacToe()
        clone.board = [row[:] for row in self.board]
        clone.move_count = self.move_count
        return clone

    def to_key(self) -> str:
        """Compact 9-character description of the board, e.g. 'XO..X....'."""
        return "".join(cell or "." for row in self.board for cell in row)

    @classmethod
    def from_key(cls, key: str) -> "TicTacToe":
        """Build a game from a 9-character string ('X', 'O' or '.')."""
        if len(key) != SIZE * SIZE or any(ch not in "XO." for ch in key):
            raise ValueError("key must be 9 characters from 'X', 'O', '.'")
        game = cls()
        for index, ch in enumerate(key):
            if ch != ".":
                row, col = index_to_move(index)
                game.board[row][col] = ch
                game.move_count += 1
        return game

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def get_valid_moves(self) -> List[Move]:
        """All empty cells, in row-major order."""
        return [
            (r, c)
            for r in range(SIZE)
            for c in range(SIZE)
            if self.board[r][c] == EMPTY
        ]

    def is_valid_move(self, move: Move) -> bool:
        """True if the cell exists and is empty."""
        row, col = move
        return 0 <= row < SIZE and 0 <= col < SIZE and self.board[row][col] == EMPTY

    def current_player(self) -> str:
        """Whose turn it is, assuming X always moves first."""
        x_count = sum(row.count(PLAYER_X) for row in self.board)
        o_count = sum(row.count(PLAYER_O) for row in self.board)
        return PLAYER_X if x_count == o_count else PLAYER_O

    def check_winner(self) -> Optional[str]:
        """Return 'X' or 'O' if that player has three in a line, else None."""
        board = self.board
        for (r1, c1), (r2, c2), (r3, c3) in WIN_LINES:
            mark = board[r1][c1]
            if mark != EMPTY and mark == board[r2][c2] == board[r3][c3]:
                return mark
        return None

    def is_draw(self) -> bool:
        """True if the board is full and nobody has won."""
        return self.move_count == SIZE * SIZE and self.check_winner() is None

    def is_terminal(self) -> bool:
        """True if the game is over (somebody won, or the board is full)."""
        return self.check_winner() is not None or self.move_count == SIZE * SIZE

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #
    def make_move(self, move: Move, player: str) -> None:
        """Place `player`'s mark on `move`. Raises ValueError if illegal."""
        if player not in (PLAYER_X, PLAYER_O):
            raise ValueError(f"unknown player {player!r}")
        if not self.is_valid_move(move):
            raise ValueError(f"illegal move {move}")
        self.board[move[0]][move[1]] = player
        self.move_count += 1

    def undo_move(self, move: Move) -> None:
        """Remove the mark on `move` (used by the search to backtrack)."""
        row, col = move
        if self.board[row][col] == EMPTY:
            raise ValueError(f"nothing to undo at {move}")
        self.board[row][col] = EMPTY
        self.move_count -= 1

    # ------------------------------------------------------------------ #
    # display
    # ------------------------------------------------------------------ #
    def display(self) -> str:
        """Return a printable picture of the board."""
        rows = []
        for row in self.board:
            rows.append(" " + " | ".join(cell or " " for cell in row))
        separator = "\n---+---+---\n"
        return separator.join(rows)

    def print_board(self) -> None:
        print(self.display())

    def __str__(self) -> str:
        return self.display()
