"""
experiment.py - Runs the experiments and collects statistics
(Assignment Parts 10, 11, 12, 13, 15 and 16).

Contents
--------
Experiment               : plays AI-vs-AI games (run_game, run_multiple_games,
                           save_results). Used for the 10-game NEXUS vs TITAN battle.
load_results()           : reads results.csv back from disk.
summarize_results()      : wins / draws / averages computed FROM THE SAVED FILE.
run_depth_experiment()   : Experiment 1 - same game, algorithm and heuristic,
                           only the search depth changes.
compare_agent_decisions(): over every reachable position, do two agents choose
                           differently?
Nothing in here is hard-coded: every number comes from games actually played.
"""

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence

from agents import Agent, create_nexus, create_titan
from game import PLAYER_O, PLAYER_X, TicTacToe, move_to_index, opponent
from heuristic import HEURISTICS
from minimax import search_best_move

DRAW = "DRAW"


# ====================================================================== #
# Single game / many games
# ====================================================================== #
@dataclass
class GameResult:
    """Everything recorded about one game (Part 12)."""

    game: int
    first: str                    # name of the agent that moved first (plays X)
    winner: str                   # agent name or "DRAW"
    moves: int
    nodes: Dict[str, int] = field(default_factory=dict)
    pruned: Dict[str, int] = field(default_factory=dict)
    think_time: Dict[str, float] = field(default_factory=dict)
    agent_moves: Dict[str, int] = field(default_factory=dict)
    time_sec: float = 0.0
    move_sequence: str = ""

    def to_row(self, names: Sequence[str]) -> Dict[str, object]:
        row: Dict[str, object] = {
            "game": self.game,
            "first": self.first,
            "winner": self.winner,
            "moves": self.moves,
        }
        for name in names:
            row[f"{name.lower()}_nodes"] = self.nodes[name]
        for name in names:
            row[f"{name.lower()}_pruned"] = self.pruned[name]
        row["time_sec"] = round(self.time_sec, 6)
        for name in names:
            row[f"{name.lower()}_time_sec"] = round(self.think_time[name], 6)
        row["move_sequence"] = self.move_sequence
        return row


class Experiment:
    """Plays games between two agents and stores the statistics."""

    def __init__(self, agent_a: Agent, agent_b: Agent) -> None:
        if agent_a.name == agent_b.name:
            raise ValueError("the two agents need different names")
        self.agent_a = agent_a
        self.agent_b = agent_b

    @property
    def names(self) -> List[str]:
        return [self.agent_a.name, self.agent_b.name]

    def run_game(self, first: Agent, second: Agent, game_number: int,
                 verbose: bool = False) -> GameResult:
        """Play one game. `first` plays X and moves first."""
        game = TicTacToe()
        players = {PLAYER_X: first, PLAYER_O: second}
        first.reset_stats()
        second.reset_stats()
        sequence: List[str] = []
        current = PLAYER_X

        if verbose:
            print(f"\n=== Game {game_number}: {first.name} (X) vs "
                  f"{second.name} (O) ===")
        started = time.perf_counter()
        while not game.is_terminal():
            agent = players[current]
            move = agent.choose_move(game, current)
            game.make_move(move, current)
            sequence.append(f"{current}{move_to_index(move)}")
            if verbose:
                result = agent.last_result
                print(f"\n{agent.name} ({current}) plays cell "
                      f"{move_to_index(move)}  [score {result.score}, "
                      f"nodes {result.stats.nodes_evaluated}, "
                      f"pruned {result.stats.nodes_pruned}]")
                print(game)
            current = opponent(current)
        elapsed = time.perf_counter() - started

        winner_mark = game.check_winner()
        winner = DRAW if winner_mark is None else players[winner_mark].name
        if verbose:
            print(f"\nResult: {winner}")

        result = GameResult(
            game=game_number,
            first=first.name,
            winner=winner,
            moves=game.move_count,
            time_sec=elapsed,
            move_sequence=" ".join(sequence),
        )
        for agent in (first, second):
            result.nodes[agent.name] = agent.stats.nodes_evaluated
            result.pruned[agent.name] = agent.stats.nodes_pruned
            result.think_time[agent.name] = agent.think_time
            result.agent_moves[agent.name] = agent.moves_made
        return result

    def run_multiple_games(self, n_games: int = 10,
                           verbose: bool = False) -> List[GameResult]:
        """Play n games, alternating the starting player.

        Game 1 -> agent A starts, game 2 -> agent B starts, game 3 -> A, ...
        """
        results = []
        for number in range(1, n_games + 1):
            if number % 2 == 1:
                first, second = self.agent_a, self.agent_b
            else:
                first, second = self.agent_b, self.agent_a
            results.append(self.run_game(first, second, number, verbose))
        return results

    def save_results(self, results: List[GameResult], path: str) -> None:
        """Write the per-game statistics to a CSV file."""
        save_rows([r.to_row(self.names) for r in results], path)


# ====================================================================== #
# File handling (Part 15)
# ====================================================================== #
def save_rows(rows: List[Dict[str, object]], path: str) -> None:
    """Write a list of dictionaries (same keys) to a CSV file."""
    if not rows:
        raise ValueError("nothing to save")
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: str) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dictionaries (all values are strings)."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _agent_names_from_header(row: Dict[str, str]) -> List[str]:
    return [key[:-len("_nodes")].upper() for key in row if key.endswith("_nodes")]


def summarize_results(path: str) -> str:
    """Compute and format the battle statistics from the saved CSV file."""
    rows = load_rows(path)
    names = _agent_names_from_header(rows[0])
    total = len(rows)

    wins = {name: sum(1 for r in rows if r["winner"] == name) for name in names}
    draws = sum(1 for r in rows if r["winner"] == DRAW)
    first_wins = sum(1 for r in rows
                     if r["winner"] != DRAW and r["winner"] == r["first"])
    second_wins = total - draws - first_wins

    def mean(values):
        values = list(values)
        return sum(values) / len(values)

    distinct = len({r["move_sequence"] for r in rows})
    lines = [f"Games played : {total}",
             "Wins         : " + ", ".join(f"{n} {wins[n]}" for n in names),
             f"Draws        : {draws}",
             f"Distinct games (different move sequences) : {distinct}",
             f"Starting player won {first_wins}, second player won "
             f"{second_wins}, draws {draws}",
             f"Average moves per game : {mean(int(r['moves']) for r in rows):.2f}",
             f"Average game time      : "
             f"{mean(float(r['time_sec']) for r in rows) * 1000:.2f} ms",
             ""]
    for name in names:
        key = name.lower()
        lines.append(
            f"{name:<6} avg nodes/game {mean(int(r[key + '_nodes']) for r in rows):8.1f}"
            f" | avg pruned/game {mean(int(r[key + '_pruned']) for r in rows):8.1f}"
            f" | avg search time/game "
            f"{mean(float(r[key + '_time_sec']) for r in rows) * 1000:7.2f} ms")
    # Wins when starting vs. when second, per agent.
    lines.append("")
    for name in names:
        started = [r for r in rows if r["first"] == name]
        others = [r for r in rows if r["first"] != name]
        lines.append(
            f"{name:<6} as first player: "
            f"{sum(1 for r in started if r['winner'] == name)} wins / "
            f"{sum(1 for r in started if r['winner'] == DRAW)} draws / "
            f"{sum(1 for r in started if r['winner'] not in (name, DRAW))} losses"
            f"   |  as second player: "
            f"{sum(1 for r in others if r['winner'] == name)} wins / "
            f"{sum(1 for r in others if r['winner'] == DRAW)} draws / "
            f"{sum(1 for r in others if r['winner'] not in (name, DRAW))} losses")
    return "\n".join(lines)


def battle_table(path: str) -> str:
    """Markdown table of the per-game records stored in results.csv."""
    rows = load_rows(path)
    names = _agent_names_from_header(rows[0])
    head = ["Game", "First", "Winner", "Moves"]
    head += [f"{n} nodes" for n in names] + [f"{n} pruned" for n in names]
    head += ["Time (ms)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["game"], r["first"], r["winner"], r["moves"]]
        cells += [r[n.lower() + "_nodes"] for n in names]
        cells += [r[n.lower() + "_pruned"] for n in names]
        cells += [f"{float(r['time_sec']) * 1000:.2f}"]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ====================================================================== #
# Position enumeration (used by the decision analyses)
# ====================================================================== #
def enumerate_positions() -> List[str]:
    """Every NON-finished position reachable in a real game (X moves first).

    Returned as 9-character keys (see TicTacToe.to_key). There are 4,520.
    """
    start = TicTacToe()
    seen = {start.to_key()}
    stack = [start.to_key()]
    positions: List[str] = []
    while stack:
        key = stack.pop()
        game = TicTacToe.from_key(key)
        if game.is_terminal():
            continue
        positions.append(key)
        mover = game.current_player()
        for move in game.get_valid_moves():
            game.make_move(move, mover)
            child = game.to_key()
            game.undo_move(move)
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return sorted(positions)


def best_move_set(key: str, depth: int, heuristic: Callable,
                  use_pruning: bool = True) -> FrozenSet[tuple]:
    """All moves tied for best at `depth` in the position `key`."""
    game = TicTacToe.from_key(key)
    result = search_best_move(game, game.current_player(), depth, heuristic,
                              use_pruning)
    return frozenset(result.best_moves)


# ====================================================================== #
# Experiment 1 - does thinking deeper help?  (Part 10)
# ====================================================================== #
def run_depth_experiment(
    depths: Sequence[int],
    games_per_depth: int = 20,
    baseline_depth: int = 2,
    heuristic_name: str = "H1",
    seed: int = 2026,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, object]]:
    """Only the search depth changes; game, algorithm and heuristic are fixed.

    For each depth d:
      * "Result": an H1 Alpha-Beta agent searching to depth d plays
        `games_per_depth` games (alternating the starting player) against a
        fixed BASELINE agent: the same algorithm and heuristic at `baseline_depth`.
      * "Cost":   nodes / pruned / time per move during those games, and the
        cost of the first move on an empty board with Alpha-Beta versus plain
        Minimax (same position, same depth).
      * "Decisions": over all 4,520 reachable positions, how often does the
        set of best moves at depth d differ from the one at the previous depth?
    """
    log = progress or (lambda message: None)
    heuristic = HEURISTICS[heuristic_name]
    positions = enumerate_positions()
    rows: List[Dict[str, object]] = []
    previous_sets: Optional[Dict[str, FrozenSet[tuple]]] = None

    for depth in depths:
        log(f"  depth {depth}: playing {games_per_depth} games ...")
        tested = Agent(f"{heuristic_name}-D{depth}", depth, heuristic,
                       seed=seed + depth)
        baseline = Agent("BASELINE", baseline_depth, heuristic,
                         seed=seed + 1000 + depth)
        experiment = Experiment(tested, baseline)
        results = experiment.run_multiple_games(games_per_depth)

        wins = sum(1 for r in results if r.winner == tested.name)
        losses = sum(1 for r in results if r.winner == baseline.name)
        draws = sum(1 for r in results if r.winner == DRAW)
        nodes = sum(r.nodes[tested.name] for r in results)
        pruned = sum(r.pruned[tested.name] for r in results)
        think = sum(r.think_time[tested.name] for r in results)
        moves = sum(r.agent_moves[tested.name] for r in results)

        # Cost of the very first move on an empty board: Alpha-Beta vs Minimax.
        def timed_root(use_pruning: bool):
            best_time = None
            for _ in range(3):
                begin = time.perf_counter()
                found = search_best_move(TicTacToe(), PLAYER_X, depth,
                                         heuristic, use_pruning)
                spent = time.perf_counter() - begin
                best_time = spent if best_time is None else min(best_time, spent)
            return found, best_time

        ab, ab_time = timed_root(True)
        mm, mm_time = timed_root(False)
        if ab.best_moves != mm.best_moves or ab.score != mm.score:
            raise AssertionError("Alpha-Beta changed the Minimax decision!")

        log(f"  depth {depth}: comparing decisions on {len(positions)} positions ...")
        current_sets = {key: best_move_set(key, depth, heuristic)
                        for key in positions}
        if previous_sets is None:
            identical = disjoint = None
        else:
            identical = 100.0 * sum(
                1 for k in positions if current_sets[k] == previous_sets[k]
            ) / len(positions)
            disjoint = 100.0 * sum(
                1 for k in positions if not (current_sets[k] & previous_sets[k])
            ) / len(positions)
        previous_sets = current_sets

        rows.append({
            "depth": depth,
            "games": games_per_depth,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "avg_nodes_per_move": round(nodes / moves, 2),
            "avg_pruned_per_move": round(pruned / moves, 2),
            "avg_ms_per_move": round(think / moves * 1000, 4),
            "total_time_sec": round(think, 6),
            "root_nodes_alphabeta": ab.stats.nodes_evaluated,
            "root_pruned_alphabeta": ab.stats.nodes_pruned,
            "root_ms_alphabeta": round(ab_time * 1000, 4),
            "root_nodes_minimax": mm.stats.nodes_evaluated,
            "root_ms_minimax": round(mm_time * 1000, 4),
            "node_reduction_pct": round(
                100.0 * (1 - ab.stats.nodes_evaluated / mm.stats.nodes_evaluated), 2),
            "root_score": ab.score,
            "root_best_cells": " ".join(str(move_to_index(m))
                                        for m in sorted(ab.best_moves)),
            "same_decisions_vs_prev_pct": "" if identical is None else round(identical, 2),
            "no_overlap_vs_prev_pct": "" if disjoint is None else round(disjoint, 2),
        })
    return rows


def depth_table(path: str) -> str:
    """Markdown tables for the depth experiment, built from the saved CSV."""
    rows = load_rows(path)
    games = rows[0]["games"]
    out = [f"Results of an H1 agent at depth d vs. a fixed baseline "
           f"({games} games per depth, alternating start), and cost per move:", "",
           "| Depth | W / D / L | Avg nodes per move | Avg pruned per move | "
           "Avg time per move (ms) |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['depth']} | {r['wins']} / {r['draws']} / {r['losses']} | "
                   f"{float(r['avg_nodes_per_move']):.1f} | "
                   f"{float(r['avg_pruned_per_move']):.1f} | "
                   f"{float(r['avg_ms_per_move']):.3f} |")
    out += ["", "First move on an empty board (same position, same depth):", "",
            "| Depth | Minimax nodes | Alpha-Beta nodes | Alpha-Beta pruned | "
            "Nodes saved | Minimax time (ms) | Alpha-Beta time (ms) | "
            "Score | Best cell(s) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['depth']} | {r['root_nodes_minimax']} | "
                   f"{r['root_nodes_alphabeta']} | {r['root_pruned_alphabeta']} | "
                   f"{r['node_reduction_pct']}% | "
                   f"{float(r['root_ms_minimax']):.3f} | "
                   f"{float(r['root_ms_alphabeta']):.3f} | "
                   f"{r['root_score']} | {r['root_best_cells']} |")
    out += ["", "Did a deeper search change the decisions? "
            "(all 4,520 reachable positions, compared with the previous depth):", "",
            "| Depth | Identical best-move set | Completely different best moves |",
            "|---|---|---|"]
    for r in rows:
        if r["same_decisions_vs_prev_pct"] == "":
            continue
        out.append(f"| {r['depth']} | {r['same_decisions_vs_prev_pct']}% | "
                   f"{r['no_overlap_vs_prev_pct']}% |")
    return "\n".join(out)


# ====================================================================== #
# Supplementary: NEXUS vs TITAN at several equal depths
# ====================================================================== #
def run_equal_depth_battles(
    depths: Sequence[int],
    games: int = 20,
    seed: int = 2026,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, object]]:
    """NEXUS (H1) vs TITAN (H2) with BOTH agents searching to the same depth.

    The required battle uses depth 3. Here the depth is varied (both agents
    always get the same depth) to see where the two heuristics actually matter:
    the shallower the search, the more the agents depend on their heuristic.
    """
    log = progress or (lambda message: None)
    rows: List[Dict[str, object]] = []
    for depth in depths:
        log(f"  both agents at depth {depth}: {games} games ...")
        nexus = create_nexus(depth, seed=seed + depth)
        titan = create_titan(depth, seed=seed + 500 + depth)
        results = Experiment(nexus, titan).run_multiple_games(games)
        n = len(results)
        rows.append({
            "depth": depth,
            "games": n,
            "nexus_wins": sum(1 for r in results if r.winner == "NEXUS"),
            "titan_wins": sum(1 for r in results if r.winner == "TITAN"),
            "draws": sum(1 for r in results if r.winner == DRAW),
            "first_player_wins": sum(1 for r in results
                                     if r.winner != DRAW and r.winner == r.first),
            "distinct_games": len({r.move_sequence for r in results}),
            "nexus_avg_nodes": round(sum(r.nodes["NEXUS"] for r in results) / n, 1),
            "titan_avg_nodes": round(sum(r.nodes["TITAN"] for r in results) / n, 1),
            "nexus_avg_ms": round(
                sum(r.think_time["NEXUS"] for r in results) / n * 1000, 3),
            "titan_avg_ms": round(
                sum(r.think_time["TITAN"] for r in results) / n * 1000, 3),
        })
    return rows


def equal_depth_table(path: str) -> str:
    """Markdown table for the equal-depth battles."""
    rows = load_rows(path)
    out = [f"NEXUS (H1) vs TITAN (H2), both at the same depth "
           f"({rows[0]['games']} games each, alternating start):", "",
           "| Depth | NEXUS wins | TITAN wins | Draws | Wins by starting player | "
           "Distinct games | NEXUS nodes/game | TITAN nodes/game | "
           "NEXUS ms/game | TITAN ms/game |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['depth']} | {r['nexus_wins']} | {r['titan_wins']} | "
                   f"{r['draws']} | {r['first_player_wins']} | {r['distinct_games']} | "
                   f"{float(r['nexus_avg_nodes']):.1f} | "
                   f"{float(r['titan_avg_nodes']):.1f} | "
                   f"{float(r['nexus_avg_ms']):.2f} | {float(r['titan_avg_ms']):.2f} |")
    return "\n".join(out)


# ====================================================================== #
# Do two agents decide differently?  (Part 13, question 5)
# ====================================================================== #
def compare_agent_decisions(agent_a: Agent, agent_b: Agent) -> List[Dict[str, object]]:
    """Compare the best-move sets of two agents on every reachable position.

    identical : both agents consider exactly the same moves best
    overlap   : the sets differ but share at least one move (a tie-break could
                make them play the same move)
    disjoint  : no move is best for both -> the agents MUST play differently
    """
    positions = enumerate_positions()
    buckets: Dict[int, Dict[str, int]] = {}
    for key in positions:
        pieces = 9 - key.count(".")
        set_a = best_move_set(key, agent_a.depth, agent_a.heuristic)
        set_b = best_move_set(key, agent_b.depth, agent_b.heuristic)
        bucket = buckets.setdefault(
            pieces, {"positions": 0, "identical": 0, "overlap": 0, "disjoint": 0})
        bucket["positions"] += 1
        if set_a == set_b:
            bucket["identical"] += 1
        elif set_a & set_b:
            bucket["overlap"] += 1
        else:
            bucket["disjoint"] += 1

    rows: List[Dict[str, object]] = []
    totals = {"positions": 0, "identical": 0, "overlap": 0, "disjoint": 0}
    for pieces in sorted(buckets):
        rows.append({"pieces_on_board": pieces, **buckets[pieces]})
        for name in totals:
            totals[name] += buckets[pieces][name]
    rows.append({"pieces_on_board": "ALL", **totals})
    return rows


def decision_table(path: str, name_a: str, name_b: str) -> str:
    """Markdown table for the agent decision comparison."""
    rows = load_rows(path)
    out = [f"{name_a} vs {name_b}: do they choose different moves "
           f"(by number of pieces already on the board)?", "",
           "| Pieces on board | Positions | Identical best moves | "
           "Different but overlapping | Completely different |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['pieces_on_board']} | {r['positions']} | {r['identical']} | "
                   f"{r['overlap']} | {r['disjoint']} |")
    return "\n".join(out)
