"""
main.py - Command-line entry point for the AI Agent Battle (Tic-Tac-Toe).

Examples
--------
    python main.py all                      # run every experiment, save CSVs, print analysis
    python main.py battle                   # NEXUS vs TITAN, 10 games -> results/results.csv
    python main.py battle --games 20 --nexus-depth 4 --titan-depth 2 --seed 7
    python main.py depth                    # depth experiment -> results/depth_experiment.csv
    python main.py depth --depths 1 2 3 4 --games 10
    python main.py sweep                    # NEXUS vs TITAN at equal depths 1..5 -> results/equal_depth_battles.csv
    python main.py decisions                # NEXUS vs TITAN decision comparison
    python main.py demo                     # watch one game move by move
    python main.py analyze                  # re-compute statistics from saved CSV files

Everything is seeded (default seed 2026), so results (outcomes and node counts)
are reproducible. Timings depend on the machine.
"""

import argparse
import os
import sys

from agents import create_nexus, create_titan
from experiment import (Experiment, battle_table, compare_agent_decisions,
                        decision_table, depth_table, equal_depth_table,
                        run_depth_experiment, run_equal_depth_battles,
                        save_rows, summarize_results)
from heuristic import HEURISTIC_DESCRIPTIONS

DEFAULT_SEED = 2026


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Agent Battle: Minimax + Alpha-Beta Tic-Tac-Toe agents.")
    parser.add_argument(
        "command", nargs="?", default="all",
        choices=["all", "battle", "depth", "sweep", "decisions", "demo", "analyze"],
        help="what to run (default: all)")
    parser.add_argument("--games", type=int, default=None,
                        help="games to play (battle: default 10, depth: default 20 per depth)")
    parser.add_argument("--nexus-depth", type=int, default=3,
                        help="search depth of NEXUS (default 3)")
    parser.add_argument("--titan-depth", type=int, default=3,
                        help="search depth of TITAN (default 3)")
    parser.add_argument("--depths", type=int, nargs="+",
                        default=list(range(1, 10)),
                        help="depths for the depth experiment (default 1..9)")
    parser.add_argument("--baseline-depth", type=int, default=2,
                        help="depth of the fixed baseline opponent in the depth experiment (default 2)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed for tie-breaking (default {DEFAULT_SEED})")
    parser.add_argument("--results-dir", default="results",
                        help="folder for the CSV files (default: results)")
    parser.add_argument("--verbose", action="store_true",
                        help="print every move of every game")
    return parser


def make_agents(args):
    return (create_nexus(args.nexus_depth, seed=args.seed),
            create_titan(args.titan_depth, seed=args.seed + 1))


def paths(args):
    folder = args.results_dir
    return {
        "battle": os.path.join(folder, "results.csv"),
        "depth": os.path.join(folder, "depth_experiment.csv"),
        "decisions": os.path.join(folder, "agent_decisions.csv"),
        "sweep": os.path.join(folder, "equal_depth_battles.csv"),
        "summary": os.path.join(folder, "summary.txt"),
    }


def print_agents(nexus, titan) -> None:
    print("Agents")
    for agent, key in ((nexus, "H1"), (titan, "H2")):
        print(f"  {agent.name:<6} depth {agent.depth}  Minimax + Alpha-Beta  "
              f"{key}: {HEURISTIC_DESCRIPTIONS[key]}")
    print()


def cmd_battle(args) -> None:
    nexus, titan = make_agents(args)
    games = args.games or 10
    print_agents(nexus, titan)
    print(f"Playing {games} games (starting player alternates) ...")
    experiment = Experiment(nexus, titan)
    results = experiment.run_multiple_games(games, verbose=args.verbose)
    experiment.save_results(results, paths(args)["battle"])
    print(f"Saved {len(results)} games to {paths(args)['battle']}\n")


def cmd_depth(args) -> None:
    games = args.games or 20
    print(f"Depth experiment: depths {args.depths}, {games} games each, "
          f"baseline depth {args.baseline_depth}")
    rows = run_depth_experiment(args.depths, games, args.baseline_depth,
                                seed=args.seed, progress=print)
    save_rows(rows, paths(args)["depth"])
    print(f"Saved depth experiment to {paths(args)['depth']}\n")


def cmd_sweep(args) -> None:
    games = args.games or 20
    sweep_depths = [d for d in args.depths if d <= 5] or [1, 2, 3, 4]
    print(f"Equal-depth NEXUS vs TITAN battles: depths {sweep_depths}, "
          f"{games} games each")
    rows = run_equal_depth_battles(sweep_depths, games, seed=args.seed,
                                   progress=print)
    save_rows(rows, paths(args)["sweep"])
    print(f"Saved equal-depth battles to {paths(args)['sweep']}\n")


def cmd_decisions(args) -> None:
    nexus, titan = make_agents(args)
    print("Comparing NEXUS and TITAN on all reachable positions ...")
    rows = compare_agent_decisions(nexus, titan)
    save_rows(rows, paths(args)["decisions"])
    print(f"Saved decision comparison to {paths(args)['decisions']}\n")


def cmd_demo(args) -> None:
    nexus, titan = make_agents(args)
    print_agents(nexus, titan)
    Experiment(nexus, titan).run_game(nexus, titan, 1, verbose=True)


def cmd_analyze(args) -> None:
    """Compute every statistic from the CSV files on disk and write summary.txt."""
    p = paths(args)
    sections = []
    if os.path.exists(p["battle"]):
        sections.append("=== NEXUS vs TITAN battle (from results.csv) ===\n"
                        + summarize_results(p["battle"]) + "\n\n"
                        + battle_table(p["battle"]))
    if os.path.exists(p["depth"]):
        sections.append("=== Depth experiment (from depth_experiment.csv) ===\n"
                        + depth_table(p["depth"]))
    if os.path.exists(p["sweep"]):
        sections.append("=== Equal-depth battles (from equal_depth_battles.csv) ===\n"
                        + equal_depth_table(p["sweep"]))
    if os.path.exists(p["decisions"]):
        sections.append("=== Agent decisions (from agent_decisions.csv) ===\n"
                        + decision_table(p["decisions"], "NEXUS", "TITAN"))
    if not sections:
        print("No result files found. Run `python main.py all` first.")
        return
    text = "\n\n".join(sections)
    print(text)
    with open(p["summary"], "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"\n(Summary also written to {p['summary']})")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "all":
        cmd_battle(args)
        cmd_depth(args)
        cmd_sweep(args)
        cmd_decisions(args)
        cmd_analyze(args)
    elif args.command == "sweep":
        cmd_sweep(args)
        cmd_analyze(args)
    elif args.command == "battle":
        cmd_battle(args)
        cmd_analyze(args)
    elif args.command == "depth":
        cmd_depth(args)
        cmd_analyze(args)
    elif args.command == "decisions":
        cmd_decisions(args)
        cmd_analyze(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
