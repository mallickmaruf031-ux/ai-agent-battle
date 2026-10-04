# AI Agent Battle — Tic-Tac-Toe with Minimax + Alpha-Beta

**Assignment X_02 — AI/ML Laboratory, B.Tech 5th Semester: "Python MINIMAX Search: AI Agents Battle Arena"**

**Student:** <Maruf Mallick> &nbsp;|&nbsp; **Roll No.:** <29>

Two named AI agents, **NEXUS** and **TITAN**, play Tic-Tac-Toe against each other with no human involved.
Both search with **Minimax + Alpha-Beta pruning**, but they judge unfinished positions with **different heuristics**.
The project measures three things:

1. Does thinking deeper make an AI better?
2. How does the way an AI evaluates a position change its decisions?
3. What does it cost (nodes searched, time) to make an AI think more?

The full write-up (problem formulation, algorithm design, results and analysis) is in **[REPORT.md](REPORT.md)**.

---

## 1. Agents, algorithms, heuristics, depths

| Agent | Algorithm | Heuristic | Depth (main battle) |
|---|---|---|---|
| **NEXUS** | Minimax + Alpha-Beta | **H1 – line potential** | 3 |
| **TITAN** | Minimax + Alpha-Beta | **H2 – position + threats** | 3 |

* **Minimax** — the AI is MAX, the opponent is MIN. MAX picks the highest-valued move, MIN the lowest.
* **Alpha-Beta pruning** — skips branches that cannot change the decision. It makes the search cheaper but never changes the Minimax decision (this is verified by the tests, see §6).
* **Depth** is a parameter everywhere (`depth=1` = "AI move", `depth=2` = "AI move → opponent move", …). It is never hard-coded inside the search.
* **Finished games:** win `+100`, loss `-100`, draw `0` (a win/loss at ply *p* is scored `±(100 − p)`, so quicker wins are preferred; this never changes whether a move wins, draws or loses).
* **Heuristic (when the depth runs out and the game is not over):**
  * **H1 – line potential (NEXUS):** for each of the 8 winning lines still alive for exactly one side: open two-in-a-row = **±10** (strong threat), open one-in-a-row = **±1**. A line holding both X and O is dead and scores 0.
  * **H2 – position + threats (TITAN):** square values **centre +3, corner +2, edge +1** (own minus opponent) plus open two-in-a-row = **±10**. Single marks on a line are ignored except through their square value.
  * Both are zero-sum (`H(board, X) == -H(board, O)`) and clamped to ±90, so an estimate can never outrank a real win (≥ 91).
* **Tie-breaking:** if several moves have exactly the same best score, one is chosen at random (seeded RNG). No agent is ever made weak on purpose.

## 2. Project structure

```
ai-agent-battle/
├── main.py                  # command-line entry point
├── game.py                  # TicTacToe class: board and rules
├── minimax.py               # Minimax + Alpha-Beta search, node counters
├── heuristic.py             # heuristics H1 and H2
├── agents.py                # Agent class, NEXUS and TITAN
├── experiment.py            # Experiment class, depth experiment, CSV handling, statistics
├── test_ai_battle.py        # 24 automated tests
├── results/
│   ├── results.csv              # the 10-game NEXUS vs TITAN battle
│   ├── depth_experiment.csv     # Experiment 1 (search depth 1-9)
│   ├── equal_depth_battles.csv  # supplementary: NEXUS vs TITAN at equal depths 1-5
│   ├── agent_decisions.csv      # do NEXUS and TITAN choose differently?
│   └── summary.txt              # all statistics, recomputed from the CSV files
├── REPORT.md                # the brief report
├── README.md
├── requirements.txt
└── .gitignore
```

| File | Responsibility |
|---|---|
| `game.py` | `TicTacToe`: `display`, `get_valid_moves`, `make_move`, `undo_move`, `check_winner`, `is_draw`, `is_terminal` |
| `minimax.py` | `minimax(...)` (recursive search), `search_best_move(...)` (root: tries every move), `SearchStats` (nodes evaluated / pruned) |
| `heuristic.py` | `h1_line_potential`, `h2_positional_threat`, registry `HEURISTICS` |
| `agents.py` | `Agent` (`name`, `depth`, `heuristic`, `choose_move()`, `evaluate()`), `create_nexus()`, `create_titan()` |
| `experiment.py` | `Experiment` (`run_game()`, `run_multiple_games()`, `save_results()`), `run_depth_experiment()`, `summarize_results()` |
| `main.py` | argparse CLI |

## 3. Setup and usage

Requirements: **Python 3.8 or newer** (developed and tested with 3.13). There are **no dependencies** — only the standard library.

```bash
git clone <your-repository-url>
cd ai-agent-battle

python main.py all        # run everything and print the analysis (about 25 seconds)
```

| Command | What it does |
|---|---|
| `python main.py all` | Runs the battle, depth experiment, equal-depth battles and decision comparison; writes all CSV files; prints the statistics |
| `python main.py battle` | NEXUS vs TITAN, 10 games → `results/results.csv` |
| `python main.py depth` | Experiment 1 (depth 1–9) → `results/depth_experiment.csv` |
| `python main.py sweep` | NEXUS vs TITAN at equal depths 1–5 → `results/equal_depth_battles.csv` |
| `python main.py decisions` | Compares NEXUS and TITAN on all 4,520 reachable positions → `results/agent_decisions.csv` |
| `python main.py demo` | Plays one game and prints every move (add `--verbose` to other commands for the same) |
| `python main.py analyze` | Recomputes all statistics **from the saved CSV files** and writes `results/summary.txt` |

Useful options:

```bash
python main.py battle --games 20                       # more games
python main.py battle --nexus-depth 4 --titan-depth 2  # different depths per agent
python main.py depth --depths 1 2 3 4 --games 10       # only depths 1-4, 10 games each
python main.py all --seed 7 --results-dir my_results   # different seed / output folder
```

Everything is seeded (default seed `2026`). With the same seed, **game outcomes and node counts are reproducible**; only the measured times vary from machine to machine.

## 4. Results (from the saved CSV files)

### 4.1 The 10-game battle — NEXUS (H1, depth 3) vs TITAN (H2, depth 3)

The starting player alternates (game 1 NEXUS, game 2 TITAN, …). In `move_sequence` (in the CSV), `X4` means "X played cell 4"; cells are numbered 0–8, row by row.

| Game | First | Winner | Moves | NEXUS nodes | TITAN nodes | NEXUS pruned | TITAN pruned | Time (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | NEXUS | DRAW | 9 | 363 | 233 | 252 | 139 | 3.42 |
| 2 | TITAN | DRAW | 9 | 268 | 453 | 138 | 226 | 4.07 |
| 3 | NEXUS | DRAW | 9 | 402 | 309 | 211 | 133 | 3.95 |
| 4 | TITAN | DRAW | 9 | 276 | 386 | 146 | 249 | 3.65 |
| 5 | NEXUS | DRAW | 9 | 370 | 309 | 207 | 137 | 3.99 |
| 6 | TITAN | DRAW | 9 | 276 | 386 | 146 | 249 | 3.64 |
| 7 | NEXUS | DRAW | 9 | 421 | 314 | 216 | 132 | 4.00 |
| 8 | TITAN | DRAW | 9 | 306 | 418 | 136 | 225 | 4.08 |
| 9 | NEXUS | DRAW | 9 | 421 | 314 | 216 | 132 | 4.10 |
| 10 | TITAN | DRAW | 9 | 268 | 453 | 138 | 226 | 4.17 |

**NEXUS 0 wins, TITAN 0 wins, 10 draws.** Only 6 of the 10 games were different (random tie-breaking among equally good moves produced 6 distinct move sequences).
Averages per game: NEXUS 337.1 nodes evaluated / 180.6 pruned / 1.79 ms of search time; TITAN 357.5 nodes / 184.8 pruned / 2.08 ms; 9.0 moves and 3.91 ms per game.

### 4.2 Experiment 1 — does deeper thinking help?

Only the search depth changes (Tic-Tac-Toe, Minimax + Alpha-Beta, heuristic H1). For each depth, an H1 agent searching to that depth plays 20 games (alternating start) against a fixed baseline: the same algorithm and heuristic at depth 2. "Cost" is measured per move.

| Depth | Result (W / D / L) | Avg nodes evaluated per move | Avg nodes pruned per move | Avg time per move (ms) |
|---|---|---|---|---|
| 1 | 0 / 14 / 6 | 5.3 | 0.0 | 0.040 |
| 2 | 0 / 20 / 0 | 21.5 | 10.1 | 0.137 |
| 3 | 0 / 20 / 0 | 75.9 | 40.2 | 0.406 |
| 4 | 0 / 20 / 0 | 236.2 | 148.4 | 1.198 |
| 5 | 0 / 20 / 0 | 648.2 | 337.2 | 2.994 |
| 6 | 0 / 20 / 0 | 1220.1 | 801.2 | 4.907 |
| 7 | 0 / 20 / 0 | 2890.3 | 1340.9 | 10.383 |
| 8 | 0 / 20 / 0 | 3532.0 | 1714.0 | 10.281 |
| 9 | 0 / 20 / 0 | 4348.7 | 1634.2 | 9.579 |

The same search on the **empty board** (first move), Alpha-Beta versus plain Minimax:

| Depth | Minimax nodes | Alpha-Beta nodes | Alpha-Beta pruned | Nodes saved | Minimax time (ms) | Alpha-Beta time (ms) | Best cell(s) |
|---|---|---|---|---|---|---|---|
| 1 | 9 | 9 | 0 | 0.0% | 0.052 | 0.056 | 4 |
| 2 | 81 | 45 | 36 | 44.4% | 0.452 | 0.241 | 4 |
| 3 | 585 | 184 | 163 | 68.5% | 3.364 | 0.991 | 4 |
| 4 | 3,609 | 836 | 651 | 76.8% | 20.926 | 4.501 | 4 |
| 5 | 18,729 | 2,993 | 1,748 | 84.0% | 97.847 | 14.755 | 4 |
| 6 | 73,449 | 6,174 | 4,881 | 91.6% | 382.345 | 27.781 | 4 |
| 7 | 221,625 | 19,367 | 9,375 | 91.3% | 875.819 | 75.550 | 4 |
| 8 | 422,073 | 24,554 | 12,919 | 94.2% | 1426.833 | 77.001 | all 9 cells tie |
| 9 | 549,945 | 31,620 | 12,187 | 94.3% | 1128.884 | 70.187 | all 9 cells tie |

### 4.3 Supplementary — NEXUS vs TITAN at equal depths (20 games each)

| Depth | NEXUS wins | TITAN wins | Draws | Wins by starting player |
|---|---|---|---|---|
| 1 | 13 | 0 | 7 | 10 |
| 2 | 0 | 0 | 20 | 0 |
| 3 | 0 | 0 | 20 | 0 |
| 4 | 0 | 0 | 20 | 0 |
| 5 | 0 | 0 | 20 | 0 |

## 5. Observations and conclusion (short version — details in [REPORT.md](REPORT.md))

* **Deeper search changes decisions and costs more.** Going from depth 1 to 2 changes the set of best moves in 22% of all reachable positions; each extra level changes fewer positions, and depth 8 → 9 changes none. Cost grows steeply: on the empty board Alpha-Beta evaluates 9 nodes at depth 1 and 31,620 at depth 9.
* **Alpha-Beta saves a lot and never changes the decision.** It evaluates 44% fewer nodes at depth 2 and 94% fewer at depth 9 (up to ~18× faster at depth 8), and returns the same best moves and score as plain Minimax (checked on all 4,520 reachable positions at depths 1–4, and at full depth for every position with at least 3 marks).
* **Deeper is not automatically "better" in results.** In a game that is a theoretical draw, a depth-1 agent loses 6 of 20 games to a depth-2 baseline, but every depth from 2 to 9 simply draws — extra thinking beyond depth 2 costs more without improving the outcome.
* **The two heuristics matter most when the search is shallow.** At depth 1 NEXUS beats TITAN 13–0 (7 draws). From depth 2 on, search sees through the heuristic's blind spots and all games are drawn. At the required depth 3 the heuristics only change *which* equally-good move is played: the agents agree on the best-move set in 95.4% of positions and have no move in common in only 8 of 4,520.
* **No first-player advantage appeared** in the 10-game battle (all draws). It only showed up for weak play (depth 1: NEXUS won 10 of 10 games as starter, but 3 of 10 as second player).
* **Conclusion:** in the main experiment there is no winner — 10 draws out of 10 — and that is the correct, expected outcome for two competent players of Tic-Tac-Toe. The evidence does *not* support calling either agent "better" at depth 3. It does show that NEXUS's heuristic is more robust when the agent can look only one move ahead.

## 6. Tests

```bash
python -m unittest -v      # 24 tests, about 12 seconds
```

The tests cover the game rules (all 8 winning lines, draws, illegal moves, undo), the heuristics (zero-sum, bounded, threats valued), the search (takes an immediate win, blocks a threat, depth is a real parameter, the board is left unchanged), **Alpha-Beta ≡ Minimax on every one of the 4,520 reachable positions (depths 1–4) and at full depth for positions with at least 3 marks**, Alpha-Beta evaluating fewer nodes, perfect players always drawing, alternating starters, reproducibility with a fixed seed, and the CSV round trip.

## 7. How the code works (for the evaluation / viva)

1. **`TicTacToe`** stores a 3×3 list of lists (`'X'`, `'O'`, `''`). `make_move` / `undo_move` let the search try a move and take it back, so no board copies are needed.
2. **`search_best_move(game, player, depth, heuristic)`** is the AI's `choose_move`. For every legal move it: applies the move → calls `minimax` with `depth - 1` → undoes the move → keeps the best value. Ties are broken randomly.
3. **`minimax(...)`** is recursive. At each node it counts one *node evaluated*. If the game is over it returns the exact result (±100−ply, or 0). If `depth == 0` it returns the heuristic. Otherwise it recurses: MAX takes the largest child value, MIN the smallest.
4. **Alpha-Beta:** `alpha` is the best value MAX can already guarantee, `beta` the best value MIN can already guarantee. When `alpha >= beta`, the remaining moves at that node cannot matter, so the loop `break`s and the number of skipped moves is added to *nodes pruned*. Running with `use_pruning=False` gives plain Minimax, which is how the two are compared fairly.
5. **Detail worth knowing:** at the root, each move is searched with the window `(best_so_far − 1, +∞)`. Scores are integers, so any move that *ties* the best score is still computed exactly, which is what makes random tie-breaking between equally good moves correct under pruning.
6. **`Experiment.run_multiple_games`** alternates the starting player, `save_results` writes `results.csv`, and `summarize_results` recomputes wins, draws and averages by **reading that file back** — nothing is hard-coded.

## 8. Assignment checklist

| Requirement | Where |
|---|---|
| Python, game engine (display, valid moves, move, win, draw, terminal) | `game.py` |
| Minimax and Alpha-Beta, own implementation, nodes evaluated / pruned counted | `minimax.py` |
| Heuristic evaluation function, two reasonable heuristics | `heuristic.py` |
| Configurable depth, player and heuristic (nothing hard-coded) | `Agent(name, depth, heuristic)`, `--nexus-depth`, `--titan-depth` |
| Two named agents with different strategies, neither random | `agents.py` (NEXUS, TITAN) |
| Automatic AI-vs-AI games, 10 games, alternating starting player | `Experiment.run_multiple_games` |
| Depth experiment (Part 10) | `run_depth_experiment`, `results/depth_experiment.csv` |
| Per-game statistics saved to a file, analysis computed from it | `results/results.csv`, `summarize_results` |
| Classes (`TicTacToe`, `Agent`, `Experiment`) | `game.py`, `agents.py`, `experiment.py` |
| README and short analysis / report | `README.md`, `REPORT.md` |
