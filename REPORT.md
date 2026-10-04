# Brief Report — AI Agent Battle: Minimax + Alpha-Beta Tic-Tac-Toe

**Assignment X_02, AI/ML Laboratory, B.Tech 5th Semester — "Python MINIMAX Search: AI Agents Battle Arena"**

**Student:** <Maruf Mallick> &nbsp;|&nbsp; **Roll No.:** <29>

All numbers in this report come from the CSV files in `results/`, produced by `python main.py all` (seed 2026). Outcomes and node counts are exactly reproducible; times are measured on one machine and will differ on another.

---

## 1. Problem formulation

Tic-Tac-Toe is a two-player, zero-sum, deterministic game with perfect information, so it can be solved by adversarial search.

| Element | Definition |
|---|---|
| State | 3×3 board; each cell is `X`, `O` or empty |
| Players | **MAX** = the agent choosing a move, **MIN** = its opponent. `X` always moves first |
| Actions | Any empty cell (at most 9 at the start) |
| Transition | Place the mover's mark on the chosen cell |
| Terminal test | Three in a line for either side, or a full board |
| Utility | Win `+100`, draw `0`, loss `−100` (from the point of view of the agent being served) |

The game tree has 549,946 nodes (counting the empty board). A full-depth search is cheap for Tic-Tac-Toe, but the assignment asks us to **limit the depth** and judge unfinished positions with a heuristic, because that is how search is used on games that are too large to solve. The questions studied are: *(a)* does a deeper search make the agent better, *(b)* how does the heuristic change its decisions, and *(c)* what does thinking deeper cost?

## 2. Algorithm design

**Minimax.** `search_best_move` (in `minimax.py`) tries every legal move: it applies the move temporarily, runs the recursive `minimax` on the result with `depth − 1`, undoes the move, and keeps the best value. At MAX nodes the largest child value is taken, at MIN nodes the smallest.

**Depth.** Depth is a parameter of the search, never a constant: depth 1 = "AI move"; depth 2 = "AI move → opponent move"; depth 3 = "AI → opponent → AI", and so on. When `depth == 0` and the game is not over, the position is scored by the agent's heuristic.

**Scoring finished games.** A win at ply *p* scores `100 − p`, a loss `−100 + p`, a draw `0`. The small ply term makes the agent prefer quicker wins and slower losses; it never changes whether a move wins, draws or loses. Heuristic values are clamped to ±90, so they can never outrank a real win (at least 91).

**Alpha-Beta pruning.** `alpha` is the best value MAX can already guarantee, `beta` the best value MIN can already guarantee. When `alpha ≥ beta` the rest of the node's children cannot influence the decision and are skipped. The same function runs plain Minimax when `use_pruning=False`, so the two are compared on identical code.

**Counters.** *Nodes evaluated* = number of positions visited by the search (each recursive call counts once). *Nodes pruned* = number of child moves skipped by a cut-off (each stands for a whole subtree, so this is a lower bound on the work saved).

**Tie-breaking.** If several moves share exactly the best score, one is chosen at random with a seeded RNG, as the assignment allows. At the root, each move is searched with the window `(best_so_far − 1, +∞)`. Because scores are integers, this keeps every move that *ties* the best exact, so pruning cannot disturb the tie set.

**Correctness check.** Alpha-Beta must not change the Minimax decision. The automated tests compare the two on **every one of the 4,520 reachable non-finished positions** (both heuristics at depths 1–4 on all positions, and heuristic H2 at depth 9 on all positions with at least 3 marks): the best-move sets and scores are identical in every case. The depth experiment also asserts this on each run.

## 3. Evaluation heuristics

Both heuristics have the signature `h(board, player) → int` (score from `player`'s view), are zero-sum (`h(X) = −h(O)`), and are used unchanged throughout the experiments.

**H1 — line potential (NEXUS).** Examines the 8 winning lines. A line that contains both X and O is dead (0). A line open for one side only scores: two marks and an empty cell (an *open two*, i.e. a threat) **±10**, one mark **±1**; positive for the evaluated player, negative for the opponent. H1 therefore rewards creating many live lines and punishes leaving the opponent live lines.

**H2 — position + threats (TITAN).** Uses a square-value table (centre **+3**, corners **+2**, edges **+1**), counted as own squares minus opponent squares, plus the same **±10** for open twos. Lines with a single mark are ignored apart from the value of the square.

They share the threat weight and differ in the rest, so they are reasonable and comparable but not identical: H1 thinks in *lines*, H2 thinks in *squares*.

## 4. Experimental setup

| Item | Setting |
|---|---|
| Agents | NEXUS (Minimax + Alpha-Beta, H1) and TITAN (Minimax + Alpha-Beta, H2) |
| Main battle | both at depth 3, 10 games, starting player alternates (NEXUS starts games 1, 3, 5, …) |
| Experiment 1 | H1 agent at depth *d* = 1…9 vs a fixed baseline (H1, depth 2), 20 games per depth, alternating start; plus the first-move search on an empty board, Alpha-Beta vs plain Minimax |
| Decision analysis | best-move sets compared across all 4,520 reachable non-finished positions |
| Supplementary | NEXUS vs TITAN with both at depth 1, 2, 3, 4, 5 (20 games each) |
| Recorded per game | game number, first player, winner, moves, nodes evaluated and pruned per agent, game time, search time per agent, full move sequence → `results/results.csv` |

## 5. Results

### 5.1 The 10-game battle (NEXUS H1, depth 3 vs TITAN H2, depth 3)

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

| Statistic (from `results.csv`) | NEXUS | TITAN |
|---|---|---|
| Wins | 0 | 0 |
| Draws | 10 | 10 |
| Average nodes evaluated per game | 337.1 | 357.5 |
| Average nodes pruned per game | 180.6 | 184.8 |
| Average search time per game | 1.79 ms | 2.08 ms |
| Result as first player / as second player | 0 W, 5 D, 0 L / 0 W, 5 D, 0 L | 0 W, 5 D, 0 L / 0 W, 5 D, 0 L |

Average game length 9.0 moves, average game time 3.91 ms. **Only 6 of the 10 games were distinct** (games 4 & 6, 7 & 9, and 2 & 10 are identical move for move), which is why their node counts match. X opened in the centre (cell 4) in every game, whichever agent started; the second player answered with a corner (cell 0, 6 or 8).

### 5.2 Experiment 1 — does deeper thinking help?

Result and cost per move for an H1 agent at depth *d* against the fixed depth-2 baseline (20 games each):

| Depth | Result (W / D / L) | Avg nodes evaluated / move | Avg nodes pruned / move | Avg time / move (ms) |
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

The first move on an empty board, Alpha-Beta against plain Minimax (same position, same depth, identical decision):

| Depth | Minimax nodes | Alpha-Beta nodes | Alpha-Beta pruned | Nodes saved | Minimax time (ms) | Alpha-Beta time (ms) | Root score | Best cell(s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 9 | 9 | 0 | 0.0% | 0.052 | 0.056 | 4 | 4 |
| 2 | 81 | 45 | 36 | 44.4% | 0.452 | 0.241 | 1 | 4 |
| 3 | 585 | 184 | 163 | 68.5% | 3.364 | 0.991 | 12 | 4 |
| 4 | 3,609 | 836 | 651 | 76.8% | 20.926 | 4.501 | 0 | 4 |
| 5 | 18,729 | 2,993 | 1,748 | 84.0% | 97.847 | 14.755 | 12 | 4 |
| 6 | 73,449 | 6,174 | 4,881 | 91.6% | 382.345 | 27.781 | 0 | 4 |
| 7 | 221,625 | 19,367 | 9,375 | 91.3% | 875.819 | 75.550 | 11 | 4 |
| 8 | 422,073 | 24,554 | 12,919 | 94.2% | 1426.833 | 77.001 | 0 | all 9 tie |
| 9 | 549,945 | 31,620 | 12,187 | 94.3% | 1128.884 | 70.187 | 0 | all 9 tie |

How much did the decisions change as the depth increased (all 4,520 reachable positions, compared with the previous depth)?

| Depth | Identical best-move set | Completely different best moves |
|---|---|---|
| 2 (vs 1) | 78.05% | 4.96% |
| 3 (vs 2) | 89.51% | 3.85% |
| 4 (vs 3) | 94.51% | 1.15% |
| 5 (vs 4) | 96.86% | 0.44% |
| 6 (vs 5) | 98.94% | 0.00% |
| 7 (vs 6) | 99.82% | 0.00% |
| 8 (vs 7) | 99.98% | 0.00% |
| 9 (vs 8) | 100.00% | 0.00% |

### 5.3 Do NEXUS and TITAN choose different moves?

Over all 4,520 reachable positions (each agent at depth 3, comparing its set of best moves):

| Pieces on board | Positions | Identical best moves | Different but overlapping | Completely different |
|---|---|---|---|---|
| 0 | 1 | 1 | 0 | 0 |
| 1 | 9 | 9 | 0 | 0 |
| 2 | 72 | 44 | 28 | 0 |
| 3 | 252 | 198 | 54 | 0 |
| 4 | 756 | 708 | 40 | 8 |
| 5 | 1,140 | 1,064 | 76 | 0 |
| 6 | 1,372 | 1,372 | 0 | 0 |
| 7 | 696 | 696 | 0 | 0 |
| 8 | 222 | 222 | 0 | 0 |
| **All** | **4,520** | **4,314 (95.4%)** | **198 (4.4%)** | **8 (0.2%)** |

### 5.4 Supplementary: NEXUS vs TITAN at equal depths (20 games each)

| Depth | NEXUS wins | TITAN wins | Draws | Wins by starting player | Distinct games | NEXUS nodes/game | TITAN nodes/game |
|---|---|---|---|---|---|---|---|
| 1 | 13 | 0 | 7 | 10 | 14 | 21.7 | 20.9 |
| 2 | 0 | 0 | 20 | 0 | 15 | 96.6 | 96.5 |
| 3 | 0 | 0 | 20 | 0 | 11 | 350.5 | 355.8 |
| 4 | 0 | 0 | 20 | 0 | 18 | 1098.2 | 1040.5 |
| 5 | 0 | 0 | 20 | 0 | 17 | 2902.8 | 3092.4 |

At depth 1, NEXUS won all 10 of its games as the starting player and 3 of 10 as the second player; TITAN won none.

## 6. Analysis (the questions of Part 13)

**About depth**

1. **Did increasing search depth change the AI's decisions?** Yes, but with diminishing effect. Going from depth 1 to 2 changed the set of best moves in 21.95% of the reachable positions (and left no common best move in 4.96%); 2→3 changed 10.5%, 3→4 5.5%, 4→5 3.1%, 5→6 1.1%, and 8→9 changed none. On the empty board the best move is the centre (cell 4) for depths 1–7; at depths 8 and 9 the search reaches (or almost reaches) the end of the game and finds every first move equal, so all nine cells tie at score 0, which is the draw that perfect play gives. The root scores at shallow depths (4, 1, 12, 0, 12, 0, 11) are heuristic estimates and are not comparable between depths; they also alternate with depth parity, because the heuristic is applied after the agent's own move at odd depths and after the opponent's reply at even depths.
2. **Did deeper search increase execution time?** Yes. The empty-board search with Alpha-Beta took 0.056 ms at depth 1 and about 70–77 ms at depths 7–9; plain Minimax took up to 1,427 ms (depth 8). In real games the time per move rises from 0.040 ms (depth 1) to about 10 ms (depth 7) and then levels off, because the search cannot look further than the number of empty cells left, so at depths 7–9 most moves are searched all the way to the end of the game anyway.
3. **Did the number of evaluated nodes increase?** Yes, steeply. On the empty board, Alpha-Beta evaluated 9, 45, 184, 836, 2,993, 6,174, 19,367, 24,554 and 31,620 nodes for depths 1–9; plain Minimax 9, 81, 585, 3,609, 18,729, 73,449, 221,625, 422,073 and 549,945. Per move in real games it went from 5.3 to 4,349.
4. **Did Alpha-Beta reduce the number of nodes explored?** Yes. It saved 0% at depth 1 (nothing to prune with one ply), 44.4% at depth 2, 68.5% at depth 3, 84.0% at depth 5 and 94.3% at depth 9, making the search up to 18.5× faster (depth 8: 1,427 ms vs 77 ms), while returning exactly the same decision.

**Game performance versus computation.** Depth helped game results only at the very bottom: the depth-1 agent lost 6 of 20 games to the depth-2 baseline, whereas every agent from depth 2 to 9 lost none and drew all 20. Beyond depth 2 the extra computation (about 30× more nodes per move at depth 5 than at depth 2, and about 200× more at depth 9) bought no better *result*, because Tic-Tac-Toe is a draw with correct play and a depth-2 baseline already avoids every mistake it can be punished for in these games. Deeper search is therefore better in the sense of *decision quality* (it finds the same moves as perfect play in more positions) but, as the assignment warns, not automatically better in outcome, and always more expensive.

**About the two agents**

5. **Did the two agents make different decisions?** Sometimes. They agree exactly on their set of best moves in 95.4% of the 4,520 reachable positions, share some but not all best moves in 4.4%, and have no best move in common in only 8 positions (all with four marks on the board). In the real games this showed up as different second moves and different later moves: 6 distinct games in 10, but the same opening (centre) every time.
6. **How did their heuristics influence their behaviour?** At depth 3 almost not at all in the outcome: the search already sees the immediate tactics, so the heuristic only breaks ties between moves that are equally good in game-theoretic terms (for example which corner to take). The influence is clear when the search is short. With a one-move lookahead, TITAN (H2) could not win a single game against NEXUS (H1) in 20 attempts. A concrete case from game 1 of the depth-1 battle: after `X4 O2 X8 O0 X1`, X threatens to complete the 1-4-7 column. Blocking at cell 7 scores 0 under H1 and −1 under H2, while the counter-threat at cell 6 scores −1 under H1 and 0 under H2. H2's corner bonus makes it prefer the counter-threat, which loses a move; H1, which also counts the opponent's live lines, prefers the block. This is one example, not a full proof of why all 13 games were lost, but it shows the mechanism: both heuristics weight a threat equally for either side and neither knows whose turn it is, so the tie is decided by their secondary terms.
7. **Did the first-player advantage appear?** Not in the main battle: all 10 games were drawn, so neither the starter nor the second player won (0 starter wins). It appeared only when play was weak: at depth 1, NEXUS won 10 of 10 as starter but 3 of 10 as second player, and 10 of the 13 decisive games were won by the starting player. From depth 2 onward there were no decisive games at all, matching the fact that Tic-Tac-Toe is a draw with correct play.
8. **Which agent won more games in your experiment?** Neither in the required experiment: 0 wins each at depth 3. Only in the supplementary depth-1 experiment did one agent (NEXUS, 13–0) win more. The evidence does not support calling either agent better at depth 3.
9. **Were there many draws?** Yes: 10 of 10 in the main battle, and 20 of 20 at each of the equal depths 2 to 5. This is expected: two non-losing players of Tic-Tac-Toe draw.
10. **Did the agent that won more games also require more computation?** There is no winner at depth 3. At depth 3 TITAN used about 6% more nodes per game than NEXUS (357.5 vs 337.1) and 16% more search time (2.08 ms vs 1.79 ms) with the same result, and at depth 1, where NEXUS won most games, the two used almost the same effort (21.7 vs 20.9 nodes per game). In this experiment computation was set by the depth and by the positions reached, not by who played better.

## 7. Conclusion

* A depth-limited Minimax with Alpha-Beta pruning and a heuristic evaluation, built from scratch, plays Tic-Tac-Toe correctly: it takes wins, blocks threats, never lost at depth ≥ 2 in the games tried, and at depth 8 its best-move choices already match the full-depth (depth 9, perfect-play) search in 99.98% of all reachable positions.
* **Deeper search costs a lot** (nodes grow from 9 to 31,620 on the first move) **but improves results only while the agent is too shallow to see tactics.** Beyond depth 2 the games end in draws regardless, so deeper is not automatically better.
* **Alpha-Beta is essentially free speed:** up to 94% fewer nodes and up to about 18× less time, with an identical decision (verified on all reachable positions).
* **The heuristic matters when the search is shallow and fades as the search deepens.** At depth 1, H1 (line potential) clearly beat H2 (position + threats); at depth 3 the two agents are indistinguishable in results: 10 draws in 10 games, and no evidence that either is better.

**Limitations.** Tic-Tac-Toe is small and solved, so differences between agents are hard to see in game results; the 20-game samples are small, and the depth-1 result is based on one seed. The pruned-node counter counts skipped child moves, not the size of the skipped subtrees. Times are single-machine, single-threaded Python measurements.

**Reproduce:** `python main.py all` (about 25 seconds), then `python main.py analyze` to recompute every statistic from the CSV files. Run the tests with `python -m unittest -v`.
