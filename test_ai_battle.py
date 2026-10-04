"""
test_ai_battle.py - Automated tests for the game engine, the search and the experiments.

Run with:
    python -m unittest -v
"""

import os
import random
import tempfile
import unittest

from agents import Agent, create_nexus, create_titan
from experiment import (DRAW, Experiment, enumerate_positions, load_rows,
                        summarize_results)
from game import PLAYER_O, PLAYER_X, TicTacToe
from heuristic import HEURISTICS, MAX_HEURISTIC, WIN_SCORE
from minimax import search_best_move


class TestGameEngine(unittest.TestCase):
    def test_new_board_is_empty(self):
        game = TicTacToe()
        self.assertEqual(len(game.get_valid_moves()), 9)
        self.assertFalse(game.is_terminal())
        self.assertIsNone(game.check_winner())

    def test_make_and_undo_move(self):
        game = TicTacToe()
        game.make_move((1, 1), PLAYER_X)
        self.assertEqual(game.board[1][1], PLAYER_X)
        self.assertNotIn((1, 1), game.get_valid_moves())
        game.undo_move((1, 1))
        self.assertIn((1, 1), game.get_valid_moves())
        self.assertEqual(game.move_count, 0)

    def test_illegal_moves_rejected(self):
        game = TicTacToe()
        game.make_move((0, 0), PLAYER_X)
        with self.assertRaises(ValueError):
            game.make_move((0, 0), PLAYER_O)
        with self.assertRaises(ValueError):
            game.make_move((3, 0), PLAYER_O)

    def test_all_eight_winning_lines(self):
        keys = ["XXX......", "...XXX...", "......XXX",
                "X..X..X..", ".X..X..X.", "..X..X..X",
                "X...X...X", "..X.X.X.."]
        for key in keys:
            game = TicTacToe.from_key(key)
            self.assertEqual(game.check_winner(), PLAYER_X, key)
            self.assertTrue(game.is_terminal(), key)
            self.assertFalse(game.is_draw(), key)

    def test_draw_detection(self):
        game = TicTacToe.from_key("XOXXOOOXX")
        self.assertIsNone(game.check_winner())
        self.assertTrue(game.is_draw())
        self.assertTrue(game.is_terminal())

    def test_current_player_and_key_round_trip(self):
        game = TicTacToe.from_key("XO.......")
        self.assertEqual(game.current_player(), PLAYER_X)
        self.assertEqual(TicTacToe.from_key(game.to_key()).to_key(), "XO.......")

    def test_reachable_positions_count(self):
        # 5,478 reachable positions in total; 958 of them are finished games.
        self.assertEqual(len(enumerate_positions()), 4520)


class TestHeuristics(unittest.TestCase):
    def test_zero_sum_and_bounded(self):
        for key in enumerate_positions():
            game = TicTacToe.from_key(key)
            for name, heuristic in HEURISTICS.items():
                score_x = heuristic(game.board, PLAYER_X)
                score_o = heuristic(game.board, PLAYER_O)
                self.assertEqual(score_x, -score_o, (name, key))
                self.assertLessEqual(abs(score_x), MAX_HEURISTIC, (name, key))

    def test_threat_is_valued(self):
        # X owns an open two-in-a-row (cells 0,1 with 2 empty); O has one mark.
        with_threat = TicTacToe.from_key("XX.O.....")
        for heuristic in HEURISTICS.values():
            self.assertGreater(heuristic(with_threat.board, PLAYER_X), 0)
            # An opponent threat must be punished just as strongly.
            self.assertLess(heuristic(with_threat.board, PLAYER_O), 0)
        # A threat is worth clearly more than a single mark.
        lone = TicTacToe.from_key("X..O.....")
        for heuristic in HEURISTICS.values():
            self.assertGreater(heuristic(with_threat.board, PLAYER_X),
                               heuristic(lone.board, PLAYER_X))

    def test_finished_game_scores(self):
        won = TicTacToe.from_key("XXXOO....")
        for heuristic in HEURISTICS.values():
            self.assertEqual(heuristic(won.board, PLAYER_X), WIN_SCORE)
            self.assertEqual(heuristic(won.board, PLAYER_O), -WIN_SCORE)


class TestSearch(unittest.TestCase):
    def test_takes_immediate_win(self):
        for name, heuristic in HEURISTICS.items():
            for depth in (1, 2, 3, 4):
                game = TicTacToe.from_key("XX.OO....")
                result = search_best_move(game, PLAYER_X, depth, heuristic)
                self.assertEqual(result.best_moves, [(0, 2)], (name, depth))

    def test_must_block(self):
        # X: 0, 8   O: 3, 4   -> X to move, O threatens cell 5. X cannot win now, so it must block.
        game = TicTacToe.from_key("X..OO...X")
        self.assertEqual(game.current_player(), PLAYER_X)
        for name, heuristic in HEURISTICS.items():
            for depth in (2, 3, 4):
                result = search_best_move(game, PLAYER_X, depth, heuristic)
                self.assertEqual(result.best_moves, [(1, 2)], (name, depth))

    def test_depth_is_a_parameter(self):
        game = TicTacToe()
        heuristic = HEURISTICS["H1"]
        counts = [search_best_move(game, PLAYER_X, d, heuristic).stats.nodes_evaluated
                  for d in (1, 2, 3, 4)]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(set(counts)), 4)
        with self.assertRaises(ValueError):
            search_best_move(game, PLAYER_X, 0, heuristic)

    def test_alpha_beta_matches_minimax_everywhere(self):
        """Pruning must never change the decision (checked on all positions)."""
        positions = enumerate_positions()
        for name, heuristic in HEURISTICS.items():
            for depth in (1, 2, 3, 4):
                for key in positions:
                    game = TicTacToe.from_key(key)
                    player = game.current_player()
                    pruned = search_best_move(game, player, depth, heuristic, True)
                    plain = search_best_move(game, player, depth, heuristic, False)
                    self.assertEqual(pruned.best_moves, plain.best_moves,
                                     (name, depth, key))
                    self.assertEqual(pruned.score, plain.score, (name, depth, key))

    def test_alpha_beta_full_depth_matches_minimax(self):
        heuristic = HEURISTICS["H2"]
        for key in enumerate_positions():
            if 9 - key.count(".") < 3:
                continue  # keep the plain-Minimax runs short
            game = TicTacToe.from_key(key)
            player = game.current_player()
            pruned = search_best_move(game, player, 9, heuristic, True)
            plain = search_best_move(game, player, 9, heuristic, False)
            self.assertEqual(pruned.best_moves, plain.best_moves, key)
            self.assertEqual(pruned.score, plain.score, key)

    def test_alpha_beta_evaluates_fewer_nodes(self):
        heuristic = HEURISTICS["H1"]
        for depth in (3, 4, 5, 6):
            pruned = search_best_move(TicTacToe(), PLAYER_X, depth, heuristic, True)
            plain = search_best_move(TicTacToe(), PLAYER_X, depth, heuristic, False)
            self.assertLess(pruned.stats.nodes_evaluated, plain.stats.nodes_evaluated)
            self.assertGreater(pruned.stats.nodes_pruned, 0)
            self.assertEqual(plain.stats.nodes_pruned, 0)

    def test_full_depth_empty_board_is_a_draw(self):
        for heuristic in HEURISTICS.values():
            result = search_best_move(TicTacToe(), PLAYER_X, 9, heuristic)
            self.assertEqual(result.score, 0)

    def test_search_leaves_board_unchanged(self):
        game = TicTacToe.from_key("XO..X....")
        before = game.to_key()
        search_best_move(game, PLAYER_O, 4, HEURISTICS["H2"])
        self.assertEqual(game.to_key(), before)


class TestAgentsAndExperiment(unittest.TestCase):
    def test_agents_have_names_and_depths(self):
        nexus, titan = create_nexus(3), create_titan(4)
        self.assertEqual((nexus.name, nexus.depth), ("NEXUS", 3))
        self.assertEqual((titan.name, titan.depth), ("TITAN", 4))
        self.assertIsNot(nexus.heuristic, titan.heuristic)

    def test_perfect_players_draw(self):
        nexus, titan = create_nexus(9, seed=1), create_titan(9, seed=2)
        results = Experiment(nexus, titan).run_multiple_games(4)
        self.assertTrue(all(r.winner == DRAW for r in results))

    def test_deeper_agent_never_loses_to_depth_one(self):
        strong = Agent("STRONG", 9, HEURISTICS["H1"], seed=3)
        weak = Agent("WEAK", 1, HEURISTICS["H1"], seed=4)
        results = Experiment(strong, weak).run_multiple_games(10)
        self.assertTrue(all(r.winner != "WEAK" for r in results))

    def test_starting_player_alternates(self):
        results = Experiment(create_nexus(2, seed=1),
                             create_titan(2, seed=2)).run_multiple_games(6)
        self.assertEqual([r.first for r in results],
                         ["NEXUS", "TITAN"] * 3)

    def test_same_seed_same_games(self):
        def play():
            return Experiment(create_nexus(3, seed=5),
                              create_titan(3, seed=6)).run_multiple_games(4)
        first, second = play(), play()
        self.assertEqual([r.move_sequence for r in first],
                         [r.move_sequence for r in second])
        self.assertEqual([r.nodes for r in first], [r.nodes for r in second])

    def test_csv_round_trip_and_summary(self):
        experiment = Experiment(create_nexus(3, seed=1), create_titan(3, seed=2))
        results = experiment.run_multiple_games(10)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "results.csv")
            experiment.save_results(results, path)
            rows = load_rows(path)
            self.assertEqual(len(rows), 10)
            for row, result in zip(rows, results):
                self.assertEqual(row["winner"], result.winner)
                self.assertEqual(int(row["nexus_nodes"]), result.nodes["NEXUS"])
                self.assertEqual(int(row["titan_pruned"]), result.pruned["TITAN"])
            text = summarize_results(path)
            self.assertIn("Games played : 10", text)


if __name__ == "__main__":
    unittest.main()
