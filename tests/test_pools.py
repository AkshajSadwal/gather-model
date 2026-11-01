"""Unit tests for time windows, pools, time-local batches and the softmax batch builder."""
from __future__ import annotations

import unittest

import numpy as np

from common.pools import (
    HOUR_MS, KnownPairs, WindowPools, build_softmax_batch, expected_ranks_in_pool,
    percentile_from_ranks, positive_frequency, reference_pool_softmax_loss,
    time_local_batches, window_ids,
)


class WindowTests(unittest.TestCase):
    def test_window_ids_count_from_origin(self):
        t = np.array([0, 1, 24 * HOUR_MS - 1, 24 * HOUR_MS, 49 * HOUR_MS], dtype=np.int64)
        np.testing.assert_array_equal(window_ids(t, 0, 24), [0, 0, 0, 1, 2])
        np.testing.assert_array_equal(window_ids(t + 5, 5, 24), [0, 0, 0, 1, 2])
        with self.assertRaises(ValueError):
            window_ids(t, 0, 0)

    def test_pools_hold_unique_items_per_window_and_skip_item_zero(self):
        windows = np.array([0, 0, 0, 1, 1, 3, 3, 3])
        items = np.array([5, 5, 7, 0, 9, 2, 2, 8])
        pools = WindowPools.build(windows, items)
        np.testing.assert_array_equal(pools.get(0), [5, 7])
        np.testing.assert_array_equal(pools.get(1), [9])
        self.assertEqual(pools.size(2), 0)           # gap window
        np.testing.assert_array_equal(pools.get(3), [2, 8])
        self.assertEqual(pools.size(99), 0)          # out of range
        self.assertEqual(pools.size(-5), 0)
        self.assertEqual(pools.nearest_nonempty(2) in (1, 3), True)
        self.assertEqual(pools.nearest_nonempty(0), 0)

    def test_sampling_stays_inside_the_window_pool(self):
        pools = WindowPools.build(np.array([4, 4, 4, 5]), np.array([10, 11, 12, 99]))
        rng = np.random.default_rng(0)
        draw = pools.sample(4, 500, rng)
        self.assertEqual(draw.size, 500)
        self.assertTrue(set(draw.tolist()) <= {10, 11, 12})
        self.assertEqual(pools.sample(7, 10, rng).size, 0)

    def test_empty_input_gives_empty_pools(self):
        pools = WindowPools.build(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        self.assertEqual(pools.size(0), 0)
        self.assertIsNone(pools.nearest_nonempty(0))


class BatchTests(unittest.TestCase):
    def test_batches_are_consecutive_complete_and_shuffled(self):
        rows = np.arange(10_000, 10_000 + 1000)
        rng = np.random.default_rng(1)
        batches = time_local_batches(rows, 64, rng)
        self.assertEqual(len(batches), 1000 // 64)
        for b in batches:
            self.assertEqual(b.size, 64)
            np.testing.assert_array_equal(np.diff(b), np.ones(63))      # consecutive in time
        starts = [int(b[0]) for b in batches]
        self.assertNotEqual(starts, sorted(starts))                      # order shuffled
        used = np.concatenate(batches)
        self.assertEqual(np.unique(used).size, used.size)                # no row twice
        self.assertLess(1000 - used.size, 64)                            # at most batch-1 skipped

    def test_offset_changes_between_epochs_and_small_inputs_work(self):
        rows = np.arange(500)
        firsts = {int(min(b[0] for b in time_local_batches(rows, 64, np.random.default_rng(s)))) for s in range(30)}
        self.assertGreater(len(firsts), 1)
        self.assertEqual(len(time_local_batches(np.arange(10), 64, np.random.default_rng(0))), 1)
        self.assertEqual(time_local_batches(np.array([], dtype=int), 64, np.random.default_rng(0)), [])
        # slack never causes an empty result when rows >= batch_size
        for n in (64, 65, 127, 128, 129):
            self.assertEqual(len(time_local_batches(np.arange(n), 64, np.random.default_rng(n))), n // 64)


class KnownPairsTests(unittest.TestCase):
    def test_grid_matches_python_set_including_non_multiple_of_8_item_counts(self):
        rng = np.random.default_rng(2)
        n_users, n_items = 12, 203
        users = rng.integers(0, n_users, 400)
        items = rng.integers(0, n_items, 400)
        known = KnownPairs(users, items, n_users, n_items)
        truth = set(zip(users.tolist(), items.tolist()))
        q_users = rng.integers(0, n_users, 9)
        q_items = np.concatenate([[0, n_items - 1], rng.integers(0, n_items, 50)])
        grid = known.grid(q_users, q_items)
        self.assertEqual(grid.shape, (9, q_items.size))
        for a, u in enumerate(q_users):
            for b, i in enumerate(q_items):
                self.assertEqual(bool(grid[a, b]), (int(u), int(i)) in truth)
        self.assertEqual(known.count(), len(truth))

    def test_out_of_range_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            KnownPairs(np.array([5]), np.array([1]), 3, 10)
        with self.assertRaises(ValueError):
            KnownPairs(np.array([1]), np.array([10]), 3, 10)


class SoftmaxBatchTests(unittest.TestCase):
    def setUp(self):
        self.n_items = 50
        # user 1 knows items 5, 6, 7; user 2 knows 7, 9
        users = np.array([1, 1, 1, 2, 2])
        items = np.array([5, 6, 7, 7, 9])
        self.known = KnownPairs(users, items, 4, self.n_items)
        self.freq = positive_frequency(items, self.n_items)

    def test_candidates_columns_and_exclusions(self):
        batch_users = np.array([1, 2, 1])
        positives = np.array([5, 9, 7])
        shared = np.array([6, 20, 20, 30])
        cands, own, exclude, logq = build_softmax_batch(batch_users, positives, shared, self.known, self.freq, pool_size=40)
        np.testing.assert_array_equal(cands, [5, 6, 7, 9, 20, 30])
        np.testing.assert_array_equal(cands[own], positives)
        col = {int(c): i for i, c in enumerate(cands)}
        # row 0: user 1, own item 5. Other known positives of user 1 (6, 7) are excluded; own stays in.
        self.assertFalse(exclude[0, col[5]])
        self.assertTrue(exclude[0, col[6]] and exclude[0, col[7]])
        self.assertFalse(exclude[0, col[9]] or exclude[0, col[20]] or exclude[0, col[30]])
        # row 1: user 2, own item 9; user 2 also knows 7
        self.assertFalse(exclude[1, col[9]])
        self.assertTrue(exclude[1, col[7]])
        self.assertFalse(exclude[1, col[5]])
        # row 2: user 1, own item 7; 5 and 6 are excluded, 7 stays
        self.assertFalse(exclude[2, col[7]])
        self.assertTrue(exclude[2, col[5]] and exclude[2, col[6]])
        # every row keeps its own positive in the denominator
        self.assertTrue(np.all(~exclude[np.arange(3), own]))
        expected = 3 * self.freq[cands] + 4 / 40
        np.testing.assert_allclose(logq, np.log(expected).astype(np.float32), rtol=1e-6)

    def test_logq_strength_zero_disables_correction(self):
        _, _, _, logq = build_softmax_batch(np.array([1]), np.array([5]), np.array([20]), self.known, self.freq, 10, logq_strength=0.0)
        np.testing.assert_array_equal(logq, np.zeros(2, dtype=np.float32))

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            build_softmax_batch(np.array([1, 2]), np.array([5]), np.array([6]), self.known, self.freq, 10)


class RankTests(unittest.TestCase):
    def test_ties_and_invalid_items(self):
        scores = np.array([[0.9, 0.5, 0.5, 0.5, 0.1, 0.99],
                           [0.2, 0.9, 0.1, 0.3, 0.3, 0.3]])
        target = np.array([0.5, 0.3])
        valid = np.array([[True, True, True, True, True, False],   # item 5 (0.99) is not allowed to compete
                          [True, True, True, True, True, True]])
        ranks, comps = expected_ranks_in_pool(scores, target, valid)
        # row 0: 1 strictly greater (0.9) + 3 tied (0.5 x3, incl. the target's own column treated as competitor here)
        self.assertEqual(comps.tolist(), [5, 6])
        self.assertAlmostEqual(ranks[0], 1 + 1 + 0.5 * 3)
        self.assertAlmostEqual(ranks[1], 1 + 1 + 0.5 * 3)
        pct = percentile_from_ranks(np.array([1.0, 3.5, 11.0]), np.array([10, 10, 10]))
        np.testing.assert_allclose(pct, [0.0, 0.25, 1.0])


class ReferenceLossTests(unittest.TestCase):
    def test_matches_loop_implementation_and_handles_exclusion(self):
        rng = np.random.default_rng(3)
        B, C, D = 7, 11, 5
        u = rng.normal(size=(B, D)); v = rng.normal(size=(C, D))
        own = rng.integers(0, C, B)
        exclude = rng.random((B, C)) < 0.3
        exclude[np.arange(B), own] = False
        logq = rng.normal(size=C).astype(np.float32)
        w = rng.random(B) + 0.5
        loss, per_row = reference_pool_softmax_loss(u, v, own, exclude, logq, 0.1, w)
        for r in range(B):
            logits = (u[r] @ v.T) / 0.1 - logq
            keep = ~exclude[r]
            expected = np.log(np.exp(logits[keep]).sum()) - logits[own[r]]
            self.assertAlmostEqual(per_row[r], expected, places=9)
        self.assertAlmostEqual(loss, float((per_row * w).sum() / w.sum()), places=9)
        self.assertTrue(np.all(per_row >= -1e-9))      # own logit is part of the denominator

    def test_perfect_separation_gives_near_zero_loss(self):
        v = np.eye(4)
        u = np.eye(4)
        loss, _ = reference_pool_softmax_loss(u, v, np.arange(4), np.zeros((4, 4), bool), np.zeros(4), 0.05, np.ones(4))
        self.assertLess(loss, 1e-6)


if __name__ == "__main__":
    unittest.main()
