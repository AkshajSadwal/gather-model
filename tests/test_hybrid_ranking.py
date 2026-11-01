import unittest
import numpy as np

from scripts.evaluate_hybrid_ranking import (
    expected_rank, build_ranker_features, pool_columns_to_raw_ids, summarize_random_expected, add_random_comparison,
)


class HybridRankingTests(unittest.TestCase):
    def test_pool_columns_translate_through_active_pool_before_raw_id_map(self):
        # Pool entries are 1-based catalogue indices, while candidate_columns
        # are 0-based positions inside that pool.
        pool = np.array([2, 5, 9], dtype=np.int64)
        raw_ids = np.array([101, 202, 303, 404, 505, 606, 707, 808, 909], dtype=np.int64)
        actual = pool_columns_to_raw_ids(pool, np.array([0, 2]), raw_ids)
        self.assertEqual(actual.tolist(), [202, 909])

    def test_expected_rank_handles_ties_neutrally(self):
        self.assertEqual(expected_rank(np.array([0.9, 0.5, 0.5, 0.1]), 1), 2.5)
        self.assertEqual(expected_rank(np.array([0.9, 0.8, 0.1]), 0), 1.0)

    def test_build_ranker_features_maps_raw_ids_and_standardizes(self):
        catalog = {
            "video_ids": np.array([101, 202], dtype=np.int64),
            "video_model_indices": np.array([0, 1, 2], dtype=np.int32),
            "video_cat": np.array([[0, 0], [3, 4], [5, 6]], dtype=np.int32),
            "duration": np.array([0.0, 10.0, 20.0], dtype=np.float32),
            "width": np.array([0.0, 100.0, 200.0], dtype=np.float32),
            "height": np.array([0.0, 200.0, 400.0], dtype=np.float32),
            "upload_ms": np.array([0, 1_000, 2_000], dtype=np.int64),
        }
        meta = {
            "video_duration_fallback_ms": 0.0,
            "numeric_mean": [0.0] * 8,
            "numeric_std": [1.0] * 8,
            "num_numeric_features": 8,
        }
        cat, num, selected = build_ranker_features(
            np.array([202, 999, 101]), 3_600_000, 7, 2,
            np.array([11, 12], dtype=np.int32), np.array([0.2, 0.3], dtype=np.float32),
            catalog, meta,
        )
        self.assertEqual(selected.tolist(), [0, 2])
        self.assertEqual(cat.shape, (2, 7))
        self.assertEqual(num.shape, (2, 8))
        self.assertEqual(cat[:, 0].tolist(), [7, 7])
        self.assertEqual(cat[:, 1].tolist(), [2, 1])
        self.assertTrue(np.isfinite(num).all())

    def test_build_ranker_features_returns_empty_for_unmapped_items(self):
        catalog = {
            "video_ids": np.array([101], dtype=np.int64),
            "video_model_indices": np.array([0, 1], dtype=np.int32),
            "video_cat": np.zeros((2, 2), dtype=np.int32),
            "duration": np.zeros(2, dtype=np.float32),
            "width": np.zeros(2, dtype=np.float32),
            "height": np.zeros(2, dtype=np.float32),
            "upload_ms": np.zeros(2, dtype=np.int64),
        }
        meta = {"num_numeric_features": 8}
        cat, num, selected = build_ranker_features(
            np.array([999]), 0, 0, 0, np.zeros(2), np.zeros(2), catalog, meta
        )
        self.assertEqual(cat.shape[0], 0)
        self.assertEqual(num.shape[0], 0)
        self.assertEqual(selected.size, 0)


    def test_random_baseline_is_exact_expectation_for_same_candidate_sets(self):
        counts = np.array([10, 20, 0], dtype=np.int32)
        recovered = np.array([True, True, False])
        users = np.array([1, 2, 3])
        result = summarize_random_expected(counts, recovered, users, n_boot=100, seed=7)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        # Expected random Hit@10: 1 for n=10, 0.5 for n=20, 0 for absent.
        self.assertAlmostEqual(result["hit_at_10"]["value"], 0.5)
        self.assertAlmostEqual(result["mean_rank_when_recovered"], 8.0)


    def test_ranker_minus_random_is_paired_and_positive_for_top_rank(self):
        counts = np.array([10, 20, 30, 40], dtype=np.int32)
        ranks = np.array([1.0, 1.0, 1.0, 1.0])
        recovered = np.ones(4, dtype=bool)
        users = np.array([1, 2, 3, 4])
        ranker = {"hit_at_100": {"value": 1.0}}
        random_metrics = summarize_random_expected(counts, recovered, users, n_boot=100, seed=3)
        comparison = add_random_comparison(
            ranker, random_metrics, ranks, counts, recovered, users, n_boot=100, seed=3
        )
        self.assertGreater(comparison["ranker_minus_random"]["hit_at_10"]["difference"], 0.0)
        self.assertGreater(comparison["ranker_minus_random"]["mrr"]["difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
