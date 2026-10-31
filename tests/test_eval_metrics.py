"""Unit tests for the pure-NumPy evaluation metrics (no torch required)."""
from __future__ import annotations

import unittest

import numpy as np

from common.eval_metrics import (
    auc_binary, average_precision, bootstrap_mean_ci, calibration_summary,
    expected_rank_from_scores, grouped_ndcg, make_blocks, paired_diff_ci,
    per_group_auc, recall_hits, relevance_gains, reciprocal_ranks,
)

TARGETS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward",
           "is_hate", "long_view", "is_profile_enter", "complete_view", "short_view"]


class MetricTests(unittest.TestCase):
    def test_auc_and_ap_match_scikit_learn_including_ties(self):
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(1)
        y = (rng.random(2000) < 0.2).astype(int)
        score = np.round(rng.random(2000) + 0.5 * y, 1)  # rounding creates many ties
        self.assertAlmostEqual(auc_binary(y, score), roc_auc_score(y, score), places=10)
        continuous = rng.random(2000) + 0.5 * y
        self.assertAlmostEqual(average_precision(y, continuous), average_precision_score(y, continuous), places=10)

    def test_auc_is_nan_with_one_class(self):
        self.assertTrue(np.isnan(auc_binary(np.zeros(5), np.arange(5))))

    def test_per_group_auc_skips_single_class_groups(self):
        y = np.array([1, 0, 1, 0, 0, 0])
        s = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.4])
        g = np.array([1, 1, 2, 2, 3, 3])
        aucs, weights = per_group_auc(y, s, g)
        np.testing.assert_allclose(aucs, [1.0, 0.0])
        np.testing.assert_allclose(weights, [2, 2])

    def test_relevance_gains(self):
        labels = np.zeros((4, 10), dtype=np.float32)
        labels[0, TARGETS.index("is_like")] = 1                      # 1.0
        labels[1, TARGETS.index("is_like")] = 1
        labels[1, TARGETS.index("is_hate")] = 1                      # hate wins -> 0
        labels[2, TARGETS.index("is_click")] = 1
        labels[2, TARGETS.index("short_view")] = 1                   # 0.25 - 0.5 -> clipped to 0
        labels[3, TARGETS.index("complete_view")] = 1                # 2.0
        np.testing.assert_allclose(relevance_gains(labels, TARGETS), [1.0, 0.0, 0.0, 2.0])

    def test_expected_rank_splits_ties(self):
        catalogue = np.sort(np.array([0, 0, 0, 0, 5, 9], dtype=float))
        ranks = expected_rank_from_scores(catalogue, np.array([9.0, 5.0, 0.0]))
        np.testing.assert_allclose(ranks, [1.0, 2.0, 4.5])  # 2 above, 4 tied -> positions 3..6, mean 4.5

    def test_recall_and_mrr(self):
        ranks = np.array([1, 5, 50, 500])
        np.testing.assert_allclose(recall_hits(ranks, 10), [1, 1, 0, 0])
        np.testing.assert_allclose(reciprocal_ranks(ranks), [1, 0.2, 0.02, 0.002])

    def test_calibration_detects_inflated_probabilities(self):
        rng = np.random.default_rng(2)
        true_p = rng.random(20000) * 0.1
        y = (rng.random(20000) < true_p).astype(int)
        good = calibration_summary(true_p, y)
        inflated = calibration_summary(np.minimum(true_p * 5, 1.0), y)
        self.assertLess(good["ece"], 0.01)
        self.assertGreater(inflated["prediction_over_base"], 3.0)
        self.assertGreater(inflated["ece"], good["ece"])

    def test_bootstrap_ci_contains_mean_and_is_reproducible(self):
        values = np.random.default_rng(3).random(500)
        a = bootstrap_mean_ci(values, n_boot=200, seed=5)
        b = bootstrap_mean_ci(values, n_boot=200, seed=5)
        self.assertEqual(a, b)
        self.assertLessEqual(a[1], a[0])
        self.assertGreaterEqual(a[2], a[0])
        diff = paired_diff_ci(values, values, n_boot=50, seed=1)
        self.assertEqual(diff[0], 0.0)


class GroupedNdcgTests(unittest.TestCase):
    def test_cluster_bootstrap_is_reproducible_and_validates_shapes(self):
        from common.eval_metrics import cluster_bootstrap_mean_ci

        values = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0])
        users = np.asarray([10, 10, 20, 30, 30])
        first = cluster_bootstrap_mean_ci(values, users, n_boot=100, seed=7)
        second = cluster_bootstrap_mean_ci(values, users, n_boot=100, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], values.mean())
        with self.assertRaises(ValueError):
            cluster_bootstrap_mean_ci(values, users[:-1], n_boot=10)

    def test_hand_computed_values(self):
        # One group of 4 items, gains [3, 0, 2, 1]
        gain = np.array([3, 0, 2, 1], dtype=float)
        group = np.zeros(4, dtype=int)
        ideal = grouped_ndcg(gain, gain, group, (2, 4))
        np.testing.assert_allclose(ideal[2], [1.0])
        worst = grouped_ndcg(-gain, gain, group, (2, 4))   # reverse order: 0,1,2,3
        dcg2 = 0 / np.log2(2) + 1 / np.log2(3)
        idcg2 = 3 / np.log2(2) + 2 / np.log2(3)
        np.testing.assert_allclose(worst[2], [dcg2 / idcg2])

    def test_groups_without_relevance_are_dropped_and_tied_scores_are_not_ordered_by_input(self):
        gain = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=float)
        group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        out = grouped_ndcg(np.zeros(8), gain, group, (1,), seed=0)
        self.assertEqual(out[1].size, 2)
        # With all scores tied, the relevant item lands first about 1 time in 4 on average.
        many = [grouped_ndcg(np.zeros(8), gain, group, (1,), seed=s)[1].mean() for s in range(200)]
        self.assertTrue(0.1 < float(np.mean(many)) < 0.4)

    def test_make_blocks_splits_each_user_in_time_order(self):
        user = np.array([7, 3, 7, 3, 7, 7, 3])
        blocks = make_blocks(user, 2)
        # user 7 has 4 rows -> blocks 0,0,1,1 ; user 3 has 3 rows -> 0,0,1
        self.assertEqual(blocks[0], blocks[2])
        self.assertNotEqual(blocks[2], blocks[4])
        self.assertEqual(blocks[4], blocks[5])
        self.assertEqual(blocks[1], blocks[3])
        self.assertNotEqual(blocks[3], blocks[6])
        self.assertEqual(len(set(blocks.tolist())), 4)


if __name__ == "__main__":
    unittest.main()


class RankingReportTests(unittest.TestCase):
    """End-to-end check of the report on synthetic predictions with a known signal."""

    def _synthetic(self, informative: bool):
        rng = np.random.default_rng(11)
        users, per_user = 30, 120
        n = users * per_user
        user = np.repeat(np.arange(1, users + 1), per_user)
        video = rng.integers(1, 50, n)
        labels = (rng.random((n, 10)) < 0.08).astype(np.float32)
        labels[:, TARGETS.index("is_like")] = (rng.random(n) < 0.15)
        noise = rng.random((n, 10)) * 0.3
        probs = np.clip(noise + (0.5 * labels if informative else 0.0), 0.001, 0.999).astype(np.float32)
        watch_true = rng.random(n).astype(np.float32)
        watch_pred = (watch_true + rng.normal(0, 0.1, n)).astype(np.float32)
        return probs, watch_pred, labels, watch_true, user, video

    def test_informative_predictions_beat_random_and_report_is_json_safe(self):
        import json
        from common.ranking_eval import REFERENCE_SCORE, evaluate_ranking_predictions

        probs, wp, labels, wt, user, video = self._synthetic(informative=True)
        weights = {"is_like": 0.6, "long_view": 1.0, "short_view": -0.9, "is_hate": -2.0}
        report = evaluate_ranking_predictions(probs, wp, labels, wt, user, video, TARGETS, weights,
                                              popularity_by_video=np.random.default_rng(0).random(60),
                                              block_size=30, ks=(10,), n_boot=60, seed=3)
        self.assertGreater(report["mean_auc"], 0.9)
        self.assertGreater(report["mean_gauc"], 0.9)
        self.assertGreater(report["watch_time_log_seconds"]["r2"], 0.8)
        scores = report["ndcg_reranking_logged_impressions"]["scores"]
        self.assertEqual(report["ndcg_reranking_logged_impressions"]["blocks_used"], 30 * 4)
        self.assertGreater(scores[REFERENCE_SCORE]["@10"]["ndcg"], scores["random_order"]["@10"]["ndcg"] + 0.1)
        diff = scores["random_order"]["@10"]["diff_vs_hand_weighted"]
        self.assertLess(diff["ci95"][1], 0.0)  # random is clearly worse than the informative score
        self.assertNotIn("diff_vs_hand_weighted", scores[REFERENCE_SCORE]["@10"])
        json.dumps(report, allow_nan=True)

    def test_uninformative_predictions_are_near_chance(self):
        from common.ranking_eval import evaluate_ranking_predictions

        probs, wp, labels, wt, user, video = self._synthetic(informative=False)
        report = evaluate_ranking_predictions(probs, wp, labels, wt, user, video, TARGETS,
                                              {"is_like": 1.0}, block_size=30, ks=(10,), n_boot=30, seed=1)
        self.assertTrue(0.45 < report["mean_auc"] < 0.55)
        self.assertNotIn("popularity (train engagement)", report["ndcg_reranking_logged_impressions"]["scores"])


class CatalogRankTests(unittest.TestCase):
    def test_matches_brute_force_across_chunk_and_block_boundaries(self):
        from common.retrieval_eval import rank_against_catalog

        rng = np.random.default_rng(5)
        n, d, q = 1000, 16, 53
        catalog = rng.normal(size=(n, d)).astype(np.float16)
        users = rng.normal(size=(q, d)).astype(np.float32)
        positives = rng.integers(0, n, q)
        got = rank_against_catalog({"a": users, "b": users[::-1].copy()}, positives, catalog,
                                   chunk_size=128, query_block=7)
        full = catalog.astype(np.float32)
        for name, vectors in (("a", users), ("b", users[::-1])):
            scores = vectors @ full.T
            expected = np.empty(q, dtype=np.float64)
            for i in range(q):
                row = scores[i].copy()
                pos = row[positives[i]]
                row[positives[i]] = -np.inf
                expected[i] = 1 + int((row > pos).sum()) + 0.5 * int((row == pos).sum())
            np.testing.assert_allclose(got[name], expected)

    def test_exact_ties_receive_average_rank_not_optimistic_rank(self):
        from common.retrieval_eval import rank_against_catalog

        catalog = np.ones((8, 4), dtype=np.float32)
        users = np.ones((2, 4), dtype=np.float32)
        ranks = rank_against_catalog({"all_tied": users}, np.array([0, 7]), catalog,
                                     chunk_size=3, query_block=1)
        np.testing.assert_allclose(ranks["all_tied"], [4.5, 4.5])

    def test_perfect_user_vector_ranks_first(self):
        from common.retrieval_eval import rank_against_catalog

        rng = np.random.default_rng(6)
        catalog = rng.normal(size=(300, 8)).astype(np.float32)
        catalog /= np.linalg.norm(catalog, axis=1, keepdims=True)
        positives = np.array([3, 150, 299])
        ranks = rank_against_catalog({"oracle": catalog[positives]}, positives, catalog, chunk_size=64)
        np.testing.assert_array_equal(ranks["oracle"], [1, 1, 1])
