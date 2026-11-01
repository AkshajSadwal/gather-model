import unittest

import numpy as np

from common.eval_metrics import RELEVANCE_WEIGHTS, SHORT_VIEW_PENALTY
from common.ranker_variants import (
    build_score_variants, percentile_of_rank, summarize_percentile,
    retrieval_target_weights, variant_score_matrix,
)

TARGETS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
           "long_view", "is_profile_enter", "complete_view", "short_view"]


class RankerVariantTests(unittest.TestCase):
    def test_retrieval_weights_match_positive_rows_exactly(self):
        w = retrieval_target_weights()
        expected = dict(RELEVANCE_WEIGHTS)
        expected["short_view"] = -SHORT_VIEW_PENALTY
        self.assertEqual(w, expected)
        # positive_rows excludes is_hate examples; it does not apply an additive hate weight.
        self.assertNotIn("is_hate", w)

    def test_variants_include_every_head_and_negative_heads_are_flipped(self):
        variants = build_score_variants(TARGETS, {"is_like": 1.0})
        self.assertIn("current_weights", variants)
        self.assertIn("retrieval_target_weighted_sum", variants)
        self.assertEqual(variants["head:long_view"], {"long_view": 1.0})
        self.assertEqual(variants["head:short_view"], {"short_view": -1.0})
        self.assertEqual(variants["head:is_hate"], {"is_hate": -1.0})

    def test_score_matrix_columns_follow_variants(self):
        variants = build_score_variants(TARGETS, {"is_like": 2.0, "short_view": -1.0})
        probs = np.zeros((2, len(TARGETS)), dtype=np.float32)
        probs[0, TARGETS.index("is_like")] = 0.5
        probs[1, TARGETS.index("short_view")] = 0.25
        names, scores = variant_score_matrix(probs, TARGETS, variants)
        cur = names.index("current_weights")
        self.assertAlmostEqual(float(scores[0, cur]), 1.0)
        self.assertAlmostEqual(float(scores[1, cur]), -0.25)
        self.assertAlmostEqual(float(scores[1, names.index("head:short_view")]), -0.25)
        self.assertEqual(scores.shape, (2, len(names)))

    def test_empty_probabilities_give_empty_matrix(self):
        names, scores = variant_score_matrix(np.empty((0, 0)), TARGETS, build_score_variants(TARGETS, {}))
        self.assertEqual(scores.shape, (0, len(names)))

    def test_percentile_scale_and_missing_targets(self):
        out = percentile_of_rank(np.array([1.0, 5.5, 10.0, np.inf, 1.0]), np.array([10, 10, 10, 10, 1]))
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 0.5)
        self.assertAlmostEqual(out[2], 1.0)
        self.assertTrue(np.isnan(out[3]))      # target not retrieved
        self.assertTrue(np.isnan(out[4]))      # one-item slate carries no ordering information

    def test_random_order_averages_half_and_better_order_is_below(self):
        rng = np.random.default_rng(0)
        n = 4000
        counts = np.full(n, 200)
        users = rng.integers(0, 100, n)
        random_ranks = rng.integers(1, 201, n).astype(float)
        r = summarize_percentile(random_ranks, counts, users, 200, 1)
        self.assertAlmostEqual(r["mean_percentile"], 0.5, delta=0.02)
        self.assertLess(r["delta_ci95_user_cluster"][0], 0.0)
        self.assertGreater(r["delta_ci95_user_cluster"][1], 0.0)
        good = np.minimum(random_ranks, rng.integers(1, 201, n)).astype(float)   # biased toward the top
        g = summarize_percentile(good, counts, users, 200, 1)
        self.assertLess(g["delta_ci95_user_cluster"][1], 0.0)

    def test_no_covered_queries(self):
        r = summarize_percentile(np.array([np.inf]), np.array([5]), np.array([1]), 10, 0)
        self.assertEqual(r["queries_scored"], 0)
        self.assertIsNone(r["mean_percentile"])


if __name__ == "__main__":
    unittest.main()
