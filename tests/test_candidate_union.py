import unittest

import numpy as np

from scripts.evaluate_retrieval_pools import (
    empty_candidate_state, record_candidate_state, select_topk_columns,
)


class CandidateSelectionTests(unittest.TestCase):
    def test_selects_top_k_eligible_columns(self):
        scores = np.array([100.0, 0.1, 0.9, 0.8, 0.7])
        eligible = np.array([1, 2, 3, 4])
        chosen = select_topk_columns(scores, eligible, 2)
        self.assertEqual(set(chosen.tolist()), {2, 3})

    def test_author_selection_does_not_fill_with_zero_affinity(self):
        scores = np.array([0.0, 0.4, 0.0, 0.2])
        eligible = np.array([0, 1, 2, 3])
        chosen = select_topk_columns(scores, eligible, 10, require_positive=True)
        self.assertEqual(set(chosen.tolist()), {1, 3})

    def test_empty_eligible_set(self):
        chosen = select_topk_columns(np.zeros(3), np.array([], dtype=np.int64), 5)
        self.assertEqual(chosen.size, 0)

    def test_weighted_author_scores_can_change_candidate_selection(self):
        eligible = np.arange(5, dtype=np.int64)
        model_scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        unweighted = np.array([3.0, 0.0, 2.0, 0.0, 1.0])
        weighted = np.array([0.2, 2.0, 0.1, 1.5, 0.0])

        model_cols = select_topk_columns(model_scores, eligible, 2)
        unweighted_cols = select_topk_columns(
            unweighted, eligible, 2, require_positive=True
        )
        weighted_cols = select_topk_columns(
            weighted, eligible, 2, require_positive=True
        )

        self.assertEqual(set(unweighted_cols.tolist()), {0, 2})
        self.assertEqual(set(weighted_cols.tolist()), {1, 3})

        unweighted_state = empty_candidate_state(1)
        weighted_state = empty_candidate_state(1)
        record_candidate_state(unweighted_state, 0, model_cols, unweighted_cols, 3)
        record_candidate_state(weighted_state, 0, model_cols, weighted_cols, 3)

        self.assertFalse(unweighted_state["union_hit"][0])
        self.assertTrue(weighted_state["union_hit"][0])
        self.assertTrue(weighted_state["author_hit"][0])


if __name__ == "__main__":
    unittest.main()
