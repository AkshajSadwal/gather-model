import unittest

import numpy as np

from scripts.diagnose_retrieval import (
    cluster_bootstrap_mean,
    popularity_ranks,
    subtract_seen_positive_competitors,
)


class RetrievalDiagnosticTests(unittest.TestCase):
    def test_seen_positive_filter_removes_only_other_known_positive_items(self):
        # Catalogue rows are 1-based IDs; target ID 3 has score 0.5.
        embeddings = np.asarray([
            [1.0, 0.0],
            [0.8, 0.0],
            [0.5, 0.0],
            [-1.0, 0.0],
        ], dtype=np.float16)
        vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
        raw = np.asarray([3], dtype=np.int64)
        positive_scores = np.asarray([0.5], dtype=np.float32)
        filtered = subtract_seen_positive_competitors(
            raw, vectors, positive_scores,
            np.asarray([7]), np.asarray([3]),
            {7: np.asarray([1, 2, 3], dtype=np.int32)}, embeddings,
        )
        self.assertEqual(int(filtered[0]), 1)

    def test_filter_does_not_remove_target_when_it_is_a_train_positive(self):
        embeddings = np.asarray([[1.0, 0.0], [0.5, 0.0]], dtype=np.float16)
        filtered = subtract_seen_positive_competitors(
            np.asarray([2]), np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([0.5], dtype=np.float32), np.asarray([4]), np.asarray([2]),
            {4: np.asarray([2], dtype=np.int32)}, embeddings,
        )
        self.assertEqual(int(filtered[0]), 2)

    def test_popularity_filter_adjusts_other_seen_items_only(self):
        popularity = np.asarray([10.0, 8.0, 5.0, 1.0])
        users = np.asarray([3])
        target = np.asarray([3])
        raw = popularity_ranks(popularity, target, users, {3: np.asarray([1, 2, 3])}, False)
        filtered = popularity_ranks(popularity, target, users, {3: np.asarray([1, 2, 3])}, True)
        self.assertGreater(raw[0], filtered[0])
        self.assertAlmostEqual(float(filtered[0]), 1.0)

    def test_cluster_bootstrap_is_reproducible_and_query_weighted(self):
        values = np.asarray([1.0, 0.0, 1.0, 1.0])
        users = np.asarray([1, 1, 2, 3])
        a = cluster_bootstrap_mean(values, users, 100, 123)
        b = cluster_bootstrap_mean(values, users, 100, 123)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a[0], 0.75)


if __name__ == "__main__":
    unittest.main()
