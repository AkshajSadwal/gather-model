import unittest

import numpy as np

from scripts.evaluate_retrieval_pools import AuthorAffinity


class AuthorAffinityTests(unittest.TestCase):
    def test_unweighted_counts_match_frequency_and_ignore_unknown_author(self):
        model = AuthorAffinity(
            users=np.array([1, 1, 1, 2]),
            authors=np.array([3, 3, 4, 3]),
            n_authors=6,
        )
        scores = model.scores(1, np.array([0, 3, 4, 5]))
        np.testing.assert_allclose(scores, [0.0, 2.0, 1.0, 0.0])

    def test_weighted_counts_sum_per_user_author(self):
        model = AuthorAffinity(
            users=np.array([1, 1, 1, 2]),
            authors=np.array([3, 3, 4, 3]),
            n_authors=6,
            weights=np.array([0.5, 1.5, 2.0, 7.0]),
        )
        scores = model.scores(1, np.array([3, 4]))
        np.testing.assert_allclose(scores, [2.0, 2.0])
        np.testing.assert_allclose(model.scores(2, np.array([3])), [7.0])

    def test_invalid_weights_and_shapes_raise(self):
        with self.assertRaises(ValueError):
            AuthorAffinity(np.array([1]), np.array([2]), 4, weights=np.array([-1.0]))
        with self.assertRaises(ValueError):
            AuthorAffinity(np.array([1, 2]), np.array([2]), 4)


if __name__ == "__main__":
    unittest.main()
