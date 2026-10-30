"""Fast unit tests for the two-tower retrieval feature contract."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from model1_retrieval.data import PreparedSplit, positive_rows, sample_negative_video_indices
from model1_retrieval.model import TwoTowerRetrieval


TARGETS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "long_view", "is_profile_enter", "complete_view", "short_view",
]


class RetrievalCoreTests(unittest.TestCase):
    def test_user_and_video_towers_emit_normalized_128d_vectors(self) -> None:
        metadata = {
            "dataset": "unit-test",
            "categorical_sizes": [max(8, 4 + (i % 4)) for i in range(36)],
        }
        model = TwoTowerRetrieval(metadata)
        user_cat = torch.ones((3, len(model.USER_COLUMNS)), dtype=torch.long)
        user_num = torch.zeros((3, len(model.USER_NUMERIC_COLUMNS)), dtype=torch.float32)
        video_cat = torch.ones((3, len(model.VIDEO_COLUMNS)), dtype=torch.long)
        video_num = torch.zeros((3, len(model.VIDEO_NUMERIC_COLUMNS)), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            user_vectors = model.user_tower(user_cat, user_num)
            video_vectors = model.video_tower(video_cat, video_num)
        self.assertEqual(tuple(user_vectors.shape), (3, 128))
        self.assertEqual(tuple(video_vectors.shape), (3, 128))
        self.assertTrue(torch.allclose(user_vectors.norm(dim=1), torch.ones(3), atol=1e-5))
        self.assertTrue(torch.allclose(video_vectors.norm(dim=1), torch.ones(3), atol=1e-5))
        self.assertEqual(len(model.VIDEO_NUMERIC_COLUMNS), 3)

    def test_hate_overrides_positive_and_oov_rows_are_excluded(self) -> None:
        categorical = np.zeros((4, 36), dtype=np.int32)
        categorical[:, 0] = [1, 1, 0, 1]
        categorical[:, 1] = [1, 2, 3, 4]
        categorical = np.vstack([categorical, np.zeros((2, 36), dtype=np.int32)])
        categorical[4, 0], categorical[4, 1] = 1, 5
        categorical[5, 0], categorical[5, 1] = 1, 6
        labels = np.zeros((6, 10), dtype=np.float32)
        labels[0, 0] = 1.0  # weak click positive
        labels[1, 1] = 1.0  # like, but also explicit hate: hate wins
        labels[1, 5] = 1.0
        labels[2, 8] = 1.0  # OOV user: excluded
        labels[3, 8] = 1.0  # complete view positive
        labels[4, 0] = 1.0  # click + short skip: excluded
        labels[4, 9] = 1.0
        labels[5, 1] = 1.0  # like outweighs short-view penalty
        labels[5, 9] = 1.0
        split = PreparedSplit(categorical, np.zeros((6, 10), dtype=np.float32), labels)
        metadata = {"binary_targets": TARGETS}
        rows, weights = positive_rows(split, metadata)
        np.testing.assert_array_equal(rows, np.asarray([0, 3, 5]))
        self.assertEqual(weights.shape, (3,))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_negative_sampler_rejects_known_positive_pairs_and_uses_hate_pair(self) -> None:
        users = np.asarray([1, 2], dtype=np.int64)
        positive_videos = np.asarray([2, 3], dtype=np.int32)
        num_videos = 10
        known_positive_keys = np.asarray([1 * num_videos * 1 + 2, 2 * num_videos * 1 + 3], dtype=np.int64)
        hates = {1: np.asarray([8], dtype=np.int32), 2: np.asarray([7], dtype=np.int32)}
        negatives = sample_negative_video_indices(
            users, positive_videos, num_videos, known_positive_keys, hates,
            np.random.default_rng(7), negatives_per_positive=1, hard_negative_sample_rate=1.0,
        )
        np.testing.assert_array_equal(negatives[:, 0], np.asarray([8, 7], dtype=np.int32))
        negative_keys = users * num_videos + negatives[:, 0].astype(np.int64)
        self.assertFalse(np.isin(negative_keys, known_positive_keys).any())
        self.assertFalse(np.any(negatives[:, 0] == positive_videos))


if __name__ == "__main__":
    unittest.main()
