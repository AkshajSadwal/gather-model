from __future__ import annotations

import unittest

import numpy as np

from common.creator_history import (
    CreatorHistoryIndex,
    build_creator_events,
    positive_event_strengths,
)

TARGETS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "long_view", "is_profile_enter", "complete_view", "short_view",
]


class CreatorHistoryTests(unittest.TestCase):
    def test_positive_strength_matches_retrieval_relevance_and_excludes_hate(self):
        labels = np.zeros((4, len(TARGETS)), dtype=np.float32)
        labels[0, TARGETS.index("is_like")] = 1
        labels[1, TARGETS.index("long_view")] = 1
        labels[2, TARGETS.index("short_view")] = 1
        labels[3, TARGETS.index("is_like")] = 1
        labels[3, TARGETS.index("is_hate")] = 1

        got = positive_event_strengths(labels, TARGETS)
        np.testing.assert_allclose(got, [1.0, 2.0, 0.0, 0.0])

    def test_builder_filters_oov_users_videos_authors_hate_and_nonpositive_rows(self):
        labels = np.zeros((6, len(TARGETS)), dtype=np.float32)
        labels[0, TARGETS.index("is_like")] = 1
        labels[1, TARGETS.index("is_like")] = 1  # user OOV
        labels[2, TARGETS.index("is_like")] = 1  # video OOV
        labels[3, TARGETS.index("is_like")] = 1  # author OOV
        labels[4, TARGETS.index("short_view")] = 1
        labels[5, TARGETS.index("is_like")] = 1
        labels[5, TARGETS.index("is_hate")] = 1

        result = build_creator_events(
            users=np.array([1, 0, 1, 1, 1, 1]),
            catalog_video_indices=np.array([1, 1, 0, 2, 1, 1]),
            labels=labels,
            times_ms=np.array([10, 20, 30, 40, 50, 60]),
            catalog_author_ids=np.array([0, 4, 0, 5]),
            target_names=TARGETS,
            num_users=3,
            num_authors=8,
            split_code=1,
        )
        np.testing.assert_array_equal(result["users"], [1])
        np.testing.assert_array_equal(result["authors"], [4])
        np.testing.assert_array_equal(result["times_ms"], [10])
        np.testing.assert_allclose(result["weights"], [1.0])
        np.testing.assert_array_equal(result["split_codes"], [1])

    def make_index(self):
        return CreatorHistoryIndex.from_events(
            users=np.array([1, 1, 1, 1, 2]),
            authors=np.array([5, 5, 6, 7, 5]),
            times_ms=np.array([100, 200, 200, 300, 150]),
            weights=np.array([1.0, 2.0, 4.0, 8.0, 9.0]),
            num_users=4,
            num_authors=10,
        )

    def test_query_uses_only_events_strictly_before_timestamp(self):
        index = self.make_index()
        # Author 5 at time 100 is visible; events at 200 are not visible at 200.
        got = index.scores(1, 200, np.array([5, 6, 7, 0]))
        np.testing.assert_allclose(got, [1.0, 0.0, 0.0, 0.0])

    def test_future_and_same_time_events_cannot_change_earlier_query(self):
        base = CreatorHistoryIndex.from_events(
            users=np.array([1]), authors=np.array([5]), times_ms=np.array([100]),
            weights=np.array([2.0]), num_users=3, num_authors=8,
        )
        extended = CreatorHistoryIndex.from_events(
            users=np.array([1, 1, 1]), authors=np.array([5, 5, 5]),
            times_ms=np.array([100, 150, 200]), weights=np.array([2.0, 50.0, 75.0]),
            num_users=3, num_authors=8,
        )
        a = base.scores(1, 150, np.array([5]))
        b = extended.scores(1, 150, np.array([5]))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(b, [2.0])

    def test_lookback_window_has_inclusive_lower_and_exclusive_upper_bound(self):
        index = CreatorHistoryIndex.from_events(
            users=np.array([1, 1, 1]), authors=np.array([5, 6, 7]),
            times_ms=np.array([99, 100, 200]), weights=np.array([1.0, 2.0, 4.0]),
            num_users=3, num_authors=10,
        )
        got = index.scores(1, 200, np.array([5, 6, 7]), lookback_days=100 / 86_400_000)
        # With 100ms lookback, [100, 200): time 100 is included, 200 excluded.
        np.testing.assert_allclose(got, [0.0, 2.0, 0.0])

    def test_half_life_decays_only_prior_events(self):
        index = CreatorHistoryIndex.from_events(
            users=np.array([1, 1]), authors=np.array([5, 5]),
            times_ms=np.array([0, 86_400_000]), weights=np.array([1.0, 1.0]),
            num_users=3, num_authors=8,
        )
        got = index.scores(1, 2 * 86_400_000, np.array([5]), half_life_days=1)
        np.testing.assert_allclose(got, [0.75], rtol=1e-6)

    def test_top_authors_is_deterministic_on_equal_scores(self):
        index = CreatorHistoryIndex.from_events(
            users=np.array([1, 1, 1]), authors=np.array([7, 5, 6]),
            times_ms=np.array([10, 10, 10]), weights=np.array([1.0, 1.0, 1.0]),
            num_users=3, num_authors=10,
        )
        authors, scores = index.top_authors(1, 11, 2)
        np.testing.assert_array_equal(authors, [5, 6])
        np.testing.assert_allclose(scores, [1.0, 1.0])

    def test_unknown_user_returns_zero_scores(self):
        index = self.make_index()
        np.testing.assert_array_equal(index.scores(0, 1000, np.array([5, 6])), [0.0, 0.0])

    def test_invalid_query_options_raise(self):
        index = self.make_index()
        with self.assertRaises(ValueError):
            index.scores(1, 1000, np.array([5]), lookback_days=0)
        with self.assertRaises(ValueError):
            index.scores(1, 1000, np.array([5]), half_life_days=-1)


if __name__ == "__main__":
    unittest.main()
