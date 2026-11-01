"""Leakage-safe, timestamped user-to-creator positive-interaction history.

The index stores positive interaction events by user and event time. A query at
``query_time_ms`` can only see events with ``event_time_ms < query_time_ms``;
events at the exact query timestamp are excluded as a group.

This module is NumPy-only so its temporal rules can be tested without PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.eval_metrics import RELEVANCE_WEIGHTS, SHORT_VIEW_PENALTY

MS_PER_DAY = 86_400_000


def positive_event_strengths(labels: np.ndarray, target_names: list[str]) -> np.ndarray:
    """Return unnormalised retrieval-positive strength per impression.

    This matches ``model1_retrieval.data.positive_rows`` before its split-level
    mean normalisation: weighted positive actions/watch targets, minus the
    short-view penalty, explicit hate rows removed, then clipped to [0.25, 5].
    A row with non-positive net strength receives zero.
    """
    values = np.asarray(labels, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(target_names):
        raise ValueError("labels must be [N, len(target_names)]")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("labels must be finite values in [0, 1]")

    index = {name: i for i, name in enumerate(target_names)}
    required = set(RELEVANCE_WEIGHTS) | {"short_view", "is_hate"}
    missing = sorted(required.difference(index))
    if missing:
        raise ValueError(f"target_names missing required targets: {missing}")

    strength = np.zeros(values.shape[0], dtype=np.float32)
    for name, weight in RELEVANCE_WEIGHTS.items():
        strength += values[:, index[name]] * np.float32(weight)
    strength -= values[:, index["short_view"]] * np.float32(SHORT_VIEW_PENALTY)
    hate = values[:, index["is_hate"]] > 0
    positive = (strength > 0) & ~hate
    out = np.zeros_like(strength)
    out[positive] = np.clip(strength[positive], 0.25, 5.0)
    return out


def build_creator_events(
    users: np.ndarray,
    catalog_video_indices: np.ndarray,
    labels: np.ndarray,
    times_ms: np.ndarray,
    catalog_author_ids: np.ndarray,
    target_names: list[str],
    num_users: int,
    num_authors: int,
    split_code: int,
) -> dict[str, np.ndarray]:
    """Map one prepared split to valid positive user-creator events."""
    users = np.asarray(users, dtype=np.int64)
    video_indices = np.asarray(catalog_video_indices, dtype=np.int64)
    times = np.asarray(times_ms, dtype=np.int64)
    authors_by_video = np.asarray(catalog_author_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float32)

    n = users.size
    if users.ndim != 1 or video_indices.shape != (n,) or times.shape != (n,):
        raise ValueError("users, catalog_video_indices and times_ms must be matching 1-D arrays")
    if labels.shape != (n, len(target_names)):
        raise ValueError("labels do not align with users or target_names")
    if authors_by_video.ndim != 1 or authors_by_video.size < 1:
        raise ValueError("catalog_author_ids must be a non-empty 1-D array")
    if num_users < 1 or num_authors < 1:
        raise ValueError("num_users and num_authors must be positive")
    if split_code not in (0, 1, 2):
        raise ValueError("split_code must be 0=train, 1=validation or 2=test")
    if np.any(video_indices < 0) or np.any(video_indices >= authors_by_video.size):
        raise ValueError("catalog video indices are outside catalog_author_ids")

    strengths = positive_event_strengths(labels, target_names)
    author_ids = authors_by_video[video_indices]
    keep = (
        (strengths > 0)
        & (users > 0)
        & (users < int(num_users))
        & (video_indices > 0)
        & (author_ids > 0)
        & (author_ids < int(num_authors))
    )
    count = int(np.count_nonzero(keep))
    return {
        "users": users[keep].astype(np.int32, copy=False),
        "authors": author_ids[keep].astype(np.int32, copy=False),
        "times_ms": times[keep].astype(np.int64, copy=False),
        "weights": strengths[keep].astype(np.float32, copy=False),
        "split_codes": np.full(count, split_code, dtype=np.uint8),
    }


@dataclass(frozen=True)
class CreatorHistoryIndex:
    """CSR-like per-user event index for strict as-of creator-affinity queries."""

    offsets: np.ndarray
    times_ms: np.ndarray
    author_ids: np.ndarray
    weights: np.ndarray
    num_users: int
    num_authors: int

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets, dtype=np.int64)
        times = np.asarray(self.times_ms, dtype=np.int64)
        authors = np.asarray(self.author_ids, dtype=np.int32)
        weights = np.asarray(self.weights, dtype=np.float32)
        if offsets.shape != (int(self.num_users) + 1,):
            raise ValueError("offsets must have shape [num_users + 1]")
        if offsets[0] != 0 or offsets[-1] != times.size or np.any(np.diff(offsets) < 0):
            raise ValueError("offsets must be monotone from 0 to number of events")
        if authors.shape != times.shape or weights.shape != times.shape:
            raise ValueError("times_ms, author_ids and weights must have matching shapes")
        if np.any(authors <= 0) or np.any(authors >= int(self.num_authors)):
            raise ValueError("event author IDs must be known, nonzero author indices")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("event weights must be finite and positive")
        for user in range(int(self.num_users)):
            lo, hi = int(offsets[user]), int(offsets[user + 1])
            if hi - lo > 1 and np.any(np.diff(times[lo:hi]) < 0):
                raise ValueError(f"events for user {user} must be sorted by time")

    @classmethod
    def from_events(
        cls,
        users: np.ndarray,
        authors: np.ndarray,
        times_ms: np.ndarray,
        weights: np.ndarray,
        num_users: int,
        num_authors: int,
    ) -> "CreatorHistoryIndex":
        users = np.asarray(users, dtype=np.int64)
        authors = np.asarray(authors, dtype=np.int32)
        times = np.asarray(times_ms, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float32)
        if users.ndim != 1 or authors.shape != users.shape or times.shape != users.shape or weights.shape != users.shape:
            raise ValueError("event arrays must be matching 1-D arrays")
        if np.any(users <= 0) or np.any(users >= int(num_users)):
            raise ValueError("event user IDs must be known, nonzero user indices")
        if np.any(authors <= 0) or np.any(authors >= int(num_authors)):
            raise ValueError("event author IDs must be known, nonzero author indices")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("event weights must be finite and positive")

        order = np.lexsort((times, users))
        users = users[order]
        times = times[order]
        authors = authors[order]
        weights = weights[order]
        counts = np.bincount(users, minlength=int(num_users))
        offsets = np.empty(int(num_users) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return cls(offsets, times, authors, weights, int(num_users), int(num_authors))

    def scores(
        self,
        user_index: int,
        query_time_ms: int,
        candidate_author_ids: np.ndarray,
        lookback_days: float | None = None,
        half_life_days: float | None = None,
    ) -> np.ndarray:
        """Score candidate creators using only events strictly before query time.

        ``lookback_days`` restricts the history to [query_time - window, query_time).
        ``half_life_days`` optionally exponentially decays weights by event age.
        Both are independent; omit both for lifetime positive-affinity totals.
        """
        candidates = np.asarray(candidate_author_ids, dtype=np.int64)
        if candidates.ndim != 1:
            raise ValueError("candidate_author_ids must be 1-D")
        if np.any(candidates < 0) or np.any(candidates >= self.num_authors):
            raise ValueError("candidate author index outside vocabulary")
        if lookback_days is not None and (not np.isfinite(lookback_days) or lookback_days <= 0):
            raise ValueError("lookback_days must be finite and positive")
        if half_life_days is not None and (not np.isfinite(half_life_days) or half_life_days <= 0):
            raise ValueError("half_life_days must be finite and positive")

        out = np.zeros(candidates.size, dtype=np.float32)
        user = int(user_index)
        if user <= 0 or user >= self.num_users or candidates.size == 0:
            return out

        lo, hi = int(self.offsets[user]), int(self.offsets[user + 1])
        if lo == hi:
            return out
        user_times = self.times_ms[lo:hi]
        query_time = int(query_time_ms)
        end = int(np.searchsorted(user_times, query_time, side="left"))
        if end == 0:
            return out
        start = 0
        if lookback_days is not None:
            cutoff = query_time - int(float(lookback_days) * MS_PER_DAY)
            start = int(np.searchsorted(user_times[:end], cutoff, side="left"))
        if start >= end:
            return out

        authors = self.author_ids[lo + start:lo + end]
        event_times = user_times[start:end]
        event_weights = self.weights[lo + start:lo + end].astype(np.float64, copy=True)
        if half_life_days is not None:
            half_life_ms = float(half_life_days) * MS_PER_DAY
            ages = query_time - event_times
            event_weights *= np.exp2(-ages.astype(np.float64) / half_life_ms)

        unique_authors, inverse = np.unique(authors, return_inverse=True)
        totals = np.bincount(inverse, weights=event_weights)
        positions = np.searchsorted(unique_authors, candidates)
        valid = positions < unique_authors.size
        safe = np.minimum(positions, max(unique_authors.size - 1, 0))
        valid &= unique_authors[safe] == candidates
        valid &= candidates > 0
        out[valid] = totals[safe[valid]].astype(np.float32)
        return out

    def top_authors(
        self,
        user_index: int,
        query_time_ms: int,
        top_k: int,
        lookback_days: float | None = None,
        half_life_days: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return known creators sorted by score descending, then author ID ascending."""
        if int(top_k) < 1:
            raise ValueError("top_k must be positive")
        user = int(user_index)
        if user <= 0 or user >= self.num_users:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        lo, hi = int(self.offsets[user]), int(self.offsets[user + 1])
        if lo == hi:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        user_times = self.times_ms[lo:hi]
        end = int(np.searchsorted(user_times, int(query_time_ms), side="left"))
        if end == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        start = 0
        if lookback_days is not None:
            if not np.isfinite(lookback_days) or lookback_days <= 0:
                raise ValueError("lookback_days must be finite and positive")
            cutoff = int(query_time_ms) - int(float(lookback_days) * MS_PER_DAY)
            start = int(np.searchsorted(user_times[:end], cutoff, side="left"))
        authors = self.author_ids[lo + start:lo + end]
        weights = self.weights[lo + start:lo + end].astype(np.float64, copy=True)
        if half_life_days is not None:
            if not np.isfinite(half_life_days) or half_life_days <= 0:
                raise ValueError("half_life_days must be finite and positive")
            ages = int(query_time_ms) - user_times[start:end]
            weights *= np.exp2(-ages.astype(np.float64) / (float(half_life_days) * MS_PER_DAY))
        unique, inverse = np.unique(authors, return_inverse=True)
        scores = np.bincount(inverse, weights=weights).astype(np.float32)
        order = np.lexsort((unique, -scores))[:int(top_k)]
        return unique[order].astype(np.int32), scores[order]
