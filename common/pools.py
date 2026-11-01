"""Time-window item pools, time-local batches and known-positive lookups (pure NumPy).

Why this module exists
----------------------
Short-video feeds are served from an *active pool*, not from every video that ever
existed. In the KuaiRand logs most held-out positives are on videos that are new in the
validation period, so (a) training negatives should come from the same time window as the
positive, and (b) evaluation should rank a positive against the items that were active in
its window. These helpers build those windows and the batches/masks that use them.

Nothing here imports torch, so all of it is unit-tested without a GPU stack.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HOUR_MS = 3_600_000


def window_ids(time_ms: np.ndarray, origin_ms: int, window_hours: float) -> np.ndarray:
    """Integer window number of every timestamp, counted from ``origin_ms``."""
    window_ms = int(round(float(window_hours) * HOUR_MS))
    if window_ms <= 0:
        raise ValueError("window_hours must be positive")
    shifted = np.asarray(time_ms, dtype=np.int64) - int(origin_ms)
    return np.floor_divide(shifted, window_ms).astype(np.int64)


class WindowPools:
    """Unique items exposed in each time window, stored as CSR arrays."""

    def __init__(self, first_window: int, ptr: np.ndarray, items: np.ndarray):
        self.first_window = int(first_window)
        self.ptr = np.asarray(ptr, dtype=np.int64)
        self.items = np.asarray(items, dtype=np.int32)

    @classmethod
    def build(cls, windows: np.ndarray, items: np.ndarray) -> "WindowPools":
        windows = np.asarray(windows, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        keep = items > 0
        windows, items = windows[keep], items[keep]
        if windows.size == 0:
            return cls(0, np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32))
        first, last = int(windows.min()), int(windows.max())
        base = int(items.max()) + 1
        keys = np.unique((windows - first) * base + items)
        window_of_key = keys // base
        counts = np.bincount(window_of_key, minlength=last - first + 1)
        ptr = np.concatenate([[0], np.cumsum(counts)])
        return cls(first, ptr, (keys % base).astype(np.int32))

    @property
    def n_windows(self) -> int:
        return int(self.ptr.size - 1)

    def get(self, window: int) -> np.ndarray:
        local = int(window) - self.first_window
        if local < 0 or local >= self.n_windows:
            return self.items[:0]
        return self.items[self.ptr[local]:self.ptr[local + 1]]

    def size(self, window: int) -> int:
        return int(self.get(window).size)

    def sample(self, window: int, count: int, rng: np.random.Generator) -> np.ndarray:
        """``count`` items drawn uniformly (with replacement) from one window's pool."""
        pool = self.get(window)
        if pool.size == 0 or count <= 0:
            return self.items[:0]
        return pool[rng.integers(0, pool.size, size=int(count))]

    def nearest_nonempty(self, window: int) -> int | None:
        """The window itself if it has items, otherwise the closest window that does."""
        if self.size(window) > 0:
            return int(window)
        sizes = np.diff(self.ptr)
        filled = np.flatnonzero(sizes > 0)
        if filled.size == 0:
            return None
        local = int(window) - self.first_window
        return int(self.first_window + filled[np.argmin(np.abs(filled - local))])


def time_local_batches(rows: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Cut time-ordered rows into batches of consecutive rows, then shuffle the batch order.

    A batch therefore holds interactions from the same short stretch of time (about 45
    minutes for 4096 rows of this dataset), so the other users' positives inside it are
    realistic negatives: items that were active at that moment. A random start offset
    changes the cut points every epoch; at most ``batch_size - 1`` rows are skipped.
    """
    rows = np.asarray(rows)
    if rows.size == 0:
        return []
    if rows.size < batch_size:
        return [rows]
    n_batches = rows.size // batch_size
    slack = rows.size - n_batches * batch_size
    offset = int(rng.integers(0, slack + 1))
    usable = rows[offset:offset + n_batches * batch_size].reshape(n_batches, batch_size)
    return [usable[i] for i in rng.permutation(n_batches)]


class KnownPairs:
    """Bit matrix answering 'is (user, item) a known positive?' for whole grids at once.

    Memory is ``n_users * n_items / 8`` bytes (about 0.55 GB for 1,001 users x 4.37M items).
    """

    def __init__(self, users: np.ndarray, items: np.ndarray, n_users: int, n_items: int):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.bits = np.zeros((self.n_users, (self.n_items + 7) // 8), dtype=np.uint8)
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        if users.size:
            if users.min() < 0 or users.max() >= self.n_users or items.min() < 0 or items.max() >= self.n_items:
                raise ValueError("user or item index outside the declared KnownPairs range")
            keys = np.unique(users * self.n_items + items)
            u, i = keys // self.n_items, keys % self.n_items
            np.bitwise_or.at(self.bits, (u, i >> 3), (1 << (i & 7)).astype(np.uint8))

    def grid(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Boolean matrix [len(users), len(items)]: True where the pair is a known positive."""
        users = np.asarray(users, dtype=np.int64)[:, None]
        items = np.asarray(items, dtype=np.int64)[None, :]
        byte = self.bits[users, items >> 3]
        return ((byte >> (items & 7).astype(np.uint8)) & 1).astype(bool)

    def count(self) -> int:
        return int(np.unpackbits(self.bits, axis=None).sum())


def positive_frequency(items: np.ndarray, n_items: int) -> np.ndarray:
    """Share of all positive rows that each item accounts for (index 0 unused)."""
    counts = np.bincount(np.asarray(items, dtype=np.int64), minlength=int(n_items)).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def build_softmax_batch(users: np.ndarray, positives: np.ndarray, shared_negatives: np.ndarray,
                        known: KnownPairs, frequency: np.ndarray, pool_size: int,
                        logq_strength: float = 1.0):
    """Candidate set, own-positive columns, exclusion mask and logQ correction for one batch.

    Every row's own positive is the softmax target. All the user's *other* known positives
    among the candidates are excluded from the denominator (they are not negatives).
    ``logq`` is the log of the expected number of times each candidate appears in the batch
    (as somebody's positive, or among the shared negatives); subtracting it from the logits
    stops popular items from being penalised merely for being sampled often.
    """
    users = np.asarray(users, dtype=np.int64)
    positives = np.asarray(positives, dtype=np.int64)
    shared_negatives = np.asarray(shared_negatives, dtype=np.int64)
    if users.shape != positives.shape or users.ndim != 1 or users.size == 0:
        raise ValueError("users and positives must be matching non-empty 1-D arrays")
    candidates = np.unique(np.concatenate([positives, shared_negatives]))
    own_cols = np.searchsorted(candidates, positives)
    unique_users, inverse = np.unique(users, return_inverse=True)
    exclude = known.grid(unique_users, candidates)[inverse]
    exclude[np.arange(users.size), own_cols] = False
    expected = users.size * frequency[candidates] + shared_negatives.size / max(int(pool_size), 1)
    logq = (float(logq_strength) * np.log(np.maximum(expected, 1e-12))).astype(np.float32)
    return candidates.astype(np.int32), own_cols.astype(np.int64), exclude, logq


def expected_ranks_in_pool(scores: np.ndarray, target_scores: np.ndarray, valid: np.ndarray,
                           rtol: float = 1e-6, atol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Tie-aware 1-based rank of each target among the valid competitors in its row.

    ``scores`` is [Q, P]; ``valid`` marks pool items that may compete (the target itself and
    known positives of the user should be False). Returns (ranks, competitor_counts).
    """
    thresholds = np.asarray(target_scores, dtype=np.float64)[:, None]
    tolerance = atol + rtol * np.abs(thresholds)
    greater = ((scores > thresholds + tolerance) & valid).sum(axis=1)
    equal = ((np.abs(scores - thresholds) <= tolerance) & valid).sum(axis=1)
    ranks = 1.0 + greater.astype(np.float64) + 0.5 * equal.astype(np.float64)
    return ranks, valid.sum(axis=1).astype(np.int64)


def percentile_from_ranks(ranks: np.ndarray, competitors: np.ndarray) -> np.ndarray:
    """0 = ranked first, 1 = ranked last, 0.5 = no better than a coin flip."""
    return (np.asarray(ranks, dtype=np.float64) - 1.0) / np.maximum(np.asarray(competitors, dtype=np.float64), 1.0)


def reference_pool_softmax_loss(user_vec: np.ndarray, cand_vec: np.ndarray, own_cols: np.ndarray,
                                exclude: np.ndarray, logq: np.ndarray, temperature: float,
                                weights: np.ndarray) -> tuple[float, np.ndarray]:
    """NumPy reference for the trainer's loss (used by tests to check the torch version).

    per_row = logsumexp(non-excluded corrected logits) - corrected logit of the row's own positive.
    """
    logits = (user_vec @ cand_vec.T) / float(temperature) - logq[None, :]
    rows = np.arange(user_vec.shape[0])
    own = logits[rows, own_cols]
    masked = np.where(exclude, -np.inf, logits)
    peak = masked.max(axis=1, keepdims=True)
    log_denominator = peak[:, 0] + np.log(np.exp(masked - peak).sum(axis=1))
    per_row = log_denominator - own
    return float((per_row * weights).sum() / max(float(weights.sum()), 1e-6)), per_row


def load_row_times(data_dir: Path | str, split_name: str) -> np.ndarray:
    """Per-row timestamps (ms) for 'train', 'validation' or 'test' from row_times.npz."""
    path = Path(data_dir) / "row_times.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Create it once with: python -m scripts.add_row_times "
            f"--raw-dir <folder with the KuaiRand CSVs> --data-dir {data_dir}")
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload[split_name], dtype=np.int64)
