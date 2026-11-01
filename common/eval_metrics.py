"""Pure-NumPy evaluation metrics shared by the diagnostic and evaluation scripts.

Nothing here imports torch, so the maths can be unit-tested on any machine.

Conventions
-----------
* "gain" is a non-negative relevance score per impression, built from the event
  labels (see ``relevance_gains``). It uses the same event weights as the
  retrieval positives, so ranking and retrieval are judged on one definition.
* Confidence intervals are percentile bootstraps over independent units
  (queries, users or groups), never over raw rows that share a user.
"""
from __future__ import annotations

import numpy as np

# Same positive-event weights as model1_retrieval.data.target_indices.
RELEVANCE_WEIGHTS = {
    "is_click": 0.25,
    "is_like": 1.0,
    "is_follow": 1.5,
    "is_comment": 1.25,
    "is_forward": 1.5,
    "long_view": 2.0,
    "is_profile_enter": 0.75,
    "complete_view": 2.0,
}
SHORT_VIEW_PENALTY = 0.5


def relevance_gains(labels: np.ndarray, target_names: list[str]) -> np.ndarray:
    """Graded relevance per row: weighted positive events, minus a short-view penalty.

    Rows with an explicit ``is_hate`` get gain 0. Gains are clipped at 0 so NDCG is defined.
    """
    index = {name: i for i, name in enumerate(target_names)}
    gain = np.zeros(labels.shape[0], dtype=np.float32)
    for name, weight in RELEVANCE_WEIGHTS.items():
        gain += labels[:, index[name]] * np.float32(weight)
    gain -= labels[:, index["short_view"]] * np.float32(SHORT_VIEW_PENALTY)
    gain[labels[:, index["is_hate"]] > 0] = 0.0
    return np.maximum(gain, 0.0)


def action_only_relevance_gains(labels: np.ndarray, target_names: list[str]) -> np.ndarray:
    """Duration-independent gains from explicit engagement actions only.

    Excludes long_view, complete_view, short_view and watch time.
    Explicit hate feedback suppresses gain, matching relevance_gains().
    """
    index = {name: i for i, name in enumerate(target_names)}
    action_weights = {
        "is_click": 0.25, "is_like": 1.0, "is_follow": 1.5,
        "is_comment": 1.25, "is_forward": 1.5,
    }
    gain = np.zeros(labels.shape[0], dtype=np.float32)
    for name, weight in action_weights.items():
        if name in index:
            gain += labels[:, index[name]] * np.float32(weight)
    if "is_hate" in index:
        gain[labels[:, index["is_hate"]] > 0] = 0.0
    return np.maximum(gain, 0.0)


# --------------------------------------------------------------------------- bootstrap
def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap CI of a per-unit array (one value per query/group/user)."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    # Chunk the resampling so a large n does not need an n_boot x n index matrix at once.
    step = max(1, int(4_000_000 // max(n, 1)))
    for start in range(0, n_boot, step):
        stop = min(start + step, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def cluster_bootstrap_mean_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 1000,
                               seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Query-weighted mean and percentile CI with clusters (for example users) resampled.

    Every sampled cluster contributes all its rows. This preserves within-user dependence
    while retaining the original query-weighted estimand.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if values.ndim != 1 or groups.ndim != 1 or values.shape != groups.shape:
        raise ValueError("values and groups must be matching 1-D arrays")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=unique.size)
    counts = np.bincount(inverse, minlength=unique.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        draw = rng.integers(0, unique.size, size=unique.size)
        samples[i] = sums[draw].sum() / counts[draw].sum()
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0,
                   alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of (a - b) with a bootstrap CI; a and b must be aligned per unit."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired_diff_ci needs arrays of equal shape")
    return bootstrap_mean_ci(a - b, n_boot=n_boot, seed=seed, alpha=alpha)


# --------------------------------------------------------------------------- retrieval
def recall_hits(ranks: np.ndarray, k: int) -> np.ndarray:
    """1.0 where the positive's 1-based rank is within the top k, else 0.0."""
    return (np.asarray(ranks) <= k).astype(np.float64)


def reciprocal_ranks(ranks: np.ndarray) -> np.ndarray:
    return 1.0 / np.asarray(ranks, dtype=np.float64)


def expected_rank_from_scores(sorted_ascending: np.ndarray, item_scores: np.ndarray) -> np.ndarray:
    """1-based expected rank of items among a catalogue, with ties split evenly.

    ``sorted_ascending`` holds every catalogue item's score, sorted ascending.
    Ties matter for popularity baselines (millions of items share a count of zero).
    """
    right = np.searchsorted(sorted_ascending, item_scores, side="right")
    left = np.searchsorted(sorted_ascending, item_scores, side="left")
    greater = sorted_ascending.size - right
    equal = right - left
    return 1.0 + greater + 0.5 * np.maximum(equal - 1, 0)


# --------------------------------------------------------------------------- AUC family
def auc_binary(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC with average ranks for ties; NaN if only one class is present."""
    y = np.asarray(y) > 0
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    _, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    average_rank = (cumulative - counts + 1 + cumulative) / 2.0
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = average_rank[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as average precision; NaN when there are no positives."""
    y = np.asarray(y) > 0
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    hits = y[order]
    cumulative_hits = np.cumsum(hits)
    precision_at_hit = cumulative_hits[hits] / (np.flatnonzero(hits) + 1)
    return float(precision_at_hit.mean())


def per_group_auc(y: np.ndarray, score: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUC inside each group (for example each user).

    Returns (auc_values, weights) for groups that contain both classes; the weight is the
    group's row count, the usual GAUC weighting.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    group = np.asarray(group)
    order = np.argsort(group, kind="stable")
    g_sorted = group[order]
    boundaries = np.flatnonzero(np.diff(g_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [g_sorted.size]])
    aucs, weights = [], []
    for a, b in zip(starts, stops):
        rows = order[a:b]
        value = auc_binary(y[rows], score[rows])
        if np.isfinite(value):
            aucs.append(value)
            weights.append(b - a)
    return np.asarray(aucs, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def weighted_bootstrap_ci(values: np.ndarray, weights: np.ndarray, n_boot: int = 1000,
                          seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Weighted mean with a bootstrap over units (users)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = values.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(np.average(values, weights=weights))
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = np.average(values[idx], weights=weights[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- calibration
def calibration_summary(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Mean predicted vs observed rate plus expected calibration error (equal-count bins).

    Equal-count bins are used because most targets are rare, and equal-width bins would
    put almost every row in the first bin.
    """
    prob = np.asarray(prob, dtype=np.float64)
    y = (np.asarray(y) > 0).astype(np.float64)
    order = np.argsort(prob, kind="mergesort")
    chunks = np.array_split(order, bins)
    ece = 0.0
    for rows in chunks:
        if rows.size:
            ece += rows.size / prob.size * abs(prob[rows].mean() - y[rows].mean())
    base = float(y.mean())
    mean_pred = float(prob.mean())
    return {
        "base_rate": base,
        "mean_predicted": mean_pred,
        "prediction_over_base": mean_pred / base if base > 0 else float("nan"),
        "ece": float(ece),
    }


# --------------------------------------------------------------------------- grouped NDCG
def make_blocks(user: np.ndarray, block_size: int) -> np.ndarray:
    """Group id per row: consecutive impressions of one user, ``block_size`` at a time.

    Rows must already be in time order (the prepared splits are). Each block approximates
    one browsing session, and ranking inside it asks: "of what this user was shown,
    did the model put the things they engaged with first?"
    """
    user = np.asarray(user, dtype=np.int64)
    order = np.argsort(user, kind="stable")
    sorted_user = user[order]
    first = np.concatenate([[True], sorted_user[1:] != sorted_user[:-1]])
    start_of_user = np.maximum.accumulate(np.where(first, np.arange(user.size), 0))
    position = np.arange(user.size) - start_of_user
    block = position // int(block_size)
    group_sorted = sorted_user * 1_000_000 + block
    group = np.empty(user.size, dtype=np.int64)
    group[order] = group_sorted
    return group


def grouped_ndcg(score: np.ndarray, gain: np.ndarray, group: np.ndarray, ks: tuple[int, ...],
                 seed: int = 0) -> dict[int, np.ndarray]:
    """NDCG@k per group (only groups with some relevance); ties broken at random.

    Random tie-breaking keeps constant or heavily tied scores (a popularity baseline)
    from getting an accidental ordering advantage.
    """
    score = np.asarray(score, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    group = np.asarray(group)
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(score.size)

    def position_in_group(sort_keys_primary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.lexsort((tiebreak, -sort_keys_primary, group))
        g_sorted = group[order]
        first = np.concatenate([[True], g_sorted[1:] != g_sorted[:-1]])
        start = np.maximum.accumulate(np.where(first, np.arange(order.size), 0))
        return order, g_sorted, np.arange(order.size) - start

    order, g_sorted, pos = position_in_group(score)
    ideal_order, _, ideal_pos = position_in_group(gain)
    unique_groups, group_index = np.unique(g_sorted, return_inverse=True)
    discount = 1.0 / np.log2(pos + 2.0)
    ideal_discount = 1.0 / np.log2(ideal_pos + 2.0)
    results: dict[int, np.ndarray] = {}
    for k in ks:
        dcg = np.bincount(group_index, weights=np.where(pos < k, gain[order] * discount, 0.0),
                          minlength=unique_groups.size)
        # ideal_order lists the same groups in the same sorted-group order as `order`.
        idcg = np.bincount(group_index, weights=np.where(ideal_pos < k, gain[ideal_order] * ideal_discount, 0.0),
                           minlength=unique_groups.size)
        valid = idcg > 0
        results[k] = dcg[valid] / idcg[valid]
    return results
