"""Torch-free helpers for comparing Model 2 score formulas on identical candidate slates.

The retrieval positive target is defined in ``model1_retrieval.data.positive_rows``:
weighted positive events minus the short-view penalty, with hate examples excluded
as a hard rule. Therefore the retrieval-weighted linear score below does not invent
an additive hate weight; the hate head is evaluated separately as a negative signal.

Percentile is conditional on the held-out target being present in the candidate slate:
0 is best, 0.5 is random expectation, and 1 is worst. Confidence intervals resample
users as clusters while retaining a query-weighted point estimate.
"""
from __future__ import annotations

import numpy as np

from common.eval_metrics import (
    RELEVANCE_WEIGHTS,
    SHORT_VIEW_PENALTY,
    cluster_bootstrap_mean_ci,
)

NEGATIVE_HEADS = ("short_view", "is_hate")


def retrieval_target_weights() -> dict[str, float]:
    """Return the exact linear event weights used by the retrieval positive target.

    Hate is intentionally absent: ``positive_rows`` excludes hate rows rather than
    subtracting an additive ``is_hate`` weight.
    """
    weights = dict(RELEVANCE_WEIGHTS)
    weights["short_view"] = -float(SHORT_VIEW_PENALTY)
    return weights


def build_score_variants(
    targets: list[str],
    current_weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Current formula, retrieval-target-weighted formula, and each individual head."""
    variants: dict[str, dict[str, float]] = {
        "current_weights": dict(current_weights),
        "retrieval_target_weighted_sum": retrieval_target_weights(),
    }
    for name in targets:
        variants[f"head:{name}"] = {
            name: -1.0 if name in NEGATIVE_HEADS else 1.0
        }

    combinations = {
        "like_follow_equal_sum": ("is_like", "is_follow"),
        "like_comment_equal_sum": ("is_like", "is_comment"),
        "like_follow_comment_equal_sum": ("is_like", "is_follow", "is_comment"),
    }
    available = set(targets)
    for variant_name, components in combinations.items():
        missing = set(components) - available
        if missing:
            raise ValueError(
                f"Cannot build {variant_name}; missing targets: {sorted(missing)}"
            )
        variants[variant_name] = {name: 1.0 for name in components}

    return variants


def variant_score_matrix(
    probs: np.ndarray,
    targets: list[str],
    variants: dict[str, dict[str, float]],
) -> tuple[list[str], np.ndarray]:
    """Return variant names and a [candidate, variant] matrix of weighted probabilities."""
    probs = np.asarray(probs, dtype=np.float32)
    names = list(variants)

    if probs.ndim != 2 or probs.shape[1] != len(targets):
        if probs.size == 0:
            return names, np.zeros((0, len(names)), dtype=np.float32)
        raise ValueError(
            f"Expected probabilities with shape [n, {len(targets)}], got {probs.shape}"
        )

    weights = np.zeros((len(targets), len(names)), dtype=np.float32)
    target_index = {name: i for i, name in enumerate(targets)}
    for j, variant_name in enumerate(names):
        for target, weight in variants[variant_name].items():
            if target in target_index:
                weights[target_index[target], j] = float(weight)

    return names, probs @ weights


def percentile_of_rank(ranks: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(expected_rank - 1) / (candidate_count - 1); NaN if uncovered or slate size <= 1."""
    ranks = np.asarray(ranks, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if ranks.ndim != 1 or counts.ndim != 1 or ranks.shape != counts.shape:
        raise ValueError("ranks and counts must be matching 1-D arrays")

    out = np.full(ranks.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ranks) & (counts > 1) & (ranks >= 1) & (ranks <= counts)
    out[valid] = (ranks[valid] - 1.0) / (counts[valid] - 1.0)
    return out


def summarize_percentile(
    ranks: np.ndarray,
    counts: np.ndarray,
    users: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Conditional mean target percentile and user-cluster CI.

    Random ordering has expectation 0.5. A negative delta means better-than-random
    target ordering. Uncovered queries are excluded from this *conditional* metric;
    candidate coverage must always be read alongside it.
    """
    percentile = percentile_of_rank(ranks, counts)
    users = np.asarray(users)
    if users.ndim != 1 or users.shape != percentile.shape:
        raise ValueError("users must match the query/rank array shape")

    covered = np.isfinite(percentile)
    result = {
        "queries_scored": int(covered.sum()),
        "users_scored": int(np.unique(users[covered]).size),
        "coverage": float(covered.mean()) if covered.size else 0.0,
    }
    if not covered.any():
        result.update({
            "mean_percentile": None,
            "ci95_user_cluster": [None, None],
            "delta_vs_random": None,
            "delta_ci95_user_cluster": [None, None],
        })
        return result

    mean, lo, hi = cluster_bootstrap_mean_ci(
        percentile[covered],
        users[covered],
        n_boot=n_boot,
        seed=seed,
    )
    result.update({
        "mean_percentile": mean,
        "ci95_user_cluster": [lo, hi],
        "delta_vs_random": mean - 0.5,
        "delta_ci95_user_cluster": [lo - 0.5, hi - 0.5],
    })
    return result


def summarize_paired_variant(
    candidate_ranks: np.ndarray,
    baseline_ranks: np.ndarray,
    counts: np.ndarray,
    users: np.ndarray,
    n_boot: int,
    seed: int,
    hit_k: int = 100,
) -> dict:
    """Compare one score formula with a baseline on identical query slates.

    Percentile differences are conditional on both formulas having a covered
    target. Hit@K differences are unconditional over all evaluated queries.
    Positive percentile delta means the candidate formula is worse; positive
    Hit@K delta means it is better.
    """
    candidate_ranks = np.asarray(candidate_ranks, dtype=np.float64)
    baseline_ranks = np.asarray(baseline_ranks, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    users = np.asarray(users)

    if not (
        candidate_ranks.shape == baseline_ranks.shape
        == counts.shape == users.shape
    ):
        raise ValueError("Ranks, candidate counts and users must have matching shapes")

    candidate_pct = percentile_of_rank(candidate_ranks, counts)
    baseline_pct = percentile_of_rank(baseline_ranks, counts)
    paired = np.isfinite(candidate_pct) & np.isfinite(baseline_pct)

    if paired.any():
        pct_mean, pct_lo, pct_hi = cluster_bootstrap_mean_ci(
            candidate_pct[paired] - baseline_pct[paired],
            users[paired],
            n_boot=n_boot,
            seed=seed,
        )
        percentile_delta = {
            "queries_compared": int(paired.sum()),
            "users_compared": int(np.unique(users[paired]).size),
            "mean_delta": pct_mean,
            "ci95_user_cluster": [pct_lo, pct_hi],
        }
    else:
        percentile_delta = {
            "queries_compared": 0,
            "users_compared": 0,
            "mean_delta": None,
            "ci95_user_cluster": [None, None],
        }

    candidate_hit = (
        np.isfinite(candidate_ranks) & (candidate_ranks <= hit_k)
    ).astype(np.float64)
    baseline_hit = (
        np.isfinite(baseline_ranks) & (baseline_ranks <= hit_k)
    ).astype(np.float64)

    hit_mean, hit_lo, hit_hi = cluster_bootstrap_mean_ci(
        candidate_hit - baseline_hit,
        users,
        n_boot=n_boot,
        seed=seed + 1,
    )

    return {
        "baseline": "head:is_like",
        "hit_k": int(hit_k),
        "percentile_delta_variant_minus_baseline": percentile_delta,
        "hit_delta_variant_minus_baseline": {
            "queries_compared": int(users.size),
            "users_compared": int(np.unique(users).size),
            "mean_delta": hit_mean,
            "ci95_user_cluster": [hit_lo, hit_hi],
        },
    }
