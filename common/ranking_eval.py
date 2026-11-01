"""Compute the full ranking-evaluation report from predictions (pure NumPy, no torch)."""
from __future__ import annotations

import numpy as np

from common.eval_metrics import (
    action_only_relevance_gains, auc_binary, average_precision, bootstrap_mean_ci, calibration_summary, grouped_ndcg,
    make_blocks, paired_diff_ci, per_group_auc, relevance_gains, weighted_bootstrap_ci,
)

REFERENCE_SCORE = "hand_weighted_score (current)"


def evaluate_ranking_predictions(
    probs: np.ndarray, watch_pred: np.ndarray, labels: np.ndarray, watch_true: np.ndarray,
    user: np.ndarray, video: np.ndarray, targets: list[str], score_weights: dict[str, float],
    popularity_by_video: np.ndarray | None = None, block_size: int = 30,
    ks: tuple[int, ...] = (10, 20), n_boot: int = 500, seed: int = 7,
    duration_score: np.ndarray | None = None,
) -> dict:
    """Per-target AUC/GAUC/PR-AUC/calibration, watch-time fit and NDCG vs baselines."""
    report: dict = {}

    per_target = {}
    gauc_values = []
    for i, name in enumerate(targets):
        y, p = labels[:, i], probs[:, i]
        aucs, weights = per_group_auc(y, p, user)
        gauc, lo, hi = weighted_bootstrap_ci(aucs, weights, n_boot=n_boot, seed=seed)
        per_target[name] = {
            "auc": auc_binary(y, p), "pr_auc": average_precision(y, p), "gauc": gauc,
            "gauc_ci95": [lo, hi], "users_in_gauc": int(aucs.size), **calibration_summary(p, y),
        }
        if np.isfinite(gauc):
            gauc_values.append(gauc)
    report["per_target"] = per_target
    report["mean_auc"] = float(np.nanmean([v["auc"] for v in per_target.values()]))
    report["mean_gauc"] = float(np.mean(gauc_values)) if gauc_values else float("nan")

    error = watch_pred - watch_true
    sst = float(np.sum((watch_true - watch_true.mean()) ** 2))
    report["watch_time_log_seconds"] = {
        "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / sst) if sst > 0 else float("nan"),
    }

    gain = relevance_gains(labels, targets)
    group = make_blocks(user, block_size)
    _, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= block_size  # full blocks only; short tail blocks are noisy
    rng = np.random.default_rng(seed)

    hand = np.zeros(probs.shape[0], dtype=np.float32)
    for name, weight in score_weights.items():
        if name in targets:
            hand += np.float32(weight) * probs[:, targets.index(name)]
    scores: dict[str, np.ndarray] = {REFERENCE_SCORE: hand}
    for name in targets:
        index = targets.index(name)
        if name in ("is_hate", "short_view"):
            scores[f"-p_{name}"] = -probs[:, index]
        else:
            scores[f"p_{name}"] = probs[:, index]
    scores["predicted_watch_time"] = watch_pred
    scores["random_order"] = rng.random(probs.shape[0]).astype(np.float32)
    if duration_score is not None:
        duration_score = np.asarray(duration_score, dtype=np.float32)
        if duration_score.shape != (probs.shape[0],):
            raise ValueError("duration_score must have one value per row")
        scores["shortest_video_first"] = duration_score
    if popularity_by_video is not None:
        scores["popularity (train engagement)"] = popularity_by_video[video].astype(np.float32)

    gain_definitions = {
        "engagement_and_view": gain,
        "action_only_duration_independent": action_only_relevance_gains(labels, targets),
    }
    all_gain_reports = {}
    for gain_name, selected_gain in gain_definitions.items():
        per_k_by_name = {
            name: grouped_ndcg(s[keep], selected_gain[keep], group[keep], ks, seed=seed)
            for name, s in scores.items()
        }
        ndcg_report = {}
        for name, per_k in per_k_by_name.items():
            entry = {}
            for k in ks:
                mean, lo, hi = bootstrap_mean_ci(per_k[k], n_boot=n_boot, seed=seed)
                row = {"ndcg": mean, "ci95": [lo, hi], "groups": int(per_k[k].size)}
                if name != REFERENCE_SCORE:
                    d, dlo, dhi = paired_diff_ci(
                        per_k[k], per_k_by_name[REFERENCE_SCORE][k], n_boot=n_boot, seed=seed
                    )
                    row["diff_vs_hand_weighted"] = {"mean": d, "ci95": [dlo, dhi]}
                entry[f"@{k}"] = row
            ndcg_report[name] = entry
        all_gain_reports[gain_name] = ndcg_report
    report["ndcg_reranking_logged_impressions"] = {
        "block_size": block_size, "blocks_used": int((counts >= block_size).sum()),
        "gain_definitions": all_gain_reports,
        # Backward-compatible alias for the historical composite engagement+view gain.
        "scores": all_gain_reports["engagement_and_view"],
    }
    return report
