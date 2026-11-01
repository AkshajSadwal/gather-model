"""Evaluate the existing Model 2 ranker on V3 + author-affinity candidate unions.

This is an offline reranking experiment. It does not train or overwrite either model.
Candidate pools are built from the retrieval validation split; Model 2 scores candidates
using its own prepared ranking feature maps and the query's event timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.data_version import schema_fingerprint
from common.eval_metrics import cluster_bootstrap_mean_ci
from common.ranker_variants import build_score_variants, summarize_percentile, summarize_paired_variant, variant_score_matrix
from common.pools import KnownPairs, WindowPools, load_row_times, window_ids
from model1_retrieval.data import (
    build_video_catalog, data_fingerprint, item_catalog_indices, load_metadata as load_retrieval_metadata,
    load_split as load_retrieval_split, positive_rows,
)
from model1_retrieval.model import TwoTowerRetrieval
from model2_ranking.model_v2 import ReelsRankerV2
from scripts.evaluate_retrieval_pools import AuthorAffinity, select_topk_columns

AUTHOR_FIELD_INDEX = 29
SCORE_WEIGHTS = {
    "is_click": 0.15, "is_like": 0.60, "is_follow": 0.75, "is_comment": 0.65,
    "is_forward": 1.00, "long_view": 1.00, "is_profile_enter": 0.25,
    "complete_view": 0.40, "short_view": -0.90, "is_hate": -2.00,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_index(sorted_values: np.ndarray, value: int) -> int:
    idx = int(np.searchsorted(sorted_values, value))
    return idx + 1 if idx < len(sorted_values) and int(sorted_values[idx]) == int(value) else 0


def expected_rank(scores: np.ndarray, target_pos: int) -> float:
    """One-based average rank with ties handled neutrally."""
    target = float(scores[target_pos])
    greater = int(np.count_nonzero(scores > target))
    equal_other = int(np.count_nonzero(scores == target)) - 1
    return 1.0 + greater + 0.5 * max(equal_other, 0)


def pool_columns_to_raw_ids(
    pool_catalog_indices: np.ndarray,
    candidate_columns: np.ndarray,
    catalog_raw_ids: np.ndarray,
) -> np.ndarray:
    """Translate candidate columns (positions in pool) to raw video IDs.

    `candidate_columns` index the active `pool`; values in `pool` are 1-based
    catalogue indices; `catalog_raw_ids` is 0-based over catalogue entries.
    """
    pool_catalog_indices = np.asarray(pool_catalog_indices, dtype=np.int64)
    candidate_columns = np.asarray(candidate_columns, dtype=np.int64)
    catalog_raw_ids = np.asarray(catalog_raw_ids, dtype=np.int64)
    if candidate_columns.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(candidate_columns < 0) or np.any(candidate_columns >= pool_catalog_indices.size):
        raise IndexError("Candidate column is outside the active pool")
    catalogue_indices = pool_catalog_indices[candidate_columns]
    if np.any(catalogue_indices < 1) or np.any(catalogue_indices > catalog_raw_ids.size):
        raise IndexError("Pool contains a catalogue index outside the raw-ID map")
    return catalog_raw_ids[catalogue_indices - 1]


def build_ranker_features(
    raw_video_ids: np.ndarray,
    query_time_ms: int,
    user_index: int,
    tab_index: int,
    user_cat: np.ndarray,
    user_num: np.ndarray,
    catalog_arrays: dict[str, np.ndarray],
    rank_metadata: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exactly the feature layout used by model2_ranking.recommend_v2."""
    video_ids = catalog_arrays["video_ids"]
    positions = np.searchsorted(video_ids, raw_video_ids)
    valid = positions < len(video_ids)
    matched = np.zeros(len(positions), dtype=bool)
    matched[valid] = video_ids[positions[valid]] == raw_video_ids[valid]
    if not np.any(matched):
        return (np.empty((0, 3 + user_cat.size + catalog_arrays["video_cat"].shape[1]), dtype=np.int64),
                np.empty((0, int(rank_metadata["num_numeric_features"])), dtype=np.float32),
                np.flatnonzero(matched))
    selected = np.flatnonzero(matched)
    cat_idx = positions[selected] + 1
    model_video_idx = catalog_arrays["video_model_indices"][cat_idx]
    video_cat = catalog_arrays["video_cat"][cat_idx]
    categorical = np.column_stack([
        np.full(selected.size, user_index, dtype=np.int32),
        model_video_idx,
        np.full(selected.size, tab_index, dtype=np.int32),
        np.repeat(user_cat.reshape(1, -1), selected.size, axis=0),
        video_cat,
    ]).astype(np.int64, copy=False)

    duration = catalog_arrays["duration"][cat_idx].copy()
    fallback_ms = float(rank_metadata.get("video_duration_fallback_ms", 0.0))
    duration = np.where(duration > 0, duration, fallback_ms / 1000.0)
    upload = catalog_arrays["upload_ms"][cat_idx]
    age_hours = np.where(upload > 0, np.maximum((query_time_ms - upload) / 3_600_000.0, 0.0), 0.0)
    hour = (int(query_time_ms) // 3_600_000) % 24
    angle = 2.0 * np.pi * hour / 24.0
    numeric_raw = np.column_stack([
        np.log1p(np.maximum(duration, 0.0)),
        np.full(selected.size, np.sin(angle)),
        np.full(selected.size, np.cos(angle)),
        np.repeat(user_num.reshape(1, -1), selected.size, axis=0),
        np.log1p(age_hours),
        np.log1p(np.maximum(catalog_arrays["width"][cat_idx], 0.0)),
        np.log1p(np.maximum(catalog_arrays["height"][cat_idx], 0.0)),
    ]).astype(np.float32, copy=False)
    means = np.asarray(rank_metadata["numeric_mean"], dtype=np.float32)
    stds = np.asarray(rank_metadata["numeric_std"], dtype=np.float32)
    stds = np.where(stds > 1e-6, stds, 1.0)
    numeric = ((numeric_raw - means) / stds).astype(np.float32, copy=False)
    return categorical, numeric, selected


@torch.no_grad()
def ranker_scores(model, categorical, numeric, targets, device, batch_size):
    model.eval()
    output = np.empty(categorical.shape[0], dtype=np.float32)
    for start in range(0, categorical.shape[0], batch_size):
        end = min(start + batch_size, categorical.shape[0])
        logits, _watch = model(
            torch.from_numpy(categorical[start:end]).to(device),
            torch.from_numpy(numeric[start:end]).to(device),
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        scores = np.zeros(end - start, dtype=np.float32)
        for name, weight in SCORE_WEIGHTS.items():
            if name in targets:
                scores += float(weight) * probs[:, targets.index(name)]
        output[start:end] = scores
    return output


@torch.no_grad()
def ranker_probs(model, categorical, numeric, device, batch_size):
    """Sigmoid probabilities of every head, [n_candidates, n_targets]."""
    model.eval()
    parts = []
    for start in range(0, categorical.shape[0], batch_size):
        end = min(start + batch_size, categorical.shape[0])
        logits, _watch = model(
            torch.from_numpy(categorical[start:end]).to(device),
            torch.from_numpy(numeric[start:end]).to(device),
        )
        parts.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0, 0), dtype=np.float32)


def summarize_ranks(ranks: np.ndarray, users: np.ndarray, n_boot: int, seed: int) -> dict:
    finite = np.isfinite(ranks)
    result = {"queries": int(ranks.size), "coverage": float(finite.mean())}
    for k in (10, 20, 50, 100):
        values = (ranks <= k).astype(np.float64)
        mean, lo, hi = cluster_bootstrap_mean_ci(values, users, n_boot=n_boot, seed=seed + k)
        result[f"hit_at_{k}"] = {"value": mean, "ci95_user_cluster": [lo, hi]}
    reciprocal = np.zeros(ranks.size, dtype=np.float64)
    reciprocal[finite] = 1.0 / ranks[finite]
    mean, lo, hi = cluster_bootstrap_mean_ci(reciprocal, users, n_boot=n_boot, seed=seed + 999)
    result["mrr"] = {"value": mean, "ci95_user_cluster": [lo, hi]}
    result["mean_rank_when_recovered"] = float(np.mean(ranks[finite])) if finite.any() else None
    return result


def summarize_random_expected(candidate_counts: np.ndarray, recovered: np.ndarray,
                              users: np.ndarray, n_boot: int, seed: int) -> dict:
    """Exact expectation for a uniformly random permutation of each same candidate set."""
    counts = np.asarray(candidate_counts, dtype=np.int64)
    recovered = np.asarray(recovered, dtype=bool)
    if counts.shape != recovered.shape or counts.shape != users.shape:
        raise ValueError("candidate_counts, recovered and users must have matching shapes")
    valid = recovered & (counts > 0)
    result = {"queries": int(counts.size), "coverage": float(valid.mean())}
    safe_n = np.maximum(counts, 1).astype(np.float64)
    for k in (10, 20, 50, 100):
        per_query = valid.astype(np.float64) * np.minimum(k, safe_n) / safe_n
        mean, lo, hi = cluster_bootstrap_mean_ci(per_query, users, n_boot=n_boot, seed=seed + k)
        result[f"hit_at_{k}"] = {"value": mean, "ci95_user_cluster": [lo, hi]}
    harmonic = np.zeros(counts.size, dtype=np.float64)
    for i in np.flatnonzero(valid):
        harmonic[i] = np.sum(1.0 / np.arange(1, int(counts[i]) + 1))
    reciprocal = np.zeros(counts.size, dtype=np.float64)
    reciprocal[valid] = harmonic[valid] / safe_n[valid]
    mean, lo, hi = cluster_bootstrap_mean_ci(reciprocal, users, n_boot=n_boot, seed=seed + 999)
    result["mrr"] = {"value": mean, "ci95_user_cluster": [lo, hi]}
    result["mean_rank_when_recovered"] = (
        float(np.mean((counts[valid] + 1.0) / 2.0)) if valid.any() else None
    )
    result["mean_candidate_count"] = float(np.mean(counts))
    return result


def add_random_comparison(ranker_metrics: dict, random_metrics: dict,
                          ranker_ranks: np.ndarray, random_counts: np.ndarray,
                          recovered: np.ndarray, users: np.ndarray,
                          n_boot: int, seed: int) -> dict:
    """Paired user-cluster bootstrap of ranker metric minus exact random expectation."""
    output = {"random_order_expected": random_metrics, "ranker_minus_random": {}}
    for k in (10, 20, 50, 100):
        n = np.maximum(random_counts, 1).astype(np.float64)
        expected = recovered.astype(np.float64) * np.minimum(k, n) / n
        observed = (ranker_ranks <= k).astype(np.float64)
        mean, lo, hi = cluster_bootstrap_mean_ci(observed - expected, users, n_boot=n_boot, seed=seed + 2000 + k)
        output["ranker_minus_random"][f"hit_at_{k}"] = {
            "difference": mean, "ci95_user_cluster": [lo, hi]
        }
    n = np.maximum(random_counts, 1).astype(np.float64)
    expected_rr = np.zeros(n.size, dtype=np.float64)
    for i in np.flatnonzero(recovered):
        expected_rr[i] = np.sum(1.0 / np.arange(1, int(random_counts[i]) + 1)) / n[i]
    observed_rr = np.zeros(n.size, dtype=np.float64)
    finite = np.isfinite(ranker_ranks)
    observed_rr[finite] = 1.0 / ranker_ranks[finite]
    mean, lo, hi = cluster_bootstrap_mean_ci(observed_rr - expected_rr, users, n_boot=n_boot, seed=seed + 2999)
    output["ranker_minus_random"]["mrr"] = {
        "difference": mean, "ci95_user_cluster": [lo, hi]
    }
    output["ranker_metrics"] = ranker_metrics
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-data-dir", default="data/processed/ranking/1k_v3_catalog")
    parser.add_argument("--ranking-data-dir", default="data/processed/ranking/1k_v3_catalog")
    parser.add_argument("--retrieval-model-path", required=True)
    parser.add_argument("--retrieval-index-dir", required=True)
    parser.add_argument("--ranker-model-path", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--pool-hours", type=float, default=24.0)
    parser.add_argument("--pool-cap", type=int, default=0)
    parser.add_argument("--query-block", type=int, default=64)
    parser.add_argument("--candidate-budget", type=int, default=1000,
                        help="Top-K candidates from each source; union is up to 2K")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", default="reports/experiments/hybrid_ranker_validation.json")
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        parser.error("Test split requires --allow-test; tune on validation first.")
    if args.queries < 1 or args.candidate_budget < 1 or args.batch_size < 1:
        parser.error("--queries, --candidate-budget and --batch-size must be positive")

    started = time.perf_counter()
    retrieval_dir = Path(args.retrieval_data_dir)
    ranking_dir = Path(args.ranking_data_dir)
    retrieval_model_path = Path(args.retrieval_model_path)
    retrieval_index_dir = Path(args.retrieval_index_dir)
    ranker_path = Path(args.ranker_model_path)

    retrieval_meta = load_retrieval_metadata(retrieval_dir)
    retrieval_fp = data_fingerprint(retrieval_dir)
    retrieval_ckpt = torch.load(retrieval_model_path, map_location="cpu", weights_only=False)
    index_meta = json.loads((retrieval_index_dir / "index_metadata.json").read_text())
    if retrieval_ckpt.get("data_fingerprint") != retrieval_fp or index_meta.get("data_fingerprint") != retrieval_fp:
        raise ValueError("Retrieval checkpoint/index do not match retrieval prepared data")
    if index_meta.get("checkpoint_sha256") != sha256_file(retrieval_model_path):
        raise ValueError("Retrieval index was built from a different checkpoint")
    catalog_maps_path = retrieval_dir / "catalog_maps.npz"
    if index_meta.get("catalog_maps_sha256") != (sha256_file(catalog_maps_path) if catalog_maps_path.exists() else None):
        raise ValueError("Retrieval index/catalog maps mismatch")

    rank_meta = json.loads((ranking_dir / "metadata.json").read_text())
    rank_ckpt = torch.load(ranker_path, map_location="cpu", weights_only=False)
    if rank_ckpt.get("dataset") != rank_meta.get("dataset"):
        raise ValueError("Ranker checkpoint and ranking prepared data belong to different datasets")
    saved_schema = rank_ckpt.get("schema_fingerprint")
    if saved_schema is not None and saved_schema != schema_fingerprint(ranking_dir):
        raise ValueError("Ranker checkpoint feature schema does not match ranking prepared data")
    if rank_ckpt.get("binary_targets") != rank_meta.get("binary_targets"):
        raise ValueError("Ranker checkpoint targets differ from ranking prepared data")

    # Load Model 2's own user and item metadata maps. Raw IDs bridge the two prepared versions.
    with np.load(ranking_dir / "id_maps.npz", allow_pickle=False) as maps:
        rank_user_ids = np.asarray(maps["user_ids"], dtype=np.int64)
        rank_tab_ids = np.asarray(maps["tab_ids"], dtype=np.int64)
        rank_user_cat = np.asarray(maps["user_cat_by_index"], dtype=np.int32)
        rank_user_num = np.asarray(maps["user_numeric_by_index"], dtype=np.float32)
    with np.load(ranking_dir / "catalog_maps.npz", allow_pickle=False) as maps:
        rank_catalog = {
            "video_ids": np.asarray(maps["video_ids"], dtype=np.int64),
            "video_model_indices": np.asarray(maps["video_model_indices_by_index"], dtype=np.int32),
            "video_cat": np.asarray(maps["video_cat_by_index"], dtype=np.int32),
            "duration": np.asarray(maps["video_duration_seconds_by_index"], dtype=np.float32),
            "width": np.asarray(maps["video_width_by_index"], dtype=np.float32),
            "height": np.asarray(maps["video_height_by_index"], dtype=np.float32),
            "upload_ms": np.asarray(maps["video_upload_ms_by_index"], dtype=np.int64),
        }
    if np.any(np.diff(rank_catalog["video_ids"]) < 0):
        raise ValueError("Ranker catalogue video_ids must be sorted")
    if rank_catalog["video_model_indices"].shape != (len(rank_catalog["video_ids"]) + 1,):
        raise ValueError("Ranker video-ID embedding map does not align with its catalogue")
    if rank_user_cat.shape[0] != len(rank_user_ids) + 1 or rank_user_num.shape[0] != len(rank_user_ids) + 1:
        raise ValueError("Ranker user feature maps do not align with user_ids")
    if rank_catalog["video_cat"].shape[0] != len(rank_catalog["video_ids"]) + 1:
        raise ValueError("Ranker catalogue features do not align with video_ids")

    device = choose_device()
    print(f"Device: {device}")
    ranker = ReelsRankerV2(
        rank_ckpt["categorical_sizes"], rank_ckpt["num_numeric_features"],
        num_targets=len(rank_ckpt["binary_targets"]), embedding_dim=rank_ckpt["embedding_dim"],
    ).to(device)
    ranker.load_state_dict(rank_ckpt["model_state_dict"])
    ranker.eval()

    embeddings = np.load(retrieval_index_dir / index_meta["files"]["embeddings"], mmap_mode="r")
    n_catalog = int(embeddings.shape[0])
    retrieval_train = load_retrieval_split(retrieval_dir, "train")
    retrieval_split = load_retrieval_split(retrieval_dir, args.split)
    train_times = load_row_times(retrieval_dir, "train")
    split_times = load_row_times(retrieval_dir, args.split)
    if train_times.size != retrieval_train.rows or split_times.size != retrieval_split.rows:
        raise ValueError("row_times.npz does not align with retrieval splits")
    retrieval_catalog = build_video_catalog(retrieval_dir, retrieval_meta)
    train_items = item_catalog_indices(retrieval_train)
    split_items = item_catalog_indices(retrieval_split)
    origin_ms = int(train_times.min())
    train_windows = window_ids(train_times, origin_ms, args.pool_hours)
    split_windows = window_ids(split_times, origin_ms, args.pool_hours)
    pools = WindowPools.build(split_windows, split_items)

    rows, _ = positive_rows(retrieval_split, retrieval_meta)
    rows = rows[(split_items[rows] > 0) & (split_items[rows] <= n_catalog)]
    rng = np.random.default_rng(args.seed)
    if rows.size > args.queries:
        rows = np.sort(rng.choice(rows, size=args.queries, replace=False))
    q = rows.size
    user_idx_retrieval = retrieval_split.categorical[rows, 0].astype(np.int64)
    target_catalog_idx = split_items[rows].astype(np.int64)
    query_windows = split_windows[rows]
    query_times = split_times[rows].astype(np.int64)
    cold = retrieval_split.categorical[rows, 1] == 0

    # Translate retrieval's compact user/tab indices to raw IDs, then to Model 2 indices.
    with np.load(retrieval_dir / "id_maps.npz", allow_pickle=False) as maps:
        retrieval_user_ids = np.asarray(maps["user_ids"], dtype=np.int64)
        retrieval_tab_ids = np.asarray(maps["tab_ids"], dtype=np.int64)
    if user_idx_retrieval.size and int(user_idx_retrieval.max()) > len(retrieval_user_ids):
        raise ValueError("Retrieval user indices exceed retrieval id_maps.npz")
    raw_users = retrieval_user_ids[np.maximum(user_idx_retrieval - 1, 0)]
    raw_users = np.where(user_idx_retrieval > 0, raw_users, -1)
    retrieval_tab_idx = retrieval_split.categorical[rows, 2].astype(np.int64)
    raw_tabs = np.full(q, -1, dtype=np.int64)
    tab_valid = retrieval_tab_idx > 0
    if tab_valid.any() and int(retrieval_tab_idx[tab_valid].max()) > len(retrieval_tab_ids):
        raise ValueError("Retrieval tab indices exceed retrieval id_maps.npz")
    raw_tabs[tab_valid] = retrieval_tab_ids[retrieval_tab_idx[tab_valid] - 1]
    rank_user_idx = np.asarray([find_index(rank_user_ids, int(x)) if x >= 0 else 0 for x in raw_users], dtype=np.int32)
    rank_tab_idx = np.asarray([find_index(rank_tab_ids, int(x)) if x >= 0 else 0 for x in raw_tabs], dtype=np.int32)

    # Encode V3 user vectors for candidate generation.
    retrieval_model = TwoTowerRetrieval(retrieval_meta).to(device)
    retrieval_model.load_state_dict(retrieval_ckpt["model_state_dict"])
    retrieval_model.eval()
    user_vectors = np.empty((q, 128), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, q, 1024):
            chosen = rows[start:start + 1024]
            u_cat = torch.from_numpy(np.ascontiguousarray(
                retrieval_split.categorical[chosen][:, TwoTowerRetrieval.USER_COLUMNS])).to(device)
            u_num = torch.from_numpy(np.ascontiguousarray(
                retrieval_split.numeric[chosen][:, TwoTowerRetrieval.USER_NUMERIC_COLUMNS])).to(device)
            user_vectors[start:start + 1024] = retrieval_model.user_tower(u_cat, u_num).cpu().numpy()
    del retrieval_model

    train_pos_rows, train_pos_weights = positive_rows(retrieval_train, retrieval_meta)
    n_items = n_catalog + 1
    known = KnownPairs(
        retrieval_train.categorical[train_pos_rows, 0], train_items[train_pos_rows],
        int(retrieval_meta["num_users"]), n_items,
    )
    authors = retrieval_catalog.categorical[:, 1].astype(np.int64)
    train_authors = authors[train_items[train_pos_rows]]
    affinity = AuthorAffinity(
        retrieval_train.categorical[train_pos_rows, 0], train_authors,
        int(retrieval_meta["categorical_sizes"][AUTHOR_FIELD_INDEX]),
    )

    # Query-by-query scoring is intentionally streamed to keep memory bounded.
    ranker_ranks = {
        "v3_only": np.full(q, np.inf), "author_only": np.full(q, np.inf), "hybrid_union": np.full(q, np.inf)
    }
    candidate_counts = {name: np.zeros(q, dtype=np.int32) for name in ranker_ranks}
    retrieval_coverage = {"v3_only": np.zeros(q, dtype=bool), "author_only": np.zeros(q, dtype=bool),
                          "hybrid_union": np.zeros(q, dtype=bool)}
    missing_ranker_catalog = np.zeros(q, dtype=np.int32)
    scored_candidates = np.zeros(q, dtype=np.int32)
    query_users_for_ci = raw_users.copy()
    budget = args.candidate_budget
    # Same slates, several score formulas: the current hand-weighted sum, a sum that uses the
    # retrieval target's event weights, and every head on its own.
    variants = build_score_variants(rank_ckpt["binary_targets"], SCORE_WEIGHTS)
    variant_names = list(variants)
    variant_ranks = {v: {s: np.full(q, np.inf) for s in ranker_ranks} for v in variant_names}

    for window in np.unique(query_windows):
        query_indices = np.flatnonzero(query_windows == window)
        pool = pools.get(int(window)).astype(np.int64)
        if args.pool_cap and pool.size > args.pool_cap:
            pool = np.random.default_rng(args.seed + int(window) + 1).choice(pool, size=args.pool_cap, replace=False)
        pool = np.union1d(pool, target_catalog_idx[query_indices])
        pool_embeddings = np.asarray(embeddings[pool - 1], dtype=np.float32)
        pool_authors = authors[pool]
        pos_cols = np.searchsorted(pool, target_catalog_idx[query_indices])
        unique_users, inverse = np.unique(user_idx_retrieval[query_indices], return_inverse=True)
        known_by_user = known.grid(unique_users, pool)
        print(f"Window {int(window)}: {len(query_indices):,} queries, {len(pool):,} active videos", flush=True)

        for block_start in range(0, len(query_indices), args.query_block):
            block_end = min(block_start + args.query_block, len(query_indices))
            global_rows = query_indices[block_start:block_end]
            block_users = user_idx_retrieval[global_rows]
            block_user_inverse = inverse[block_start:block_end]
            block_pos_cols = pos_cols[block_start:block_end]
            valid_known = known_by_user[block_user_inverse]
            model_scores = user_vectors[global_rows] @ pool_embeddings.T
            author_scores = np.stack([
                affinity.scores(int(u), pool_authors) for u in block_users
            ])
            for local, qi in enumerate(global_rows):
                eligible_mask = ~valid_known[local].copy()
                eligible_mask[int(block_pos_cols[local])] = True
                eligible = np.flatnonzero(eligible_mask)
                mcols = select_topk_columns(model_scores[local], eligible, budget)
                acols = select_topk_columns(author_scores[local], eligible, budget, require_positive=True)
                union_cols = np.union1d(mcols, acols)
                sets = {
                    "v3_only": mcols,
                    "author_only": acols,
                    "hybrid_union": union_cols,
                }
                target_col = int(pos_cols[block_start + local])
                retrieval_coverage["v3_only"][qi] = bool(np.any(mcols == target_col))
                retrieval_coverage["author_only"][qi] = bool(np.any(acols == target_col))
                retrieval_coverage["hybrid_union"][qi] = bool(np.any(union_cols == target_col))
                target_catalog = int(pool[target_col])
                target_raw = int(retrieval_catalog.raw_ids[target_catalog - 1])
                rank_user = int(rank_user_idx[qi])
                rank_tab = int(rank_tab_idx[qi])
                if rank_user >= rank_user_cat.shape[0]:
                    rank_user = 0
                user_cat = rank_user_cat[rank_user]
                user_num = rank_user_num[rank_user]
                all_union_raw = pool_columns_to_raw_ids(
                    pool, union_cols, retrieval_catalog.raw_ids
                )
                cat, num, matched_positions = build_ranker_features(
                    all_union_raw, int(query_times[qi]), rank_user, rank_tab,
                    user_cat, user_num, rank_catalog, rank_meta,
                )
                if matched_positions.size:
                    if cat.shape[1] != len(rank_ckpt["categorical_sizes"]):
                        raise ValueError(
                            f"Constructed Model 2 categorical width {cat.shape[1]} does not match "
                            f"checkpoint width {len(rank_ckpt['categorical_sizes'])}"
                        )
                    if num.shape[1] != int(rank_ckpt["num_numeric_features"]):
                        raise ValueError(
                            f"Constructed Model 2 numeric width {num.shape[1]} does not match "
                            f"checkpoint width {rank_ckpt['num_numeric_features']}"
                        )
                    for column, size in enumerate(rank_ckpt["categorical_sizes"]):
                        if cat[:, column].min() < 0 or cat[:, column].max() >= int(size):
                            raise ValueError(
                                f"Constructed Model 2 categorical feature {column} is outside "
                                f"checkpoint vocabulary size {size}"
                            )
                    probs = ranker_probs(ranker, cat, num, device, args.batch_size)
                    _, score_matrix = variant_score_matrix(probs, rank_ckpt["binary_targets"], variants)
                    scored_raw = all_union_raw[matched_positions]
                    row_by_raw = {int(v): i for i, v in enumerate(scored_raw)}
                else:
                    score_matrix = np.empty((0, len(variant_names)), dtype=np.float32)
                    row_by_raw = {}
                missing_ranker_catalog[qi] = len(all_union_raw) - len(row_by_raw)
                scored_candidates[qi] = len(row_by_raw)
                for name, cols in sets.items():
                    raw_ids = pool_columns_to_raw_ids(
                        pool, cols, retrieval_catalog.raw_ids
                    )
                    ids = [int(v) for v in raw_ids if int(v) in row_by_raw]
                    candidate_counts[name][qi] = len(ids)
                    if not ids or target_raw not in ids:
                        continue
                    row_idx = np.fromiter((row_by_raw[v] for v in ids), dtype=np.int64, count=len(ids))
                    target_at = ids.index(target_raw)
                    for j, variant in enumerate(variant_names):
                        variant_ranks[variant][name][qi] = expected_rank(score_matrix[row_idx, j], target_at)

    ranker_ranks = variant_ranks["current_weights"]   # the report below keeps its existing meaning

    report = {
        "experiment": "Model 2 reranking of V3 / author / hybrid candidate sets",
        "split": args.split, "queries": int(q), "unique_users": int(np.unique(raw_users).size),
        "candidate_budget_per_source": budget, "maximum_union_size": 2 * budget,
        "retrieval_model_path": str(retrieval_model_path), "retrieval_index_dir": str(retrieval_index_dir),
        "ranker_model_path": str(ranker_path), "ranking_data_dir": str(ranking_dir),
        "candidate_generation_recall": {
            "v3_recovered": float(retrieval_coverage["v3_only"].mean()),
            "author_recovered": float(retrieval_coverage["author_only"].mean()),
            "union_recovered": float(retrieval_coverage["hybrid_union"].mean()),
        },
        "ranker_reranking": {
            name: summarize_ranks(ranks, query_users_for_ci, args.bootstrap, args.seed)
            for name, ranks in ranker_ranks.items()
        },
        "same_candidate_random_baseline": {
            name: add_random_comparison(
                summarize_ranks(ranker_ranks[name], query_users_for_ci, args.bootstrap, args.seed),
                summarize_random_expected(candidate_counts[name], np.isfinite(ranker_ranks[name]),
                                          query_users_for_ci, args.bootstrap, args.seed),
                ranker_ranks[name], candidate_counts[name], np.isfinite(ranker_ranks[name]),
                query_users_for_ci, args.bootstrap, args.seed,
            )
            for name in ranker_ranks
        },
        "candidate_set_sizes": {
            name: {
                "mean": float(candidate_counts[name].mean()),
                "median": float(np.median(candidate_counts[name])),
                "p95": float(np.quantile(candidate_counts[name], 0.95)),
                "max": int(candidate_counts[name].max()),
            } for name in candidate_counts
        },
        "cohorts": {},
        "ranker_catalog_coverage": {
            "target_available_in_ranker_catalog_after_retrieval": {
                name: float(np.mean(np.isfinite(ranker_ranks[name]))) for name in ranker_ranks
            },
            "mean_scored_candidates": float(scored_candidates.mean()),
            "mean_candidates_missing_ranker_catalog": float(missing_ranker_catalog.mean()),
            "queries_with_any_missing_ranker_candidates": float(np.mean(missing_ranker_catalog > 0)),
        },
        "notes": [
            "A held-out positive absent from a candidate set receives infinite rank and zero reciprocal rank.",
            "Ranker candidate features are built from Model 2's own user/item maps, joined to retrieval candidates by raw IDs.",
            "This evaluates an existing Model 2 checkpoint; it does not retrain the ranker.",
            "The same-candidate random baseline is the exact expectation under a uniform random permutation, conditional on the same scored candidate set and target coverage.",
            "Ranker-minus-random confidence intervals use paired per-query differences with user-cluster bootstrap.",
            "The candidate pool is the 24-hour validation exposure window. It includes videos exposed later than an individual query within that window, so absolute coverage can be optimistic relative to point-in-time serving.",
            "Conditional percentile metrics include only queries whose target was retrieved into a slate with more than one candidate; read them together with unconditional candidate coverage.",
            "retrieval_target_weighted_sum uses the exact positive-event weights and short-view penalty from the retrieval target. is_hate is a hard exclusion in positive_rows, not an additive penalty, so its head is evaluated separately with reversed direction.",
            "Score formulas are sums of uncalibrated sigmoid outputs; the target-weighted sum is a diagnostic variant, not a calibrated expected-relevance estimate.",
            "The validation split is used for tuning. Run test once only after decisions are frozen.",
        ],
    }
    report["score_variants"] = {
        "note": ("Same candidate sets, different Model 2 score formulas. mean_percentile is the target's "
                 "conditional position inside its candidate set (0 = first, 0.5 = random, 1 = last); "
                 "delta_vs_random < 0 beats random. Only queries where the target is present and slate size > 1 "
                 "are included. head:short_view and head:is_hate are ranked by the negative probability."),
        "variants": {
            source: {
                variant: {
                    **summarize_percentile(variant_ranks[variant][source], candidate_counts[source],
                                           query_users_for_ci, args.bootstrap, args.seed + 7),
                    "hit_at_100": dict(zip(("value", "ci_lo", "ci_hi"), cluster_bootstrap_mean_ci(
                        (variant_ranks[variant][source] <= 100).astype(np.float64),
                        query_users_for_ci, n_boot=args.bootstrap, seed=args.seed + 100))),
                }
                for variant in variant_names
            }
            for source in ranker_ranks
        },
    }
    report["score_variants"]["hybrid_union_paired_vs_head_is_like"] = {
        variant: summarize_paired_variant(
            variant_ranks[variant]["hybrid_union"],
            variant_ranks["head:is_like"]["hybrid_union"],
            candidate_counts["hybrid_union"],
            query_users_for_ci,
            args.bootstrap,
            args.seed + 211,
            hit_k=100,
        )
        for variant in variant_names
        if variant != "head:is_like"
    }

    for cohort, mask in (("cold_start", cold), ("seen_id", ~cold)):
        if mask.any():
            report["cohorts"][cohort] = {
                "queries": int(mask.sum()),
                **{
                    name: summarize_ranks(ranks[mask], query_users_for_ci[mask], args.bootstrap, args.seed)
                    for name, ranks in ranker_ranks.items()
                },
            }
    report["elapsed_seconds"] = time.perf_counter() - started
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print("\n=== Model 2 reranking results ===")
    for name, metrics in report["ranker_reranking"].items():
        print(f"{name:14s} coverage={metrics['coverage']:.3f} "
              f"Hit@10={metrics['hit_at_10']['value']:.3f} "
              f"Hit@20={metrics['hit_at_20']['value']:.3f} "
              f"Hit@50={metrics['hit_at_50']['value']:.3f} "
              f"MRR={metrics['mrr']['value']:.4f}")
    print(f"Mean ranker catalogue misses/query: {report['ranker_catalog_coverage']['mean_candidates_missing_ranker_catalog']:.1f}")
    print("\\n=== Model 2 vs exact random ordering of the same scored candidates ===")
    for name in ranker_ranks:
        comparison = report["same_candidate_random_baseline"][name]
        random_hit = comparison["random_order_expected"]["hit_at_100"]["value"]
        ranker_hit = comparison["ranker_metrics"]["hit_at_100"]["value"]
        delta = comparison["ranker_minus_random"]["hit_at_100"]
        print(f"{name:14s} Model2 Hit@100={ranker_hit:.4f} "
              f"random expected={random_hit:.4f} "
              f"delta={delta['difference']:+.4f} "
              f"CI=[{delta['ci95_user_cluster'][0]:+.4f},{delta['ci95_user_cluster'][1]:+.4f}]")
    for source in ("hybrid_union", "v3_only"):
        print(f"\n=== Score formulas on the same {source} slates (mean target percentile; 0.5 = random) ===")
        table = report["score_variants"]["variants"][source]
        scored = [(v, m) for v, m in table.items() if m["mean_percentile"] is not None]
        for variant, m in sorted(scored, key=lambda item: item[1]["mean_percentile"]):
            lo, hi = m["delta_ci95_user_cluster"]
            print(f"{variant:34s} n={m['queries_scored']:5d} users={m['users_scored']:4d} "
                  f"percentile={m['mean_percentile']:.3f} "
                  f"delta={m['delta_vs_random']:+.3f} CI=[{lo:+.3f},{hi:+.3f}] "
                  f"Hit@100={m['hit_at_100']['value']:.4f}")
    print(f"Saved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
