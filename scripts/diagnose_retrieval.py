"""Diagnose full-catalogue retrieval by cold-start, item popularity and seen-item filtering.

Uses the validation split only. It reproduces the exact query sampling used by
scripts.evaluate_retrieval, then reports both the unfiltered catalogue ranks and
ranks after removing other known-positive training items for the same user.
The held-out target itself is never filtered, including repeat user-item pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.eval_metrics import relevance_gains
from common.retrieval_eval import rank_against_catalog, score_tolerance
from model1_retrieval.data import (
    data_fingerprint,
    item_catalog_indices,
    load_metadata,
    load_split,
    pair_keys,
    positive_rows,
)
from model1_retrieval.model import TwoTowerRetrieval


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


def make_user_positive_items(train, metadata: dict, n_catalogue_rows: int) -> dict[int, np.ndarray]:
    rows, _ = positive_rows(train, metadata)
    keys = pair_keys(train, rows, n_catalogue_rows)
    users = keys // n_catalogue_rows
    items = keys % n_catalogue_rows
    result: dict[int, np.ndarray] = {}
    for user in np.unique(users):
        result[int(user)] = items[users == user].astype(np.int32, copy=False)
    return result


def subtract_seen_positive_competitors(
    raw_ranks: np.ndarray,
    user_vectors: np.ndarray,
    positive_scores: np.ndarray,
    query_users: np.ndarray,
    positive_items: np.ndarray,
    user_positive_items: dict[int, np.ndarray],
    embeddings: np.ndarray,
    block_size: int = 8192,
) -> np.ndarray:
    """Remove other train-positive items from each query's raw rank.

    The positive target is retained if the same user-item pair also occurred in
    training. A removed strictly-higher item subtracts 1 rank position; a removed
    exact-score tie subtracts half a position, matching expected-rank evaluation.
    """
    ranks = np.asarray(raw_ranks, dtype=np.float64).copy()
    rank_contribution_removed = np.zeros(len(ranks), dtype=np.float64)
    for q, (user, target) in enumerate(zip(query_users.tolist(), positive_items.tolist())):
        item_ids = user_positive_items.get(int(user))
        if item_ids is None or item_ids.size == 0:
            continue
        competitors = item_ids[item_ids != int(target)]
        if competitors.size == 0:
            continue
        target_score = float(positive_scores[q])
        count_greater = 0
        count_equal = 0
        for start in range(0, competitors.size, block_size):
            ids = competitors[start:start + block_size].astype(np.int64, copy=False)
            item_vectors = np.asarray(embeddings[ids - 1], dtype=np.float32)
            scores = item_vectors @ user_vectors[q].astype(np.float32, copy=False)
            tolerance = float(score_tolerance(np.asarray(target_score)))
            count_greater += int(np.count_nonzero(scores > target_score + tolerance))
            count_equal += int(np.count_nonzero(np.abs(scores - target_score) <= tolerance))
        rank_contribution_removed[q] = count_greater + 0.5 * count_equal
    filtered = ranks - rank_contribution_removed
    if np.any(filtered < 1.0 - 1e-9):
        raise RuntimeError("Filtered rank fell below 1; exclusion accounting is inconsistent")
    return filtered


def popularity_ranks(popularity: np.ndarray, positive_items: np.ndarray,
                     query_users: np.ndarray, user_positive_items: dict[int, np.ndarray],
                     filtered: bool) -> np.ndarray:
    """Expected popularity rank, optionally excluding other known positives per user."""
    ordered = np.sort(np.asarray(popularity, dtype=np.float64))
    target_scores = np.asarray(popularity[positive_items - 1], dtype=np.float64)
    right = np.searchsorted(ordered, target_scores, side="right")
    left = np.searchsorted(ordered, target_scores, side="left")
    greater = ordered.size - right
    equal = right - left
    if filtered:
        for q, user in enumerate(query_users.tolist()):
            ids = user_positive_items.get(int(user))
            if ids is None or ids.size == 0:
                continue
            ids = ids[ids != int(positive_items[q])]
            if ids.size:
                excluded = popularity[ids - 1]
                greater[q] -= int(np.count_nonzero(excluded > target_scores[q]))
                equal[q] -= int(np.count_nonzero(excluded == target_scores[q]))
    return (1.0 + greater + 0.5 * np.maximum(equal - 1, 0)).astype(np.float64)


def cluster_bootstrap_mean(values: np.ndarray, users: np.ndarray, n_boot: int,
                           seed: int) -> tuple[float, float, float]:
    """Query-weighted mean and CI with users, not interactions, as bootstrap units."""
    values = np.asarray(values, dtype=np.float64)
    users = np.asarray(users)
    unique, inverse = np.unique(users, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    point = float(values.mean()) if values.size else float("nan")
    if not values.size:
        return point, point, point
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        draw = rng.integers(0, len(unique), size=len(unique))
        samples[i] = sums[draw].sum() / counts[draw].sum()
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return point, float(lo), float(hi)


def metric_summary(ranks: np.ndarray, users: np.ndarray, ks: list[int],
                   n_boot: int, seed: int) -> dict:
    out = {
        "queries": int(len(ranks)),
        "unique_users": int(np.unique(users).size),
        "median_rank": float(np.median(ranks)) if len(ranks) else None,
        "mrr": float(np.mean(1.0 / ranks)) if len(ranks) else None,
    }
    for k in ks:
        mean, lo, hi = cluster_bootstrap_mean((ranks <= k).astype(np.float64), users, n_boot, seed + k)
        out[f"recall@{k}"] = {"value": mean, "ci95_user_cluster": [lo, hi]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v3_catalog")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--ks", type=int, nargs="+", default=[20, 100, 500, 1000])
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--query-block", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=131072)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.queries, args.bootstrap, args.query_block, args.chunk_size) < 1:
        parser.error("queries, bootstrap, query-block and chunk-size must be positive")

    started = time.perf_counter()
    data_dir, model_path, index_dir = Path(args.data_dir), Path(args.model_path), Path(args.index_dir)
    metadata = load_metadata(data_dir)
    schema = data_fingerprint(data_dir)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    index_meta = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
    if checkpoint.get("dataset") != metadata.get("dataset") or index_meta.get("dataset") != metadata.get("dataset"):
        raise ValueError("Model, index and prepared dataset do not match")
    if checkpoint.get("data_fingerprint") != schema or index_meta.get("data_fingerprint") != schema:
        raise ValueError("Model/index data fingerprint does not match prepared data")
    if index_meta.get("checkpoint_sha256") != sha256_file(model_path):
        raise ValueError("Index was not built from the supplied checkpoint")
    catalog_map = data_dir / "catalog_maps.npz"
    catalog_hash = sha256_file(catalog_map) if catalog_map.exists() else None
    if index_meta.get("catalog_maps_sha256") != catalog_hash:
        raise ValueError("Index was built from a different catalogue map")

    embeddings = np.load(index_dir / index_meta["files"]["embeddings"], mmap_mode="r")
    n_catalogue = int(embeddings.shape[0])
    validation = load_split(data_dir, "validation")
    train = load_split(data_dir, "train")
    targets = list(metadata["binary_targets"])
    gains = relevance_gains(validation.labels, targets)
    users_all = validation.categorical[:, 0]
    video_id_indices = validation.categorical[:, 1]
    catalog_indices = item_catalog_indices(validation)
    eligible = (gains > 0) & (users_all > 0) & (catalog_indices > 0) & (catalog_indices <= n_catalogue)
    eligible_rows = np.flatnonzero(eligible)
    rng = np.random.default_rng(args.seed)
    if len(eligible_rows) > args.queries:
        rows = np.sort(rng.choice(eligible_rows, size=args.queries, replace=False))
    else:
        rows = np.sort(eligible_rows)
    query_users = users_all[rows].astype(np.int64)
    positive_items = catalog_indices[rows].astype(np.int64)
    positive_vectors = np.asarray(embeddings[positive_items - 1], dtype=np.float32)

    device = choose_device()
    print(f"Encoding {len(rows):,} validation queries on {device}...")
    model = TwoTowerRetrieval(metadata).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    user_vectors = np.empty((len(rows), 128), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), 2048):
            ix = rows[start:start + 2048]
            u_cat = torch.from_numpy(np.ascontiguousarray(validation.categorical[ix][:, TwoTowerRetrieval.USER_COLUMNS])).to(device)
            u_num = torch.from_numpy(np.ascontiguousarray(validation.numeric[ix][:, TwoTowerRetrieval.USER_NUMERIC_COLUMNS])).to(device)
            user_vectors[start:start + 2048] = model.user_tower(u_cat, u_num).cpu().numpy()
    del model

    # Same full-scan rank implementation and query set as evaluate_retrieval.py.
    print(f"Exact-scanning {n_catalogue:,} catalogue vectors...")
    raw = rank_against_catalog(
        {"model": user_vectors}, positive_items - 1, embeddings,
        chunk_size=args.chunk_size, query_block=args.query_block,
        progress=lambda done, total: print(f"  scanned {done:,}/{total:,}", flush=True),
    )["model"]
    positive_scores = np.einsum("qd,qd->q", user_vectors, positive_vectors)

    train_positive_rows, _ = positive_rows(train, metadata)
    user_positive_items = make_user_positive_items(train, metadata, n_catalogue + 1)
    print("Removing other known-positive training items from each user's candidate set...")
    filtered = subtract_seen_positive_competitors(
        raw, user_vectors, positive_scores, query_users, positive_items,
        user_positive_items, embeddings,
    )

    train_catalog = item_catalog_indices(train)
    train_exposure = np.bincount(train_catalog[train_catalog > 0], minlength=n_catalogue + 1)[1:]
    train_gain = relevance_gains(train.labels, list(metadata["binary_targets"]))
    train_popularity = np.bincount(train_catalog, weights=train_gain, minlength=n_catalogue + 1)[1:]
    pop_raw = popularity_ranks(train_popularity, positive_items, query_users, user_positive_items, filtered=False)
    pop_filtered = popularity_ranks(train_popularity, positive_items, query_users, user_positive_items, filtered=True)

    # Cohorts are mutually exclusive for item-frequency analysis.
    exposure = train_exposure[positive_items - 1]
    is_unseen = video_id_indices[rows] == 0
    item_cohort = np.full(len(rows), "train_exposures_100_plus", dtype=object)
    item_cohort[exposure < 100] = "train_exposures_20_to_99"
    item_cohort[exposure < 20] = "train_exposures_5_to_19"
    item_cohort[exposure < 5] = "train_exposures_1_to_4"
    item_cohort[exposure == 0] = "zero_train_exposures"
    item_cohort[is_unseen] = "unseen_video_id_metadata_backed"

    report = {
        "model_path": str(model_path),
        "index_dir": str(index_dir),
        "split": "validation",
        "catalogue_size": n_catalogue,
        "queries": int(len(rows)),
        "unique_users": int(np.unique(query_users).size),
        "query_seed": args.seed,
        "ranking_definition": "expected 1-based rank: 1 + strictly greater scores + 0.5 * other exact-score ties",
        "filtered_definition": "remove all other unique training positive user-item pairs for the query user; retain the held-out target itself",
        "cold_start_query_pct": float(100 * is_unseen.mean()) if len(rows) else 0.0,
        "same_user_video_pair_seen_in_train_pct": float(100 * np.mean(np.isin(
            query_users.astype(np.int64) * (n_catalogue + 1) + positive_items,
            pair_keys(train, train_positive_rows, n_catalogue + 1),
        ))) if len(rows) else 0.0,
        "overall": {
            "model_unfiltered": metric_summary(raw, query_users, args.ks, args.bootstrap, args.seed),
            "model_filtered_train_positives": metric_summary(filtered, query_users, args.ks, args.bootstrap, args.seed),
            "popularity_unfiltered": metric_summary(pop_raw, query_users, args.ks, args.bootstrap, args.seed),
            "popularity_filtered_train_positives": metric_summary(pop_filtered, query_users, args.ks, args.bootstrap, args.seed),
        },
        "item_cohorts": {},
        "event_cohorts": {},
    }

    for cohort in sorted(np.unique(item_cohort).tolist()):
        mask = item_cohort == cohort
        report["item_cohorts"][cohort] = {
            "model_unfiltered": metric_summary(raw[mask], query_users[mask], args.ks, args.bootstrap, args.seed),
            "model_filtered_train_positives": metric_summary(filtered[mask], query_users[mask], args.ks, args.bootstrap, args.seed),
            "popularity_unfiltered": metric_summary(pop_raw[mask], query_users[mask], args.ks, args.bootstrap, args.seed),
            "popularity_filtered_train_positives": metric_summary(pop_filtered[mask], query_users[mask], args.ks, args.bootstrap, args.seed),
        }

    for target in ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view", "complete_view"]:
        if target not in targets:
            continue
        mask = validation.labels[rows, targets.index(target)] > 0
        if mask.any():
            report["event_cohorts"][target] = {
                "model_unfiltered": metric_summary(raw[mask], query_users[mask], args.ks, args.bootstrap, args.seed),
                "model_filtered_train_positives": metric_summary(filtered[mask], query_users[mask], args.ks, args.bootstrap, args.seed),
                "queries": int(mask.sum()),
            }

    report["elapsed_seconds"] = time.perf_counter() - started
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print("\n=== Overall full-catalogue retrieval diagnosis ===")
    for name, result in report["overall"].items():
        print(f"\n{name}: queries={result['queries']:,}; users={result['unique_users']:,}; median_rank={result['median_rank']:,.0f}; MRR={result['mrr']:.6f}")
        for k in args.ks:
            m = result[f"recall@{k}"]
            print(f"  Recall@{k}: {m['value']:.5f} [{m['ci95_user_cluster'][0]:.5f}, {m['ci95_user_cluster'][1]:.5f}] (user-cluster CI)")
    print("\n=== Item cohorts: query count and Recall@1000 ===")
    for cohort, results in report["item_cohorts"].items():
        r = results["model_filtered_train_positives"]
        print(f"{cohort:<40} n={r['queries']:>5} users={r['unique_users']:>4} R@1000={r['recall@1000']['value']:.5f}")
    print(f"\nSaved diagnostic report: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
