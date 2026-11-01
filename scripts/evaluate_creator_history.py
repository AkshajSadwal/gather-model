"""Evaluate leakage-safe creator-history retrieval on time-local active pools.

Compares the existing static train-only author-count baseline with creator-history
scores computed strictly as of each validation query timestamp. Candidate generation
is the union of Model 1 top-K and creator-affinity top-K within the same active pool.

Creator-affinity ties are resolved using Model 1 video similarity, then catalogue
index, so videos from the same selected creator are not chosen arbitrarily.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.creator_history import CreatorHistoryIndex
from common.data_version import prepared_data_fingerprint, schema_fingerprint
from common.eval_metrics import cluster_bootstrap_mean_ci
from common.pools import KnownPairs, WindowPools, load_row_times, window_ids
from model1_retrieval.data import (
    build_video_catalog,
    data_fingerprint,
    item_catalog_indices,
    load_metadata,
    load_split,
    positive_rows,
)
from model1_retrieval.model import TwoTowerRetrieval
from scripts.evaluate_retrieval_pools import (
    AuthorAffinity,
    choose_device,
    sha256_file,
    select_topk_columns,
)


def select_creator_topk(
    creator_scores: np.ndarray,
    model_scores: np.ndarray,
    eligible: np.ndarray,
    item_indices: np.ndarray,
    k: int,
) -> np.ndarray:
    """Select positive-affinity items by creator score, Model 1 score, then item ID.

    Using Model 1 similarity as a secondary key resolves the many video-level ties
    created when all videos by the same creator share one affinity score.
    """
    creator_scores = np.asarray(creator_scores, dtype=np.float32)
    model_scores = np.asarray(model_scores, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=np.int64)
    item_indices = np.asarray(item_indices, dtype=np.int64)
    if creator_scores.ndim != 1 or model_scores.shape != creator_scores.shape:
        raise ValueError("creator_scores and model_scores must be matching 1-D arrays")
    if item_indices.shape != creator_scores.shape:
        raise ValueError("item_indices must match score arrays")
    if k < 1:
        raise ValueError("k must be positive")
    eligible = eligible[creator_scores[eligible] > 0]
    if eligible.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((item_indices[eligible], -model_scores[eligible], -creator_scores[eligible]))
    return eligible[order[: int(k)]]


def _load_history(history_dir: Path, data_dir: Path, metadata: dict) -> CreatorHistoryIndex:
    manifest_path = history_dir / "metadata.json"
    artifact_path = history_dir / "creator_history.npz"
    row_times_path = data_dir / "row_times.npz"
    for path in (manifest_path, artifact_path, row_times_path):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != 1:
        raise ValueError(f"Unsupported creator history artifact version: {manifest.get('artifact_version')}")
    expected_schema = schema_fingerprint(data_dir)
    expected_prepared = prepared_data_fingerprint(data_dir)
    if manifest.get("schema_fingerprint") != expected_schema:
        raise ValueError("Creator history schema fingerprint does not match the prepared dataset")
    if manifest.get("prepared_data_fingerprint") != expected_prepared:
        raise ValueError("Creator history prepared-data fingerprint does not match the prepared dataset")
    if manifest.get("row_times_sha256") != sha256_file(row_times_path):
        raise ValueError("Creator history was built from different row_times.npz")
    if int(manifest.get("num_users", -1)) != int(metadata["num_users"]):
        raise ValueError("Creator history user vocabulary does not match the dataset")
    author_col = list(metadata["categorical_features"]).index("video:author_id")
    num_authors = int(metadata["categorical_sizes"][author_col])
    if int(manifest.get("num_authors", -1)) != num_authors:
        raise ValueError("Creator history author vocabulary does not match the dataset")

    with np.load(artifact_path, allow_pickle=False) as data:
        required = {"offsets", "times_ms", "author_ids", "weights", "split_codes"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Creator history artifact is missing arrays: {sorted(missing)}")
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        times = np.asarray(data["times_ms"], dtype=np.int64)
        authors = np.asarray(data["author_ids"], dtype=np.int32)
        weights = np.asarray(data["weights"], dtype=np.float32)
        split_codes = np.asarray(data["split_codes"], dtype=np.uint8)

    if split_codes.shape != times.shape:
        raise ValueError("Creator history split_codes are not aligned with events")
    counts = manifest.get("positive_creator_events_by_split", {})
    if int(sum(int(v) for v in counts.values())) != int(times.size):
        raise ValueError("Creator history event counts do not match its manifest")

    return CreatorHistoryIndex(
        offsets=offsets,
        times_ms=times,
        author_ids=authors,
        weights=weights,
        num_users=int(metadata["num_users"]),
        num_authors=num_authors,
    )


def _bootstrap(values: np.ndarray, users: np.ndarray, n_boot: int, seed: int) -> dict:
    mean, lo, hi = cluster_bootstrap_mean_ci(values.astype(np.float64), users, n_boot=n_boot, seed=seed)
    return {"value": mean, "ci95_user_cluster": [lo, hi]}


def _state(n: int) -> dict[str, np.ndarray]:
    return {
        "creator_hit": np.zeros(n, dtype=bool),
        "union_hit": np.zeros(n, dtype=bool),
        "model_hit": np.zeros(n, dtype=bool),
        "creator_only": np.zeros(n, dtype=bool),
        "model_only": np.zeros(n, dtype=bool),
        "both": np.zeros(n, dtype=bool),
        "creator_size": np.zeros(n, dtype=np.int32),
        "union_size": np.zeros(n, dtype=np.int32),
        "overlap_size": np.zeros(n, dtype=np.int32),
        "has_history_creator_in_pool": np.zeros(n, dtype=bool),
    }


def _summarise_state(state: dict, users: np.ndarray, cold: np.ndarray, n_boot: int, seed: int) -> dict:
    summary = {}
    for name in ("creator_hit", "model_hit", "union_hit", "creator_only", "model_only", "both"):
        summary[name] = _bootstrap(state[name], users, n_boot, seed)
    summary["mean_creator_candidates"] = float(state["creator_size"].mean())
    summary["mean_union_candidates"] = float(state["union_size"].mean())
    summary["mean_overlap_candidates"] = float(state["overlap_size"].mean())
    summary["queries_with_prior_creator_in_pool"] = float(state["has_history_creator_in_pool"].mean())
    summary["cohorts"] = {}
    for label, mask in (("cold_start", cold), ("seen_id", ~cold)):
        if mask.any():
            summary["cohorts"][label] = {
                "queries": int(mask.sum()),
                "creator_hit": float(state["creator_hit"][mask].mean()),
                "union_hit": float(state["union_hit"][mask].mean()),
                "creator_only": float(state["creator_only"][mask].mean()),
                "model_only": float(state["model_only"][mask].mean()),
            }
    return summary


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    data_dir = Path(args.data_dir)
    history_dir = Path(args.history_dir)
    model_path = Path(args.model_path)
    index_dir = Path(args.index_dir)

    metadata = load_metadata(data_dir)
    schema = data_fingerprint(data_dir)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    index_meta = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
    if checkpoint.get("dataset") != metadata.get("dataset") or index_meta.get("dataset") != metadata.get("dataset"):
        raise ValueError("Model, index and prepared dataset do not match")
    if checkpoint.get("data_fingerprint") != schema or index_meta.get("data_fingerprint") != schema:
        raise ValueError("Prepared metadata/ID maps differ from model or index")
    if index_meta.get("checkpoint_sha256") != sha256_file(model_path):
        raise ValueError("Retrieval index was built from a different checkpoint")
    catalog_map_path = data_dir / "catalog_maps.npz"
    if index_meta.get("catalog_maps_sha256") != (sha256_file(catalog_map_path) if catalog_map_path.exists() else None):
        raise ValueError("Retrieval index was built from a different catalogue map")

    history = _load_history(history_dir, data_dir, metadata)
    history_manifest = json.loads((history_dir / "metadata.json").read_text(encoding="utf-8"))
    embeddings = np.load(index_dir / index_meta["files"]["embeddings"], mmap_mode="r")
    n_catalog = int(embeddings.shape[0])
    n_items = n_catalog + 1

    print(f"Index: {n_catalog:,} videos x {embeddings.shape[1]} dims")
    print("Loading prepared splits and row times...")
    train = load_split(data_dir, "train")
    split = load_split(data_dir, args.split)
    train_times = load_row_times(data_dir, "train")
    split_times = load_row_times(data_dir, args.split)
    if train_times.size != train.rows or split_times.size != split.rows:
        raise ValueError("row_times.npz does not align with prepared split rows")

    catalog = build_video_catalog(data_dir, metadata)
    catalog_author = catalog.categorical[:, 1].astype(np.int64, copy=False)
    origin_ms = int(train_times.min())
    train_items = item_catalog_indices(train)
    split_items = item_catalog_indices(split)
    train_windows = window_ids(train_times, origin_ms, args.pool_hours)
    split_windows = window_ids(split_times, origin_ms, args.pool_hours)
    pools = WindowPools.build(split_windows, split_items)

    rows, _ = positive_rows(split, metadata)
    rows = rows[(split_items[rows] > 0) & (split_items[rows] <= n_catalog)]
    rng = np.random.default_rng(args.seed)
    if rows.size > args.queries:
        rows = np.sort(rng.choice(rows, size=args.queries, replace=False))
    q = int(rows.size)
    if q == 0:
        raise ValueError("No eligible positive rows are available for this evaluation")
    users = split.categorical[rows, 0].astype(np.int64, copy=False)
    positives = split_items[rows].astype(np.int64, copy=False)
    query_times = split_times[rows].astype(np.int64, copy=False)
    query_windows = split_windows[rows]
    cold = split.categorical[rows, 1] == 0
    print(f"{q:,} queries from {np.unique(users).size:,} users; {cold.mean():.1%} cold-start targets")

    print("Encoding Model 1 user vectors...")
    device = choose_device()
    model = TwoTowerRetrieval(metadata).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    user_vectors = np.empty((q, 128), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, q, 2048):
            chosen = rows[start:start + 2048]
            u_cat = torch.from_numpy(np.ascontiguousarray(split.categorical[chosen][:, TwoTowerRetrieval.USER_COLUMNS])).to(device)
            u_num = torch.from_numpy(np.ascontiguousarray(split.numeric[chosen][:, TwoTowerRetrieval.USER_NUMERIC_COLUMNS])).to(device)
            user_vectors[start:start + 2048] = model.user_tower(u_cat, u_num).cpu().numpy()
    del model

    print("Building training-positive exclusion and static author baseline...")
    train_pos_rows, _ = positive_rows(train, metadata)
    known = KnownPairs(
        train.categorical[train_pos_rows, 0],
        train_items[train_pos_rows],
        int(metadata["num_users"]),
        n_items,
    )
    n_authors = int(metadata["categorical_sizes"][29])
    static_author_ids = catalog_author[train_items[train_pos_rows]]
    static_affinity = AuthorAffinity(
        train.categorical[train_pos_rows, 0], static_author_ids, n_authors
    )

    variants: dict[str, float | None] = {"history_lifetime_asof": None}
    for days in args.lookbacks_days:
        variants[f"history_{days:g}d"] = float(days)
    variant_names = ["static_train_author", *variants.keys()]
    states = {
        name: {k: _state(q) for k in args.candidate_budgets}
        for name in variant_names
    }
    model_hit_by_budget = {k: np.zeros(q, dtype=bool) for k in args.candidate_budgets}
    pool_sizes: list[int] = []

    for window in np.unique(query_windows):
        idx = np.flatnonzero(query_windows == window)
        pool = pools.get(int(window)).astype(np.int64, copy=False)
        if args.pool_cap and pool.size > args.pool_cap:
            pool = np.random.default_rng(args.seed + int(window) + 1).choice(pool, size=args.pool_cap, replace=False)
        pool = np.union1d(pool, positives[idx]).astype(np.int64, copy=False)
        pool_sizes.append(int(pool.size))
        if np.any(pool <= 0) or np.any(pool > n_catalog):
            raise ValueError(f"Window {window} contains a catalogue item outside the index")

        pool_authors = catalog_author[pool]
        unique_pool_authors, author_inverse = np.unique(pool_authors, return_inverse=True)
        unique_users, user_inverse = np.unique(users[idx], return_inverse=True)
        known_by_user = known.grid(unique_users, pool)
        print(f"  window {int(window)}: {idx.size:,} queries against {pool.size:,} active videos", flush=True)

        for offset in range(0, idx.size, args.query_block):
            local_indices = idx[offset:offset + args.query_block]
            count = local_indices.size
            model_scores = user_vectors[local_indices] @ np.asarray(embeddings[pool - 1], dtype=np.float32).T
            known_block = known_by_user[user_inverse[offset:offset + count]]
            candidate_valid = ~known_block.copy()
            target_cols = np.searchsorted(pool, positives[local_indices])
            candidate_valid[np.arange(count), target_cols] = True

            for row_pos, query_pos in enumerate(local_indices):
                global_idx = int(query_pos)
                eligible = np.flatnonzero(candidate_valid[row_pos])
                if eligible.size == 0:
                    continue
                target_col = int(target_cols[row_pos])
                user = int(users[global_idx])
                time_ms = int(query_times[global_idx])
                model_row = model_scores[row_pos]

                # Compute the Model 1 top-K once; it is identical for all history variants.
                model_cols = {
                    k: select_topk_columns(model_row, eligible, k)
                    for k in args.candidate_budgets
                }
                static_scores = static_affinity.scores(user, pool_authors)

                creator_video_scores: dict[str, np.ndarray] = {
                    "static_train_author": static_scores,
                }
                creator_video_scores.update({
                    name: history.scores(
                        user_index=user,
                        query_time_ms=time_ms,
                        candidate_author_ids=unique_pool_authors,
                        lookback_days=days,
                    )[author_inverse]
                    for name, days in variants.items()
                })

                for name, score_vector in creator_video_scores.items():
                    for k in args.candidate_budgets:
                        m_cols = model_cols[k]
                        a_cols = select_creator_topk(
                            score_vector, model_row, eligible, pool, k
                        )
                        model_found = bool(np.any(m_cols == target_col))
                        author_found = bool(np.any(a_cols == target_col))
                        union = np.union1d(m_cols, a_cols)
                        state = states[name][k]
                        state["model_hit"][global_idx] = model_found
                        state["creator_hit"][global_idx] = author_found
                        state["union_hit"][global_idx] = model_found or author_found
                        state["creator_only"][global_idx] = author_found and not model_found
                        state["model_only"][global_idx] = model_found and not author_found
                        state["both"][global_idx] = author_found and model_found
                        state["creator_size"][global_idx] = a_cols.size
                        state["union_size"][global_idx] = union.size
                        state["overlap_size"][global_idx] = np.intersect1d(m_cols, a_cols).size
                        if name != "static_train_author":
                            state["has_history_creator_in_pool"][global_idx] = bool(
                                np.any(score_vector[eligible] > 0)
                            )
                        else:
                            state["has_history_creator_in_pool"][global_idx] = bool(
                                np.any(score_vector[eligible] > 0)
                            )

            del model_scores

    report_variants = {}
    for name in variant_names:
        report_variants[name] = {}
        for k in args.candidate_budgets:
            report_variants[name][str(k)] = _summarise_state(
                states[name][k], users, cold, args.bootstrap, args.seed
            )

    paired_vs_static = {}
    for name in variants:
        paired_vs_static[name] = {}
        for k in args.candidate_budgets:
            current = states[name][k]
            baseline = states["static_train_author"][k]
            paired_vs_static[name][str(k)] = {
                "creator_hit_delta": _bootstrap(
                    current["creator_hit"].astype(np.float64) - baseline["creator_hit"].astype(np.float64),
                    users, args.bootstrap, args.seed,
                ),
                "union_hit_delta": _bootstrap(
                    current["union_hit"].astype(np.float64) - baseline["union_hit"].astype(np.float64),
                    users, args.bootstrap, args.seed,
                ),
            }

    report = {
        "split": args.split,
        "data_dir": str(data_dir),
        "history_dir": str(history_dir),
        "history_artifact_metadata": history_manifest,
        "model_path": str(model_path),
        "index_dir": str(index_dir),
        "queries": q,
        "unique_users": int(np.unique(users).size),
        "cold_start_query_fraction": float(cold.mean()),
        "pool_hours": args.pool_hours,
        "pool_sizes": pool_sizes,
        "candidate_budget_definition": "Per source top-K; Model 1 and creator-history candidates are deduplicated, so the union can contain up to 2K.",
        "creator_history_asof_policy": "Only events with event_time_ms < query_time_ms are eligible. This supports chronological replay using earlier observed validation/test events without exposing same-time or future events.",
        "pool_lookahead_caveat": "The 24-hour active pool is built from all exposures in each validation window, including exposures later than an individual query timestamp. This matches the existing evaluator but is optimistic versus a strict live-as-of catalogue.",
        "creator_tie_break": "Creator score descending, then Model 1 similarity descending, then catalogue item index ascending.",
        "variants": report_variants,
        "paired_differences_vs_static_train_author": paired_vs_static,
        "elapsed_seconds": time.perf_counter() - started,
    }

    print("\n=== CREATOR HISTORY CANDIDATE UNION ===")
    print("Percentages are query recovery; unions combine Model 1 top-K with creator top-K.")
    for name in variant_names:
        print(f"\n{name}")
        for k in args.candidate_budgets:
            m = report_variants[name][str(k)]
            print(
                f"  K={k}: creator={m['creator_hit']['value']:.3f} "
                f"union={m['union_hit']['value']:.3f} "
                f"creator-only={m['creator_only']['value']:.3f} "
                f"model-only={m['model_only']['value']:.3f} "
                f"mean union size={m['mean_union_candidates']:.1f} "
                f"prior creator present={m['queries_with_prior_creator_in_pool']:.3f}"
            )
            if name != "static_train_author":
                d = paired_vs_static[name][str(k)]
                print(
                    f"       vs static author: union delta={d['union_hit_delta']['value']:+.4f} "
                    f"CI=[{d['union_hit_delta']['ci95_user_cluster'][0]:+.4f}, "
                    f"{d['union_hit_delta']['ci95_user_cluster'][1]:+.4f}]"
                )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=True, default=float), encoding="utf-8")
        print(f"\nSaved: {out.resolve()}")
    print("\nCreator-history evaluation complete.")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v3_catalog")
    parser.add_argument("--history-dir", default="data/processed/history/creator_history_v1")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--pool-hours", type=float, default=24.0)
    parser.add_argument("--pool-cap", type=int, default=0)
    parser.add_argument("--query-block", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--lookbacks-days", default="1,3,7,14")
    parser.add_argument("--candidate-budgets", default="1000,3000")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", default="reports/experiments/creator_history_validation.json")
    args = parser.parse_args()
    try:
        args.lookbacks_days = sorted({float(x.strip()) for x in args.lookbacks_days.split(",") if x.strip()})
        args.candidate_budgets = sorted({int(x.strip()) for x in args.candidate_budgets.split(",") if x.strip()})
    except ValueError as exc:
        parser.error(f"Invalid lookback or candidate budget: {exc}")
    if any((not np.isfinite(x) or x <= 0) for x in args.lookbacks_days):
        parser.error("--lookbacks-days values must be finite and positive")
    if not args.candidate_budgets or any(x <= 0 for x in args.candidate_budgets):
        parser.error("--candidate-budgets must contain positive integers")
    if args.queries < 1 or args.query_block < 1 or args.bootstrap < 1:
        parser.error("queries, query-block and bootstrap must be positive")
    if args.split == "test" and not args.allow_test:
        parser.error("Held-out test requires --allow-test")
    return args


if __name__ == "__main__":
    run(parse_args())
