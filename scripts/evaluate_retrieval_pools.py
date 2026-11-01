"""Evaluate retrieval the way a feed uses it: rank each positive against the day's active pool.

The full-catalogue test ranks a positive against 4.4M videos, most of which were old or not
yet uploaded when the user's request happened. Here each held-out positive competes only
with the videos that were actually shown to someone in the same time window (default 24 h),
minus the user's own known training positives. Results are percentiles, so 0.5 means "no
better than chance" whatever the pool size.

Compared on identical queries:
  model             the checkpoint under test
  shuffled users    same model, another user's vector (is it really personalised?)
  popularity        exposures in the PREVIOUS window (known at request time, no leakage)
  author affinity   how often the user engaged with this video's author in training
                    (a simple non-neural recommender the model must beat)
  random            0.5 mean percentile by definition

Needs row_times.npz (python -m scripts.add_row_times) and an index built from the checkpoint:

    python -m model1_retrieval.build_index --model-path <best_model.pt> --output-dir data/indexes/retrieval/<name>
    python -m scripts.evaluate_retrieval_pools --model-path <best_model.pt> --index-dir data/indexes/retrieval/<name> \
        --output reports/experiments/pool_eval_<name>.json

Validation is the default; the test split needs --allow-test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.eval_metrics import cluster_bootstrap_mean_ci
from common.pools import (
    KnownPairs, WindowPools, expected_ranks_in_pool, load_row_times, percentile_from_ranks, window_ids,
)
from model1_retrieval.data import (
    build_video_catalog, data_fingerprint, item_catalog_indices, load_metadata, load_split, positive_rows,
)
from model1_retrieval.model import TwoTowerRetrieval

AUTHOR_FIELD_INDEX = 29     # categorical field 29 is video:author_id in the prepared schema


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


def different_user_permutation(users: np.ndarray, rng: np.random.Generator, tries: int = 50) -> np.ndarray:
    """A permutation of query positions that pairs (almost) every query with another user's vector."""
    perm = rng.permutation(users.size)
    for _ in range(tries):
        clash = np.flatnonzero(users[perm] == users)
        if clash.size == 0:
            break
        perm[clash] = rng.permutation(perm[clash])
    return perm


class AuthorAffinity:
    """Per-user counts of training positives per author, looked up for any list of authors."""

    def __init__(self, users: np.ndarray, authors: np.ndarray, n_authors: int,
                 weights: np.ndarray | None = None):
        users = np.asarray(users, dtype=np.int64)
        authors = np.asarray(authors, dtype=np.int64)
        if users.shape != authors.shape:
            raise ValueError("users and authors must have matching shapes")
        if weights is None:
            weights = np.ones(users.size, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != users.shape or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("author-affinity weights must be finite, nonnegative and match users")
        self.n_authors = int(n_authors)
        if self.n_authors < 1 or np.any(authors < 0) or np.any(authors >= self.n_authors):
            raise ValueError("author indices must be within the author vocabulary")
        keys = users * self.n_authors + authors
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        unique_keys, starts = np.unique(sorted_keys, return_index=True)
        totals = np.add.reduceat(weights[order], starts) if starts.size else np.empty(0)
        self.user_of = unique_keys // self.n_authors
        self.author_of = unique_keys % self.n_authors
        self.counts = totals.astype(np.float32)

    def scores(self, user: int, pool_authors: np.ndarray) -> np.ndarray:
        start = np.searchsorted(self.user_of, user, side="left")
        stop = np.searchsorted(self.user_of, user, side="right")
        out = np.zeros(pool_authors.size, dtype=np.float32)
        if stop == start:
            return out
        authors, counts = self.author_of[start:stop], self.counts[start:stop]   # authors ascending within a user
        at = np.minimum(np.searchsorted(authors, pool_authors), authors.size - 1)
        hit = (authors[at] == pool_authors) & (pool_authors > 0)
        out[hit] = counts[at[hit]]
        return out


def summarise(name: str, percentile: np.ndarray, ranks: np.ndarray, users: np.ndarray, n_boot: int, seed: int) -> dict:
    entry: dict = {"method": name, "queries": int(percentile.size)}
    for label, values in (("hit_at_1pct", percentile <= 0.01), ("hit_at_5pct", percentile <= 0.05),
                          ("hit_at_20pct", percentile <= 0.20), ("mean_percentile", percentile),
                          ("recall_at_100", ranks <= 100), ("recall_at_1000", ranks <= 1000)):
        mean, lo, hi = cluster_bootstrap_mean_ci(values.astype(np.float64), users, n_boot=n_boot, seed=seed)
        entry[label] = {"value": mean, "ci95_user_cluster": [lo, hi]}
    return entry


def select_topk_columns(scores: np.ndarray, eligible: np.ndarray, k: int,
                        require_positive: bool = False) -> np.ndarray:
    """Select up to K pool-column indices; optionally omit zero/negative author scores."""
    eligible = np.asarray(eligible, dtype=np.int64)
    scores = np.asarray(scores)
    if require_positive:
        eligible = eligible[scores[eligible] > 0]
    keep = min(int(k), eligible.size)
    if keep <= 0:
        return np.empty(0, dtype=np.int64)
    if keep == eligible.size:
        return eligible
    local = np.argpartition(scores[eligible], -keep)[-keep:]
    return eligible[local]


def empty_candidate_state(num_queries: int) -> dict[str, np.ndarray]:
    """Allocate per-query candidate-union measurements for one author scorer."""
    return {
        "model_hit": np.zeros(num_queries, dtype=bool),
        "author_hit": np.zeros(num_queries, dtype=bool),
        "union_hit": np.zeros(num_queries, dtype=bool),
        "model_only": np.zeros(num_queries, dtype=bool),
        "author_only": np.zeros(num_queries, dtype=bool),
        "both": np.zeros(num_queries, dtype=bool),
        "union_size": np.zeros(num_queries, dtype=np.int32),
        "overlap_size": np.zeros(num_queries, dtype=np.int32),
        "model_size": np.zeros(num_queries, dtype=np.int32),
        "author_size": np.zeros(num_queries, dtype=np.int32),
    }


def record_candidate_state(
    state: dict[str, np.ndarray],
    query_index: int,
    model_columns: np.ndarray,
    author_columns: np.ndarray,
    target_column: int,
) -> None:
    """Record a model+author candidate union for one query."""
    model_found = bool(np.any(model_columns == target_column))
    author_found = bool(np.any(author_columns == target_column))
    state["model_hit"][query_index] = model_found
    state["author_hit"][query_index] = author_found
    state["union_hit"][query_index] = model_found or author_found
    state["model_only"][query_index] = model_found and not author_found
    state["author_only"][query_index] = author_found and not model_found
    state["both"][query_index] = model_found and author_found
    state["union_size"][query_index] = np.union1d(model_columns, author_columns).size
    state["overlap_size"][query_index] = np.intersect1d(
        model_columns, author_columns, assume_unique=False
    ).size
    state["model_size"][query_index] = model_columns.size
    state["author_size"][query_index] = author_columns.size


def summarise_candidate_states(
    candidate_states: dict[int, dict[str, np.ndarray]],
    users: np.ndarray,
    cold: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, dict]:
    """Summarise candidate-union states without changing the legacy report schema."""
    candidate_report = {}
    for budget, state in candidate_states.items():
        summary = {}
        for label, values in (
            ("model_hit_at_k", state["model_hit"]),
            ("author_hit_at_k", state["author_hit"]),
            ("union", state["union_hit"]),
            ("model_unique_recovery", state["model_only"]),
            ("author_unique_recovery", state["author_only"]),
            ("both_sources_recovered", state["both"]),
        ):
            mean, lo, hi = cluster_bootstrap_mean_ci(
                values.astype(np.float64), users, n_boot=n_boot, seed=seed
            )
            summary[label] = {"value": mean, "ci95_user_cluster": [lo, hi]}
        summary["neither_source_recovered"] = float(np.mean(~state["union_hit"]))
        summary["mean_model_candidates"] = float(np.mean(state["model_size"]))
        summary["mean_author_candidates"] = float(np.mean(state["author_size"]))
        summary["mean_union_candidates"] = float(np.mean(state["union_size"]))
        summary["mean_overlap_candidates"] = float(np.mean(state["overlap_size"]))
        cohort_summary = {}
        for cohort_name, cohort_mask in (("cold_start", cold), ("seen_id", ~cold)):
            if cohort_mask.any():
                cohort_summary[cohort_name] = {
                    "queries": int(cohort_mask.sum()),
                    "model_hit_at_k": float(np.mean(state["model_hit"][cohort_mask])),
                    "author_hit_at_k": float(np.mean(state["author_hit"][cohort_mask])),
                    "union_hit_at_k": float(np.mean(state["union_hit"][cohort_mask])),
                    "model_unique_recovery": float(np.mean(state["model_only"][cohort_mask])),
                    "author_unique_recovery": float(np.mean(state["author_only"][cohort_mask])),
                    "both_sources_recovered": float(np.mean(state["both"][cohort_mask])),
                }
        candidate_report[str(budget)] = {
            "per_source_top_k": budget,
            "maximum_union_size": 2 * budget,
            "metrics": summary,
            "cohorts": cohort_summary,
        }
    return candidate_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v3_catalog")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true", help="Required to read the held-out test split")
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--pool-hours", type=float, default=24.0)
    parser.add_argument("--pool-cap", type=int, default=0, help="0 = use every video of the window; otherwise a random subset")
    parser.add_argument("--query-block", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--author-recency-half-life-days", type=float, default=7.0,
                        help="Half-life for an additional recency-weighted author-affinity baseline; <=0 disables it")
    parser.add_argument("--candidate-budgets", type=str, default="100,500,1000",
                        help="Comma-separated per-source top-K budgets for the V3/author candidate-union experiment")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    try:
        candidate_budgets = sorted({int(value.strip()) for value in args.candidate_budgets.split(",") if value.strip()})
    except ValueError:
        parser.error("--candidate-budgets must be comma-separated positive integers, e.g. 100,500,1000")
    if not candidate_budgets or any(value <= 0 for value in candidate_budgets):
        parser.error("--candidate-budgets must contain positive integers")
    if args.split == "test" and not args.allow_test:
        parser.error("Reading the held-out test split needs --allow-test. Tune on validation, report on test once.")

    started = time.perf_counter()
    data_dir, model_path, index_dir = Path(args.data_dir), Path(args.model_path), Path(args.index_dir)
    metadata = load_metadata(data_dir)
    schema = data_fingerprint(data_dir)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    index_metadata = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
    if checkpoint.get("dataset") != metadata.get("dataset") or index_metadata.get("dataset") != metadata.get("dataset"):
        raise ValueError("Model, index and prepared dataset do not match")
    if checkpoint.get("data_fingerprint") != schema or index_metadata.get("data_fingerprint") != schema:
        raise ValueError("Prepared metadata/ID maps differ from the model or index; rebuild them for this data version")
    if index_metadata.get("checkpoint_sha256") != sha256_file(model_path):
        raise ValueError("This index was built from a different checkpoint. Rebuild it with model1_retrieval.build_index")
    catalog_map_path = data_dir / "catalog_maps.npz"
    if index_metadata.get("catalog_maps_sha256") != (sha256_file(catalog_map_path) if catalog_map_path.exists() else None):
        raise ValueError("The retrieval index was built from a different video catalogue map. Rebuild the index.")

    embeddings = np.load(index_dir / index_metadata["files"]["embeddings"], mmap_mode="r")
    n_catalog = int(embeddings.shape[0])
    n_items = n_catalog + 1                              # catalogue indices are 1-based
    print(f"Index: {n_catalog:,} videos x {embeddings.shape[1]} dims")

    print("Loading splits and row times...")
    train = load_split(data_dir, "train")
    split = load_split(data_dir, args.split)
    train_times, split_times = load_row_times(data_dir, "train"), load_row_times(data_dir, args.split)
    if train_times.size != train.rows or split_times.size != split.rows:
        raise ValueError("row_times.npz does not match the prepared splits; recreate it with scripts.add_row_times")
    catalog = build_video_catalog(data_dir, metadata)
    origin_ms = int(train_times.min())
    train_items, split_items = item_catalog_indices(train), item_catalog_indices(split)
    train_windows = window_ids(train_times, origin_ms, args.pool_hours)
    split_windows = window_ids(split_times, origin_ms, args.pool_hours)
    pools = WindowPools.build(split_windows, split_items)

    rows, _ = positive_rows(split, metadata)
    rows = rows[(split_items[rows] > 0) & (split_items[rows] <= n_catalog)]
    rng = np.random.default_rng(args.seed)
    if rows.size > args.queries:
        rows = np.sort(rng.choice(rows, size=args.queries, replace=False))
    q = rows.size
    users = split.categorical[rows, 0].astype(np.int64)
    positives = split_items[rows].astype(np.int64)
    query_windows = split_windows[rows]
    cold = split.categorical[rows, 1] == 0
    print(f"{q:,} queries from {np.unique(users).size:,} users; {cold.mean():.1%} are on a video unseen in training")

    print("Encoding user queries...")
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
    perm = different_user_permutation(users, rng)
    shuffled_vectors = user_vectors[perm]
    shuffled_same_user = float(np.mean(users[perm] == users))

    print("Building baselines (known positives, author affinity)...")
    train_pos_rows, train_pos_weights = positive_rows(train, metadata)
    known = KnownPairs(train.categorical[train_pos_rows, 0], train_items[train_pos_rows], int(metadata["num_users"]), n_items)
    catalog_author = catalog.categorical[:, 1].astype(np.int64)
    author_ids = catalog_author[train_items[train_pos_rows]]
    n_authors = int(metadata["categorical_sizes"][AUTHOR_FIELD_INDEX])
    affinity = AuthorAffinity(train.categorical[train_pos_rows, 0], author_ids, n_authors)
    recency_affinity = None
    if args.author_recency_half_life_days > 0:
        half_life_ms = args.author_recency_half_life_days * 24.0 * 60.0 * 60.0 * 1000.0
        age_ms = np.maximum(int(train_times.max()) - train_times[train_pos_rows], 0)
        recency_weights = train_pos_weights * np.exp2(-age_ms / half_life_ms)
        recency_affinity = AuthorAffinity(train.categorical[train_pos_rows, 0], author_ids,
                                          n_authors, weights=recency_weights)
    all_windows = np.concatenate([train_windows, split_windows])
    all_items = np.concatenate([train_items, split_items])

    methods = ["model", "shuffled users", "popularity (previous window)", "author affinity"]
    if recency_affinity is not None:
        methods.append(f"author affinity (weighted, {args.author_recency_half_life_days:g}d half-life)")
    ranks = {m: np.empty(q, dtype=np.float64) for m in methods}
    competitors = np.empty(q, dtype=np.int64)
    pool_sizes = []
    # Candidate-generation diagnostics: each source receives its own top-K budget.
    # The union can therefore contain up to 2K candidates before deduplication.
    candidate_state = {k: empty_candidate_state(q) for k in candidate_budgets}
    # Evaluate recency-weighted affinity in actual top-K candidate generation too,
    # not only as a full-pool scoring baseline.
    weighted_candidate_state = (
        {k: empty_candidate_state(q) for k in candidate_budgets}
        if recency_affinity is not None else None
    )
    for window in np.unique(query_windows):
        idx = np.flatnonzero(query_windows == window)
        pool = pools.get(int(window)).astype(np.int64)
        if args.pool_cap and pool.size > args.pool_cap:
            pool = np.random.default_rng(args.seed + int(window) + 1).choice(pool, size=args.pool_cap, replace=False)
        pool = np.union1d(pool, positives[idx])
        pool_sizes.append(int(pool.size))
        pos_cols = np.searchsorted(pool, positives[idx])
        vectors = np.asarray(embeddings[pool - 1], dtype=np.float32)
        previous = all_items[all_windows == int(window) - 1]
        popularity = np.bincount(previous, minlength=n_items).astype(np.float32)[pool]
        pool_authors = catalog_author[pool]
        unique_users, inverse = np.unique(users[idx], return_inverse=True)
        known_by_user = known.grid(unique_users, pool)
        print(f"  window {int(window)}: {idx.size:,} queries against {pool.size:,} videos", flush=True)
        for a in range(0, idx.size, args.query_block):
            block = slice(a, min(a + args.query_block, idx.size))
            sel = idx[block]
            count = sel.size
            valid = ~known_by_user[inverse[block]]
            valid[np.arange(count), pos_cols[block]] = False
            score_sets = {
                "model": user_vectors[sel] @ vectors.T,
                "shuffled users": shuffled_vectors[sel] @ vectors.T,
                "popularity (previous window)": np.broadcast_to(popularity, (count, pool.size)),
                "author affinity": np.stack([affinity.scores(int(u), pool_authors) for u in users[sel]]),
            }
            author_scores = score_sets["author affinity"]
            model_scores = score_sets["model"]
            recency_scores = None
            if recency_affinity is not None:
                recency_name = f"author affinity (weighted, {args.author_recency_half_life_days:g}d half-life)"
                recency_scores = np.stack([
                    recency_affinity.scores(int(u), pool_authors) for u in users[sel]
                ])
                score_sets[recency_name] = recency_scores
            # For candidate generation, exclude the user's training positives but retain
            # this held-out target. The ranking-percentile mask above excludes the target
            # only because it is the item being ranked against competitors.
            candidate_valid = known_by_user[inverse[block]].copy()
            candidate_valid[np.arange(count), pos_cols[block]] = False
            candidate_valid = ~candidate_valid
            candidate_valid[np.arange(count), pos_cols[block]] = True

            for budget in candidate_budgets:
                state = candidate_state[budget]
                for local_row, global_query in enumerate(sel):
                    eligible = np.flatnonzero(candidate_valid[local_row])
                    if eligible.size == 0:
                        continue
                    m_cols = select_topk_columns(model_scores[local_row], eligible, budget)
                    a_cols = select_topk_columns(
                        author_scores[local_row], eligible, budget, require_positive=True
                    )
                    target_col = int(pos_cols[block][local_row])
                    record_candidate_state(state, int(global_query), m_cols, a_cols, target_col)

                    if weighted_candidate_state is not None and recency_scores is not None:
                        weighted_a_cols = select_topk_columns(
                            recency_scores[local_row], eligible, budget, require_positive=True
                        )
                        record_candidate_state(
                            weighted_candidate_state[budget], int(global_query),
                            m_cols, weighted_a_cols, target_col
                        )
            for name, scores in score_sets.items():
                target = scores[np.arange(count), pos_cols[block]]
                r, comp = expected_ranks_in_pool(scores, target, valid)
                ranks[name][sel] = r
                competitors[sel] = comp

    percentile = {m: percentile_from_ranks(ranks[m], competitors) for m in methods}
    results = [summarise(m, percentile[m], ranks[m], users, args.bootstrap, args.seed) for m in methods]
    results.append({"method": "random (by definition)", "queries": int(q), "hit_at_1pct": {"value": 0.01},
                    "hit_at_5pct": {"value": 0.05}, "hit_at_20pct": {"value": 0.20}, "mean_percentile": {"value": 0.5}})
    paired = {}
    for other in methods[1:]:
        for label, fn in (("hit_at_5pct", lambda p: (p <= 0.05).astype(np.float64)), ("mean_percentile", lambda p: p)):
            m, lo, hi = cluster_bootstrap_mean_ci(fn(percentile["model"]) - fn(percentile[other]), users, n_boot=args.bootstrap, seed=args.seed)
            paired[f"model minus {other}: {label}"] = {"value": m, "ci95_user_cluster": [lo, hi]}
    cohorts = {}
    for label, mask in (("unseen training video ID (cold start)", cold), ("seen training video ID", ~cold)):
        if mask.any():
            cohort_users = users[mask]
            cohort_results = {}
            for method in methods:
                values = (percentile[method][mask] <= 0.05).astype(np.float64)
                mean, lo, hi = cluster_bootstrap_mean_ci(
                    values, cohort_users, n_boot=args.bootstrap, seed=args.seed
                )
                cohort_results[method] = {"hit_at_5pct": mean, "ci95_user_cluster": [lo, hi]}
            cohorts[label] = {"queries": int(mask.sum()), "unique_users": int(np.unique(cohort_users).size),
                              "methods": cohort_results,
                              "model_mean_percentile": float(np.mean(percentile["model"][mask]))}
    candidate_report = summarise_candidate_states(
        candidate_state, users, cold, args.bootstrap, args.seed
    )
    weighted_candidate_report = None
    if weighted_candidate_state is not None:
        weighted_candidate_report = summarise_candidate_states(
            weighted_candidate_state, users, cold, args.bootstrap, args.seed
        )
        for budget in candidate_budgets:
            base_state = candidate_state[budget]
            weighted_state = weighted_candidate_state[budget]
            paired_deltas = {}
            for metric_name, state_key in (
                ("author_recovery", "author_hit"),
                ("union_recovery", "union_hit"),
                ("author_unique_recovery", "author_only"),
            ):
                delta = (
                    weighted_state[state_key].astype(np.float64)
                    - base_state[state_key].astype(np.float64)
                )
                mean, lo, hi = cluster_bootstrap_mean_ci(
                    delta, users, n_boot=args.bootstrap, seed=args.seed
                )
                paired_deltas[metric_name] = {
                    "value": mean,
                    "ci95_user_cluster": [lo, hi],
                }
            union_size_delta = (
                weighted_state["union_size"].astype(np.float64)
                - base_state["union_size"].astype(np.float64)
            )
            mean, lo, hi = cluster_bootstrap_mean_ci(
                union_size_delta, users, n_boot=args.bootstrap, seed=args.seed
            )
            paired_deltas["mean_union_candidates_delta"] = {
                "value": mean,
                "ci95_user_cluster": [lo, hi],
            }
            weighted_candidate_report[str(budget)]["paired_vs_unweighted"] = paired_deltas

    report = {
        "model_path": str(model_path), "index_dir": str(index_dir), "split": args.split,
        "checkpoint_best_epoch": checkpoint.get("best_epoch"), "model_version": checkpoint.get("model_version", 1),
        "queries": int(q), "unique_users": int(np.unique(users).size), "pool_hours": args.pool_hours,
        "pool_sizes": pool_sizes, "pct_shuffled_queries_with_same_user": 100.0 * shuffled_same_user,
        "results": results, "paired_differences": paired, "cohorts": cohorts,
        "candidate_union_experiment": {
            "definition": "Per query, retrieve top-K eligible candidates independently from model scores and positive unweighted author-affinity scores, then deduplicate their union. Known training positives are excluded; the held-out target remains eligible.",
            "budget_note": "K is a per-source budget, so the deduplicated union can contain up to 2K candidates. Author-affinity candidates with zero affinity are not filled with arbitrary items.",
            "budgets": candidate_report,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }

    if weighted_candidate_report is not None:
        report["weighted_candidate_union_experiment"] = {
            "author_recency_half_life_days": float(args.author_recency_half_life_days),
            "definition": "Same model top-K candidates and same eligible active-pool slate as the unweighted experiment; only the author-affinity weights used to select author candidates differ.",
            "budgets": weighted_candidate_report,
        }

    print(f"\n=== Pool retrieval on {args.split}: {q:,} queries, pools of {min(pool_sizes):,}-{max(pool_sizes):,} videos "
          f"({args.pool_hours:g} h windows) ===")
    print(f"{'method':<30}{'hit@1%':>9}{'hit@5%':>9}{'hit@20%':>9}{'mean pct':>10}{'R@100':>9}{'R@1000':>9}   (mean pct: 0 best, 0.5 chance)")
    for row in results:
        def cell(key, width=9, digits=3):
            return f"{row[key]['value']:>{width}.{digits}f}" if key in row else " " * width
        print(f"{row['method']:<30}{cell('hit_at_1pct')}{cell('hit_at_5pct')}{cell('hit_at_20pct')}{cell('mean_percentile', 10)}"
              f"{cell('recall_at_100')}{cell('recall_at_1000')}")
    print("\nModel minus baseline (positive hit@5% difference or negative mean-percentile difference = model better), 95% CI over users:")
    for key, value in paired.items():
        print(f"  {key:<62}{value['value']:+.4f}  [{value['ci95_user_cluster'][0]:+.4f}, {value['ci95_user_cluster'][1]:+.4f}]")
    print("\nCold-start / seen-ID cohorts (Hit@5%, user-cluster 95% CI):")
    for label, cohort in cohorts.items():
        print(f"  {label}: queries={cohort['queries']:,}, users={cohort['unique_users']:,}")
        for method, values in cohort["methods"].items():
            lo, hi = values["ci95_user_cluster"]
            print(f"    {method:<48} {values['hit_at_5pct']:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"    model mean percentile={cohort['model_mean_percentile']:.3f}")
    print("\n=== Candidate union: per-source top-K; union can be up to 2K ===")
    for budget, entry in candidate_report.items():
        m = entry["metrics"]
        print(
            f"  K={budget:>4}: model={m['model_hit_at_k']['value']:.3f} "
            f"author={m['author_hit_at_k']['value']:.3f} union={m['union']['value']:.3f} "
            f"model-only recovery={m['model_unique_recovery']['value']:.3f} "
            f"author-only recovery={m['author_unique_recovery']['value']:.3f} "
            f"mean union size={m['mean_union_candidates']:.1f}"
        )
        for cohort_name, cohort in entry["cohorts"].items():
            print(
                f"       {cohort_name}: model={cohort['model_hit_at_k']:.3f} "
                f"author={cohort['author_hit_at_k']:.3f} union={cohort['union_hit_at_k']:.3f}"
            )
    if weighted_candidate_report is not None:
        print(
            f"\n=== Weighted author candidate union "
            f"({args.author_recency_half_life_days:g}d half-life) ==="
        )
        print("Paired weighted-minus-unweighted differences (user-cluster 95% CI):")
        for budget, entry in weighted_candidate_report.items():
            metrics = entry["metrics"]
            paired_delta = entry["paired_vs_unweighted"]
            union_ci = paired_delta["union_recovery"]["ci95_user_cluster"]
            author_ci = paired_delta["author_recovery"]["ci95_user_cluster"]
            print(
                f"  K={budget}: weighted author={metrics['author_hit_at_k']['value']:.3f}, "
                f"weighted union={metrics['union']['value']:.3f}; "
                f"union delta={paired_delta['union_recovery']['value']:+.4f} "
                f"[{union_ci[0]:+.4f}, {union_ci[1]:+.4f}], "
                f"author delta={paired_delta['author_recovery']['value']:+.4f} "
                f"[{author_ci[0]:+.4f}, {author_ci[1]:+.4f}]"
            )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=True, default=float), encoding="utf-8")
        print(f"\nSaved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
