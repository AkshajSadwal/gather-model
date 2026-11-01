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

    def __init__(self, users: np.ndarray, authors: np.ndarray, n_authors: int):
        keys, counts = np.unique(users.astype(np.int64) * n_authors + authors.astype(np.int64), return_counts=True)
        self.n_authors = int(n_authors)
        self.user_of = keys // self.n_authors
        self.author_of = keys % self.n_authors
        self.counts = counts.astype(np.float32)

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
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
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
    train_pos_rows, _ = positive_rows(train, metadata)
    known = KnownPairs(train.categorical[train_pos_rows, 0], train_items[train_pos_rows], int(metadata["num_users"]), n_items)
    catalog_author = catalog.categorical[:, 1].astype(np.int64)
    affinity = AuthorAffinity(train.categorical[train_pos_rows, 0], catalog_author[train_items[train_pos_rows]],
                              int(metadata["categorical_sizes"][AUTHOR_FIELD_INDEX]))
    all_windows = np.concatenate([train_windows, split_windows])
    all_items = np.concatenate([train_items, split_items])

    methods = ["model", "shuffled users", "popularity (previous window)", "author affinity"]
    ranks = {m: np.empty(q, dtype=np.float64) for m in methods}
    competitors = np.empty(q, dtype=np.int64)
    pool_sizes = []
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
            cohorts[label] = {"queries": int(mask.sum()),
                              "model_hit_at_5pct": float(np.mean(percentile["model"][mask] <= 0.05)),
                              "model_mean_percentile": float(np.mean(percentile["model"][mask])),
                              "author_affinity_hit_at_5pct": float(np.mean(percentile["author affinity"][mask] <= 0.05)),
                              "popularity_hit_at_5pct": float(np.mean(percentile["popularity (previous window)"][mask] <= 0.05))}
    report = {
        "model_path": str(model_path), "index_dir": str(index_dir), "split": args.split,
        "checkpoint_best_epoch": checkpoint.get("best_epoch"), "model_version": checkpoint.get("model_version", 1),
        "queries": int(q), "unique_users": int(np.unique(users).size), "pool_hours": args.pool_hours,
        "pool_sizes": pool_sizes, "pct_shuffled_queries_with_same_user": 100.0 * shuffled_same_user,
        "results": results, "paired_differences": paired, "cohorts": cohorts,
        "elapsed_seconds": time.perf_counter() - started,
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
    print("\nCold-start split:")
    for label, c in cohorts.items():
        print(f"  {label:<40} n={c['queries']:<6} model hit@5%={c['model_hit_at_5pct']:.3f}  author affinity={c['author_affinity_hit_at_5pct']:.3f}  "
              f"popularity={c['popularity_hit_at_5pct']:.3f}  model mean pct={c['model_mean_percentile']:.3f}")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=True, default=float), encoding="utf-8")
        print(f"\nSaved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
