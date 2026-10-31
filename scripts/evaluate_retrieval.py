"""Full-catalogue retrieval evaluation with honest baselines.

The training-time check (512 queries x 500 random negatives) is noisy and easy: positives
are popular videos while negatives are uniform random videos. This script instead ranks each
held-out positive against the WHOLE video catalogue, using the same float16 index that
serving uses, for thousands of queries, and reports 95% confidence intervals.

Three rows are compared on identical queries:
  * model            the checkpoint under test
  * shuffled users   the same model, but each query gets a different user's vector.
                     If this is close to "model", the score is mostly item popularity,
                     not personalisation.
  * popularity       rank by training engagement count (no model at all)

Build the index first (it must come from the same checkpoint):

    python -m model1_retrieval.build_index --model-path <best_model.pt> --output-dir data/indexes/retrieval/<name>
    python -m scripts.evaluate_retrieval --model-path <best_model.pt> --index-dir data/indexes/retrieval/<name> \
        --output reports/experiments/retrieval_eval_<name>.json

Validation is the default; the test split needs --allow-test. Known training positives are
not removed from the competing items (a standard "filtered" protocol would), so absolute
numbers are slightly pessimistic; comparisons between rows are fair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.eval_metrics import (
    cluster_bootstrap_mean_ci, expected_rank_from_scores, recall_hits, reciprocal_ranks, relevance_gains,
)
from common.retrieval_eval import rank_against_catalog
from model1_retrieval.data import data_fingerprint, item_catalog_indices, load_metadata, load_split
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


def summarise(name: str, ranks: np.ndarray, users: np.ndarray, ks: list[int],
              catalogue_size: int, n_boot: int, seed: int) -> dict:
    out: dict = {
        "method": name, "queries": int(ranks.size),
        "unique_users": int(np.unique(users).size),
        "median_rank": float(np.median(ranks)),
        "confidence_interval_method": "query-weighted user-cluster bootstrap",
    }
    for k in ks:
        mean, lo, hi = cluster_bootstrap_mean_ci(
            recall_hits(ranks, k), users, n_boot=n_boot, seed=seed + k
        )
        out[f"recall@{k}"] = {"value": mean, "ci95_user_cluster": [lo, hi],
                             "chance": k / catalogue_size}
    mrr, lo, hi = cluster_bootstrap_mean_ci(
        reciprocal_ranks(ranks), users, n_boot=n_boot, seed=seed
    )
    out["mrr"] = {"value": mrr, "ci95_user_cluster": [lo, hi]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v2")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true", help="Required to read the held-out test split")
    parser.add_argument("--queries", type=int, default=5000, help="Number of held-out positive interactions to rank")
    parser.add_argument("--ks", type=int, nargs="+", default=[20, 100, 500, 1000])
    parser.add_argument("--query-block", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=131072)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Reading the held-out test split needs --allow-test. Tune on validation, report on test once.")

    started = time.perf_counter()
    data_dir, model_path, index_dir = Path(args.data_dir), Path(args.model_path), Path(args.index_dir)
    metadata = load_metadata(data_dir)
    targets = list(metadata["binary_targets"])
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
    current_catalog_hash = sha256_file(catalog_map_path) if catalog_map_path.exists() else None
    if index_metadata.get("catalog_maps_sha256") != current_catalog_hash:
        raise ValueError("The retrieval index was built from a different video catalogue map. Rebuild the index.")

    embeddings = np.load(index_dir / index_metadata["files"]["embeddings"], mmap_mode="r")
    catalogue_size = int(embeddings.shape[0])
    print(f"Index: {catalogue_size:,} videos x {embeddings.shape[1]} dims ({embeddings.dtype})")

    split = load_split(data_dir, args.split)
    gain = relevance_gains(split.labels, targets)
    user_index = split.categorical[:, 0]
    train_video_index = split.categorical[:, 1]
    catalog_index = item_catalog_indices(split)
    eligible = (gain > 0) & (user_index > 0)
    retrievable = eligible & (catalog_index > 0) & (catalog_index <= catalogue_size)
    unseen_training_video = eligible & (train_video_index == 0)
    cold_start_with_metadata = unseen_training_video & (catalog_index > 0)
    missing_catalogue_metadata = eligible & (catalog_index == 0)
    report: dict = {
        "model_path": str(model_path), "index_dir": str(index_dir), "split": args.split,
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "catalogue_size": catalogue_size,
        "eligible_positive_rows": int(eligible.sum()),
        "pct_eligible_positives_on_video_unseen_in_training": float(100.0 * unseen_training_video.sum() / max(eligible.sum(), 1)),
        "pct_eligible_positives_on_unseen_video_with_catalogue_metadata": float(100.0 * cold_start_with_metadata.sum() / max(eligible.sum(), 1)),
        "pct_eligible_positives_missing_from_basic_video_catalogue": float(100.0 * missing_catalogue_metadata.sum() / max(eligible.sum(), 1)),
        "retrievable_positive_rows": int(retrievable.sum()),
        "cold_start_positive_rows_with_metadata": int(cold_start_with_metadata.sum()),
        "positive_rows_without_catalogue_metadata": int(missing_catalogue_metadata.sum()),
    }
    print(f"{eligible.sum():,} eligible positive rows; {unseen_training_video.sum():,} "
          f"({report['pct_eligible_positives_on_video_unseen_in_training']:.2f}%) use an unseen training video ID; "
          f"{cold_start_with_metadata.sum():,} of those have metadata and can be indexed; "
          f"{missing_catalogue_metadata.sum():,} ({report['pct_eligible_positives_missing_from_basic_video_catalogue']:.2f}%) "
          "have no basic-video catalogue row and cannot be indexed")

    rng = np.random.default_rng(args.seed)
    candidates = np.flatnonzero(retrievable)
    rows = np.sort(rng.choice(candidates, size=min(args.queries, candidates.size), replace=False))
    q = rows.size
    report["queries"] = int(q)

    device = choose_device()
    print(f"Encoding {q:,} user queries on {device}...")
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

    # Shuffled users: a different query's user vector for each positive.
    shift = int(rng.integers(1, max(q, 2)))
    shuffle = np.roll(np.arange(q), shift)
    report["pct_shuffled_queries_with_same_user"] = float(100.0 * np.mean(user_index[rows] == user_index[rows][shuffle]))
    positive_rows0 = catalog_index[rows].astype(np.int64) - 1   # index row i holds catalogue item index i + 1

    def show(done: int, total: int) -> None:
        print(f"  scanned {done:,}/{total:,} catalogue videos", flush=True)

    print("Ranking positives against the full catalogue (exact scan)...")
    ranks = rank_against_catalog({"model": user_vectors, "shuffled users": user_vectors[shuffle]},
                                 positive_rows0, embeddings, chunk_size=args.chunk_size,
                                 query_block=args.query_block, progress=show)

    print("Loading train.npz for the popularity baseline...")
    train = load_split(data_dir, "train")
    train_catalog_indices = item_catalog_indices(train)
    popularity = np.bincount(train_catalog_indices, weights=relevance_gains(train.labels, targets),
                             minlength=catalogue_size + 1)[1:]
    del train
    popularity_ranks = expected_rank_from_scores(np.sort(popularity), popularity[positive_rows0])

    methods = {"model": ranks["model"], "shuffled users": ranks["shuffled users"], "popularity": popularity_ranks}
    query_users = user_index[rows].astype(np.int64, copy=False)
    report["results"] = [
        summarise(name, r, query_users, args.ks, catalogue_size, args.bootstrap, args.seed)
        for name, r in methods.items()
    ]

    print(f"\n=== Full-catalogue retrieval on {args.split} ({q:,} queries, {catalogue_size:,} videos) ===")
    print(f"{'method':<16}" + "".join(f"{'R@' + str(k):>22}" for k in args.ks) + f"{'MRR':>10}{'median rank':>14}")
    for row in report["results"]:
        cells = "".join(
            f"{row[f'recall@{k}']['value']:>10.4f} [{row[f'recall@{k}']['ci95_user_cluster'][0]:.3f},{row[f'recall@{k}']['ci95_user_cluster'][1]:.3f}]"
            for k in args.ks)
        print(f"{row['method']:<16}{cells}{row['mrr']['value']:>10.4f}{row['median_rank']:>14,.0f}")
    print("\nHow to read it: 'model' above 'popularity' means it adds value beyond popular videos; "
          "'model' far above 'shuffled users' means it really personalises.")

    report["elapsed_seconds"] = time.perf_counter() - started
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=True, default=float), encoding="utf-8")
        print(f"\nSaved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
