"""Model 1 retrieval trainer v3: learn to pick the right video from the *active pool*.

What is different from train.py (and why)
-----------------------------------------
* Negatives come from the same time as the positive. Training rows are cut into batches of
  consecutive-in-time positives, so the other users' positives in the batch are videos that
  were active at that moment, and extra negatives are drawn from the videos exposed in that
  day's pool. (train.py drew negatives uniformly from all 4.4M catalogue videos, which are
  mostly old or not yet uploaded, and so taught the model that rarely seen IDs are bad.)
* One softmax over ~8,000 candidates per batch instead of 2 sampled negatives per positive,
  with logQ correction so popular videos are not punished for being sampled often, and with
  the user's other known positives excluded rather than treated as negatives.
* Video IDs seen fewer than --id-min-count times in training are treated as unknown, and a
  further fraction (--id-dropout) is hidden at random. Brand-new videos (about half of the
  validation positives) then look like videos the model already knows how to score from
  content features. The setting is stored inside the checkpoint, so build_index, the
  evaluators and retrieve.py need no change.
* Checkpoints are selected on a time-aware pool metric (hit rate in the top 1% of the day's
  pool) instead of Recall@100 against 500 uniformly sampled old videos, a metric that rewards
  exactly the memorisation described above.

Needs row_times.npz (python -m scripts.add_row_times ...). Train from scratch; warm-starting
from a v1 checkpoint is not supported because its towers were fitted to memorised IDs.

    python -m model1_retrieval.train_v3 --data-dir data/processed/ranking/1k_v3_catalog \
        --output-dir artifacts/retrieval/1k_v3_pool_seed42 --epochs 8
"""
from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from common.data_version import prepared_data_fingerprint
from common.pools import (
    KnownPairs, WindowPools, build_softmax_batch, expected_ranks_in_pool, load_row_times,
    percentile_from_ranks, positive_frequency, time_local_batches, window_ids,
)
from model1_retrieval.data import (
    PreparedSplit, build_video_catalog, data_fingerprint, item_catalog_indices, load_metadata,
    load_split, positive_rows,
)
from model1_retrieval.losses import pool_softmax_loss
from model1_retrieval.model import TwoTowerRetrieval
from model1_retrieval.train import (
    atomic_save, capture_device_rng, choose_device, report_mps_memory, restore_device_rng,
    seed_everything, write_json,
)

VIDEO_COLS = TwoTowerRetrieval.VIDEO_CATALOG_COLUMNS
VIDEO_NUM_COLS = TwoTowerRetrieval.VIDEO_CATALOG_NUMERIC_COLUMNS


def encode_videos(model: TwoTowerRetrieval, catalog, item_ids: np.ndarray, device: torch.device) -> torch.Tensor:
    cat = torch.from_numpy(np.ascontiguousarray(catalog.categorical[item_ids][:, VIDEO_COLS])).to(device)
    num = torch.from_numpy(np.ascontiguousarray(catalog.numeric[item_ids][:, VIDEO_NUM_COLS])).to(device)
    return model.video_tower(cat, num)


def encode_users(model: TwoTowerRetrieval, split: PreparedSplit, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    cat = torch.from_numpy(np.ascontiguousarray(split.categorical[rows][:, TwoTowerRetrieval.USER_COLUMNS])).to(device)
    num = torch.from_numpy(np.ascontiguousarray(split.numeric[rows][:, TwoTowerRetrieval.USER_NUMERIC_COLUMNS])).to(device)
    return model.user_tower(cat, num)


class PoolValidator:
    """Fixed validation queries, each ranked against the videos exposed in its own time window.

    The pool of a window is every distinct video shown to any user in that window of the
    validation data, capped at ``pool_cap`` random videos (the queries' own positives are
    always kept). The user's known training positives are removed from the competitors.
    Percentiles make results comparable across pool sizes: 0 is first, 0.5 is chance.
    """

    def __init__(self, validation: PreparedSplit, validation_times: np.ndarray, metadata: dict[str, Any],
                 catalog, known: KnownPairs, origin_ms: int, window_hours: float,
                 queries: int, pool_cap: int, seed: int):
        rows, _ = positive_rows(validation, metadata)
        items = item_catalog_indices(validation)
        rows = rows[(items[rows] > 0) & (items[rows] < catalog.categorical.shape[0])]
        if rows.size == 0:
            raise ValueError("Validation split has no eligible positive interactions")
        rng = np.random.default_rng(seed)
        if rows.size > queries:
            rows = np.sort(rng.choice(rows, size=queries, replace=False))
        windows = window_ids(validation_times, origin_ms, window_hours)
        pools = WindowPools.build(windows, items)
        self.validation = validation
        self.rows = rows
        self.positives = items[rows].astype(np.int64)
        query_windows = windows[rows]
        users = validation.categorical[rows, 0].astype(np.int64)
        self.groups: list[dict[str, Any]] = []
        for window in np.unique(query_windows):
            query_idx = np.flatnonzero(query_windows == window)
            pool = pools.get(int(window)).astype(np.int64)
            if pool.size > pool_cap:
                keep = np.random.default_rng(seed + int(window) + 1).choice(pool, size=pool_cap, replace=False)
                pool = np.union1d(keep, self.positives[query_idx])
            else:
                pool = np.union1d(pool, self.positives[query_idx])
            unique_users, inverse = np.unique(users[query_idx], return_inverse=True)
            self.groups.append({
                "window": int(window), "query_idx": query_idx, "pool": pool.astype(np.int32),
                "pos_cols": np.searchsorted(pool, self.positives[query_idx]),
                "known_by_user": known.grid(unique_users, pool), "inverse": inverse,
            })
        sizes = [g["pool"].size for g in self.groups]
        self.description = (f"{rows.size:,} validation queries in {len(self.groups)} time windows of {window_hours:g} h; "
                            f"pool sizes {min(sizes):,}-{max(sizes):,}")

    @torch.no_grad()
    def evaluate(self, model: TwoTowerRetrieval, catalog, device: torch.device,
                 item_batch: int = 8192, query_block: int = 256) -> dict[str, float]:
        model.eval()
        user_vectors = np.empty((self.rows.size, 128), dtype=np.float32)
        for start in range(0, self.rows.size, 2048):
            chunk = self.rows[start:start + 2048]
            user_vectors[start:start + chunk.size] = encode_users(model, self.validation, chunk, device).cpu().numpy()
        ranks = np.empty(self.rows.size, dtype=np.float64)
        competitors = np.empty(self.rows.size, dtype=np.int64)
        for group in self.groups:
            pool = group["pool"]
            vectors = np.empty((pool.size, 128), dtype=np.float32)
            for start in range(0, pool.size, item_batch):
                vectors[start:start + item_batch] = encode_videos(model, catalog, pool[start:start + item_batch], device).cpu().numpy()
            query_idx, pos_cols = group["query_idx"], group["pos_cols"]
            for a in range(0, query_idx.size, query_block):
                block = slice(a, min(a + query_block, query_idx.size))
                scores = user_vectors[query_idx[block]] @ vectors.T
                count = scores.shape[0]
                target = scores[np.arange(count), pos_cols[block]]
                valid = ~group["known_by_user"][group["inverse"][block]]
                valid[np.arange(count), pos_cols[block]] = False
                block_ranks, block_competitors = expected_ranks_in_pool(scores, target, valid)
                ranks[query_idx[block]] = block_ranks
                competitors[query_idx[block]] = block_competitors
        percentile = percentile_from_ranks(ranks, competitors)
        return {
            "pool_hit_at_1pct": float(np.mean(percentile <= 0.01)),
            "pool_hit_at_5pct": float(np.mean(percentile <= 0.05)),
            "pool_mean_percentile": float(np.mean(percentile)),
            "pool_recall_at_100": float(np.mean(ranks <= 100)),
            "pool_recall_at_1000": float(np.mean(ranks <= 1000)),
            "pool_median_size": float(np.median(competitors + 1)),
            "queries": int(self.rows.size),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v3_catalog")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096, help="Positives per batch (also the in-batch negatives)")
    parser.add_argument("--shared-negatives", type=int, default=4096, help="Extra same-window pool negatives per batch")
    parser.add_argument("--pool-hours", type=float, default=24.0, help="Length of one time window / candidate pool")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--logq-strength", type=float, default=1.0, help="0 turns the popularity correction off")
    parser.add_argument("--id-min-count", type=int, default=20,
                        help="Video IDs seen fewer times in training are treated as unknown (0 keeps every ID)")
    parser.add_argument("--id-dropout", type=float, default=0.2, help="Extra random hiding of kept video IDs in training")
    parser.add_argument("--linear-output", action="store_true",
                        help="Drop the final ReLU/Dropout of both towers (stored in the checkpoint)")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-learning-rate", type=float, default=0.00005)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-positive-rows", type=int, default=0, help="Smoke test: random subset of positives")
    parser.add_argument("--eval-queries", type=int, default=4000)
    parser.add_argument("--eval-pool-cap", type=int, default=100000)
    parser.add_argument("--eval-seed", type=int, default=314159)
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.shared_negatives, args.eval_queries, args.eval_pool_cap) < 1:
        parser.error("epochs, batch size, shared negatives and evaluation sizes must be positive")
    if args.temperature <= 0 or args.pool_hours <= 0 or args.id_min_count < 0 or not 0 <= args.id_dropout < 1:
        parser.error("temperature and pool hours must be positive; id-min-count >= 0; 0 <= id-dropout < 1")
    if args.learning_rate <= 0 or not 0 <= args.min_learning_rate <= args.learning_rate:
        parser.error("learning rates must satisfy learning_rate > 0 and 0 <= min_learning_rate <= learning_rate")

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)

    data_dir, out = Path(args.data_dir), Path(args.output_dir)
    protected = ["best_model.pt", "latest_checkpoint.pt", "training_history.json", "run_metadata.json"]
    if out.exists() and not args.resume and any((out / name).exists() for name in protected):
        raise FileExistsError(f"{out} already contains a run. Choose a new --output-dir or use --resume.")
    out.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(data_dir)
    schema_fingerprint = data_fingerprint(data_dir)
    prepared_fingerprint = prepared_data_fingerprint(data_dir)
    print("Loading arrays, catalogue and row times...")
    train = load_split(data_dir, "train")
    validation = load_split(data_dir, "validation")
    catalog = build_video_catalog(data_dir, metadata)
    train_times = load_row_times(data_dir, "train")
    validation_times = load_row_times(data_dir, "validation")
    if train_times.size != train.rows or validation_times.size != validation.rows:
        raise ValueError("row_times.npz does not match the prepared splits; recreate it with scripts.add_row_times")
    origin_ms = int(train_times.min())
    n_items = int(catalog.categorical.shape[0])
    n_users = int(metadata["num_users"])

    train_items = item_catalog_indices(train)
    train_windows = window_ids(train_times, origin_ms, args.pool_hours)
    train_pools = WindowPools.build(train_windows, train_items)
    all_positive_idx, _ = positive_rows(train, metadata)
    positive_idx, positive_weights = positive_rows(train, metadata)
    if args.max_positive_rows and positive_idx.size > args.max_positive_rows:
        chosen = np.sort(np.random.default_rng(args.seed + 101).choice(positive_idx.size, size=args.max_positive_rows, replace=False))
        positive_idx, positive_weights = positive_idx[chosen], positive_weights[chosen]
        print(f"Smoke-test mode: {positive_idx.size:,} positive rows")
    print(f"Train rows {train.rows:,}; positives {positive_idx.size:,}; time windows of {args.pool_hours:g} h: "
          f"{train_pools.n_windows}; mean pool {np.diff(train_pools.ptr).mean():,.0f} videos")
    known = KnownPairs(train.categorical[all_positive_idx, 0], train_items[all_positive_idx], n_users, n_items)
    frequency = positive_frequency(train_items[positive_idx], n_items)

    exposures = np.bincount(train.categorical[:, 1], minlength=int(metadata["categorical_sizes"][1]))
    keep_ids = exposures >= max(args.id_min_count, 1)
    positive_on_kept = float(np.mean(keep_ids[train.categorical[positive_idx, 1]]))
    print(f"Video IDs with their own embedding: {int(keep_ids.sum()):,} of {int((exposures > 0).sum()):,} seen IDs "
          f"(exposures >= {max(args.id_min_count, 1)}); {positive_on_kept:.1%} of training positives are on such IDs")

    validator = PoolValidator(validation, validation_times, metadata, catalog, known, origin_ms,
                              args.pool_hours, args.eval_queries, args.eval_pool_cap, args.eval_seed)
    print("Validation:", validator.description)

    device = choose_device()
    print(f"Device: {device}")
    model = TwoTowerRetrieval(metadata, linear_output=args.linear_output).to(device)
    model.video_tower.set_id_keep(keep_ids)
    model.video_tower.id_dropout = float(args.id_dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_learning_rate)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"(video-ID table is mostly unused: only kept IDs receive gradients)")
    report_mps_memory(device, "after model initialization")

    best_path, latest_path, history_path = out / "best_model.pt", out / "latest_checkpoint.pt", out / "training_history.json"
    history: list[dict[str, Any]] = []
    best_score, best_epoch, no_improve, start_epoch = -float("inf"), 0, 0, 1
    resume_config = {k: getattr(args, k) for k in (
        "batch_size", "shared_negatives", "pool_hours", "temperature", "logq_strength", "id_min_count", "id_dropout",
        "learning_rate", "min_learning_rate", "weight_decay", "max_positive_rows", "eval_queries", "eval_pool_cap",
        "eval_seed", "epochs", "linear_output")}
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"Cannot resume: {latest_path} does not exist")
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if int(saved.get("seed", -1)) != args.seed or saved.get("prepared_data_fingerprint") != prepared_fingerprint:
            raise ValueError("Resume needs the same seed and the same prepared data")
        saved_resume_config = dict(saved.get("resume_config", {}))
        # Runs made before --linear-output existed are equivalent to False.
        saved_resume_config.setdefault("linear_output", False)
        if saved_resume_config != resume_config:
            raise ValueError(f"Resume settings differ from the checkpoint: saved={saved_resume_config} current={resume_config}")
        model.load_state_dict(saved["model_state_dict"])
        if (model.user_tower.linear_output != args.linear_output
                or model.video_tower.linear_output != args.linear_output):
            raise ValueError("Resume needs the same --linear-output setting as the saved run")
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch, best_epoch = int(saved["epoch"]) + 1, int(saved.get("best_epoch", 0))
        best_score, no_improve = float(saved.get("best_score", -float("inf"))), int(saved.get("no_improve", 0))
        history = list(saved.get("history", []))
        rng.bit_generator.state = saved["numpy_rng_state"]
        restore_device_rng(saved, device)
        print(f"Resumed after completed epoch {start_epoch - 1}")

    run_metadata = {
        "status": "running", "started_at_utc": started_at, "seed": args.seed, "dataset": metadata.get("dataset"),
        "data_dir": str(data_dir), "output_dir": str(out), "data_fingerprint": schema_fingerprint,
        "prepared_data_fingerprint": prepared_fingerprint, "model": "TwoTowerRetrieval v3 (pool softmax)",
        "config": vars(args) | {"data_dir": str(data_dir), "output_dir": str(out)},
        "kept_video_ids": int(keep_ids.sum()), "positive_share_on_kept_ids": positive_on_kept,
        "validation_protocol": validator.description + "; selection metric pool_hit_at_1pct",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": str(device)},
    }
    write_json(run_metadata, out / "run_metadata.json")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        batches = time_local_batches(positive_idx, args.batch_size, rng)
        loss_total, rows_seen = 0.0, 0
        progress = tqdm(batches, desc=f"Retrieval v3 epoch {epoch}/{args.epochs}")
        for rows in progress:
            users = train.categorical[rows, 0].astype(np.int64)
            positives = train_items[rows].astype(np.int64)
            weights = positive_weights[np.searchsorted(positive_idx, rows)]
            window = train_pools.nearest_nonempty(int(np.median(train_windows[rows])))
            if window is None:
                shared, pool_size = rng.integers(1, n_items, size=args.shared_negatives), n_items
            else:
                shared, pool_size = train_pools.sample(window, args.shared_negatives, rng), train_pools.size(window)
            candidates, own_cols, exclude, logq = build_softmax_batch(
                users, positives, shared, known, frequency, pool_size, args.logq_strength)

            optimizer.zero_grad(set_to_none=True)
            user_vec = encode_users(model, train, rows, device)
            cand_vec = encode_videos(model, catalog, candidates, device)
            loss, _ = pool_softmax_loss(
                user_vec, cand_vec,
                torch.from_numpy(own_cols).to(device),
                torch.from_numpy(exclude).to(device),
                torch.from_numpy(logq).to(device),
                args.temperature,
                torch.from_numpy(np.ascontiguousarray(weights)).to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.item()) * rows.size
            rows_seen += rows.size
            progress.set_postfix(loss=f"{loss.item():.4f}", cand=len(candidates))

        train_loss = loss_total / max(rows_seen, 1)
        metrics = validator.evaluate(model, catalog, device)
        report_mps_memory(device, f"after epoch {epoch} validation")
        seconds = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "train_loss": train_loss, "epoch_seconds": seconds,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]), "validation": metrics})
        print(f"\nEpoch {epoch}: loss={train_loss:.4f} | pool hit@1%={metrics['pool_hit_at_1pct']:.4f} "
              f"hit@5%={metrics['pool_hit_at_5pct']:.4f} mean percentile={metrics['pool_mean_percentile']:.4f} "
              f"(0.5 = chance) recall@100={metrics['pool_recall_at_100']:.4f} time={seconds:.0f}s", flush=True)

        score = float(metrics["pool_hit_at_1pct"])
        if np.isfinite(score) and score > best_score + args.min_delta:
            best_score, best_epoch, no_improve = score, epoch, 0
            atomic_save({
                "model_state_dict": model.state_dict(), "dataset": metadata.get("dataset"),
                "data_fingerprint": schema_fingerprint, "prepared_data_fingerprint": prepared_fingerprint,
                "categorical_sizes": metadata["categorical_sizes"],
                "user_categorical_sizes": model.user_categorical_sizes, "video_categorical_sizes": model.video_categorical_sizes,
                "user_embedding_dims": model.user_embedding_dims, "video_embedding_dims": model.video_embedding_dims,
                "embedding_dim": 128, "seed": args.seed, "best_epoch": epoch, "best_validation_metrics": metrics,
                "model_version": 3, "negative_mode": "pool-softmax", "temperature": args.temperature,
                "id_min_count": args.id_min_count, "id_dropout": args.id_dropout, "pool_hours": args.pool_hours,
                "architecture": "two_tower_512_256_128_gated_video_id",
                "linear_output": args.linear_output,
            }, best_path)
            print(f"Saved best checkpoint: {best_path}")
        else:
            no_improve += 1
        scheduler.step()
        write_json(history, history_path)
        atomic_save({
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "dataset": metadata.get("dataset"),
            "data_fingerprint": schema_fingerprint, "prepared_data_fingerprint": prepared_fingerprint,
            "seed": args.seed, "epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
            "no_improve": no_improve, "history": history, "numpy_rng_state": rng.bit_generator.state,
            "linear_output": args.linear_output, "resume_config": resume_config,
            **capture_device_rng(device),
        }, latest_path)
        if args.early_stopping_patience and no_improve >= args.early_stopping_patience:
            print(f"Early stopping: no improvement in pool hit@1% for {no_improve} epochs")
            break

    if not best_path.exists():
        # Nothing beat the threshold (for example a very short smoke run): keep the latest weights.
        atomic_save({
            "model_state_dict": model.state_dict(), "dataset": metadata.get("dataset"),
            "data_fingerprint": schema_fingerprint, "prepared_data_fingerprint": prepared_fingerprint,
            "categorical_sizes": metadata["categorical_sizes"], "embedding_dim": 128, "seed": args.seed,
            "best_epoch": history[-1]["epoch"] if history else 0,
            "best_validation_metrics": history[-1]["validation"] if history else {}, "model_version": 3,
            "negative_mode": "pool-softmax", "temperature": args.temperature,
            "architecture": "two_tower_512_256_128_gated_video_id",
            "linear_output": args.linear_output,
        }, best_path)
    run_metadata.update({"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.perf_counter() - started, "best_epoch": best_epoch,
                         "best_pool_hit_at_1pct": best_score})
    write_json(run_metadata, out / "run_metadata.json")
    print(f"Best epoch {best_epoch} (pool hit@1% = {best_score:.4f}). Best model: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
