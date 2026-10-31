"""Train the Model 1 two-tower retrieval model on prepared KuaiRand interactions.

Positive events are weighted towards long/complete views and stronger actions.
Unobserved catalogue items are sampled as implicit negatives, not treated as
confirmed dislikes. Explicit `is_hate` examples are preferentially sampled as
hard negatives when available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model1_retrieval.data import (
    PreparedSplit,
    build_validation_queries,
    build_video_catalog,
    data_fingerprint,
    explicit_negative_pairs,
    load_metadata,
    load_split,
    pair_keys,
    positive_rows,
    sample_negative_video_indices,
)
from common.data_version import prepared_data_fingerprint
from model1_retrieval.model import TwoTowerRetrieval


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def capture_device_rng(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"torch_rng_state": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        state["mps_rng_state"] = torch.mps.get_rng_state()
    return state


def restore_device_rng(state: dict[str, Any], device: torch.device) -> None:
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if device.type == "cuda" and state.get("cuda_rng_states"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_states"]])
    elif device.type == "mps" and state.get("mps_rng_state") is not None:
        torch.mps.set_rng_state(state["mps_rng_state"].cpu())


def write_json(payload: Any, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temp, path)


def evaluate_sampled_recall(model: TwoTowerRetrieval, split: PreparedSplit, rows: np.ndarray,
                            candidates: np.ndarray, catalog, device: torch.device,
                            query_chunk_size: int = 32) -> dict[str, float]:
    model.eval()
    ranks: list[np.ndarray] = []
    user_cols = TwoTowerRetrieval.USER_COLUMNS
    user_num_cols = TwoTowerRetrieval.USER_NUMERIC_COLUMNS
    with torch.no_grad():
        for start in range(0, len(rows), query_chunk_size):
            end = min(start + query_chunk_size, len(rows))
            row_ids = rows[start:end]
            candidate_ids = candidates[start:end]
            row_cat = split.categorical[row_ids]
            row_num = split.numeric[row_ids]
            u_cat = torch.from_numpy(np.ascontiguousarray(row_cat[:, user_cols])).to(device)
            u_num = torch.from_numpy(np.ascontiguousarray(row_num[:, user_num_cols])).to(device)
            user_vec = model.user_tower(u_cat, u_num)

            flat_video_ids = candidate_ids.reshape(-1)
            v_cat = torch.from_numpy(np.ascontiguousarray(catalog.categorical[flat_video_ids][:, TwoTowerRetrieval.VIDEO_CATALOG_COLUMNS])).to(device)
            v_num = torch.from_numpy(np.ascontiguousarray(catalog.numeric[flat_video_ids][:, TwoTowerRetrieval.VIDEO_CATALOG_NUMERIC_COLUMNS])).to(device)
            video_vec = model.video_tower(v_cat, v_num).view(len(row_ids), candidate_ids.shape[1], -1)
            scores = torch.sum(user_vec.unsqueeze(1) * video_vec, dim=2).cpu().numpy()
            rank = 1 + np.sum(scores[:, 1:] >= scores[:, :1], axis=1)
            ranks.append(rank.astype(np.int32, copy=False))
    all_ranks = np.concatenate(ranks) if ranks else np.asarray([], dtype=np.int32)
    if not all_ranks.size:
        return {"sampled_recall_at_20": float("nan"), "sampled_recall_at_100": float("nan"), "sampled_mrr": float("nan"), "queries": 0}
    return {
        "sampled_recall_at_20": float(np.mean(all_ranks <= 20)),
        "sampled_recall_at_100": float(np.mean(all_ranks <= 100)),
        "sampled_mrr": float(np.mean(1.0 / all_ranks)),
        "median_sampled_rank": float(np.median(all_ranks)),
        "queries": int(all_ranks.size),
        "sampled_negative_count": int(candidates.shape[1] - 1),
    }



def report_mps_memory(device: torch.device, label: str) -> None:
    if device.type != "mps" or not hasattr(torch, "mps"):
        return
    try:
        torch.mps.synchronize()
        gib = 1024 ** 3
        current = torch.mps.current_allocated_memory() / gib
        driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
        recommended_fn = getattr(torch.mps, "recommended_max_memory", None)
        details = []
        if callable(driver_fn):
            details.append(f"driver={driver_fn()/gib:.2f} GiB")
        if callable(recommended_fn):
            details.append(f"recommended_limit={recommended_fn()/gib:.2f} GiB")
        suffix = ", " + ", ".join(details) if details else ""
        print(f"MPS memory ({label}): current={current:.2f} GiB{suffix}", flush=True)
    except Exception as exc:
        print(f"MPS memory report unavailable ({label}): {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v2")
    parser.add_argument("--output-dir", default="artifacts/retrieval/1k_v2_m4")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-positive-rows", type=int, default=0, help="Optional limit for a short smoke test; 0 uses every eligible positive row")
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--hard-negative-sample-rate", type=float, default=0.25)
    parser.add_argument("--eval-queries", type=int, default=512)
    parser.add_argument("--eval-negatives", type=int, default=500)
    parser.add_argument("--eval-query-chunk-size", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=314159,
                        help="Fixed seed for validation/test query and negative sampling; keep constant across training-seed repeats")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="Warm-start a new fine-tuning run from this best_model.pt; optimizer state is intentionally reset",
    )
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate sampled retrieval metrics on the held-out test split after training; keep off while tuning")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.negatives_per_positive, args.eval_queries,
           args.eval_negatives, args.eval_query_chunk_size) < 1:
        parser.error("epochs, batch size, negatives, and evaluation sizes must be positive")
    if args.seed < 0 or args.eval_seed < 0 or args.cpu_threads < 0 or args.temperature <= 0:
        parser.error("training/evaluation seeds and CPU threads must be nonnegative; temperature must be positive")
    if args.resume and args.init_from is not None:
        parser.error("Use either --resume or --init-from, not both")
    if args.init_from is not None and not args.init_from.is_file():
        parser.error(f"--init-from checkpoint does not exist: {args.init_from}")
    if args.max_positive_rows < 0:
        parser.error("--max-positive-rows cannot be negative")
    if not 0 <= args.hard_negative_sample_rate <= 1:
        parser.error("--hard-negative-sample-rate must be between 0 and 1")

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(min(args.cpu_threads, 4))
        except RuntimeError:
            pass

    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    protected = ["best_model.pt", "latest_checkpoint.pt", "training_history.json", "run_metadata.json", "test_metrics.json"]
    if out.exists() and not args.resume and any((out / name).exists() for name in protected):
        raise FileExistsError(f"{out} already contains a previous run. Choose a new output directory or use --resume.")
    out.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(data_dir)
    schema_fingerprint = data_fingerprint(data_dir)
    prepared_fingerprint = prepared_data_fingerprint(data_dir)
    print("Loading training and validation arrays...")
    train = load_split(data_dir, "train")
    validation = load_split(data_dir, "validation")
    catalog = build_video_catalog(data_dir, metadata)
    all_positive_idx, _ = positive_rows(train, metadata)
    positive_idx, positive_weights = positive_rows(train, metadata)
    if args.max_positive_rows and positive_idx.size > args.max_positive_rows:
        limit_rng = np.random.default_rng(args.seed + 101)
        selected = np.sort(limit_rng.choice(len(positive_idx), size=args.max_positive_rows, replace=False))
        positive_idx = positive_idx[selected]
        positive_weights = positive_weights[selected]
        print(f"Smoke-test mode: using {len(positive_idx):,} eligible positive rows")
    if positive_idx.size < args.batch_size:
        print(f"WARNING: only {positive_idx.size:,} positive rows; fewer than the configured batch size")
    n_video_indices = int(catalog.categorical.shape[0])
    train_positive_keys = pair_keys(train, all_positive_idx, n_video_indices)
    hard_negatives_by_user = explicit_negative_pairs(train, metadata, train_positive_keys, n_video_indices)
    print(f"Train rows: {train.rows:,}; eligible positive interactions: {positive_idx.size:,}")
    print(f"Video catalogue: {len(catalog.raw_ids):,}; users: {metadata['num_users'] - 1:,}")
    print(f"Unique positive user-video pairs: {len(train_positive_keys):,}; users with explicit hate/short-view examples: {len(hard_negatives_by_user):,}")

    eval_rows, eval_candidates = build_validation_queries(
        validation, metadata, catalog, train_positive_keys,
        seed=args.eval_seed, query_count=args.eval_queries, negative_count=args.eval_negatives,
    )
    print(f"Sampled validation: {len(eval_rows):,} queries × {eval_candidates.shape[1]:,} candidates")

    device = choose_device()
    if device.type == "mps":
        print("Device: Apple Silicon MPS")
    else:
        print(f"Device: {device}")
    model = TwoTowerRetrieval(metadata).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    report_mps_memory(device, "after model initialization")
    print(f"Batch size: {args.batch_size}; random negatives per positive: {args.negatives_per_positive}")

    best_path = out / "best_model.pt"
    latest_path = out / "latest_checkpoint.pt"
    history_path = out / "training_history.json"
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    no_improve = 0
    start_epoch = 1

    warm_start_info = None
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"Cannot resume: {latest_path} does not exist")
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if int(saved.get("seed", -1)) != args.seed:
            raise ValueError("Resume seed differs from the checkpoint seed")
        if saved.get("dataset") != metadata.get("dataset"):
            raise ValueError("Resume dataset does not match checkpoint")
        if saved.get("data_fingerprint") != schema_fingerprint:
            raise ValueError("Prepared metadata/ID maps differ from the checkpoint training schema")
        if saved.get("prepared_data_fingerprint") != prepared_fingerprint:
            raise ValueError("Prepared train/validation/test arrays differ from the checkpoint; start a new run instead of resuming")
        expected_resume_config = {
            "batch_size": args.batch_size,
            "negatives_per_positive": args.negatives_per_positive,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "hard_negative_sample_rate": args.hard_negative_sample_rate,
            "max_positive_rows": args.max_positive_rows,
            "eval_queries": args.eval_queries,
            "eval_negatives": args.eval_negatives,
            "eval_query_chunk_size": args.eval_query_chunk_size,
            "eval_seed": args.eval_seed,
            "early_stopping_patience": args.early_stopping_patience,
            "min_delta": args.min_delta,
            "cpu_threads": args.cpu_threads,
        }
        saved_resume_config = saved.get("resume_config")
        if saved_resume_config is not None and saved_resume_config != expected_resume_config:
            raise ValueError(
                f"Resume settings differ from checkpoint. Saved={saved_resume_config}; "
                f"current={expected_resume_config}. Keep the original model/training settings when resuming."
            )
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        start_epoch = int(saved["epoch"]) + 1
        best_epoch = int(saved.get("best_epoch", 0))
        best_score = float(saved.get("best_score", -float("inf")))
        no_improve = int(saved.get("no_improve", 0))
        history = list(saved.get("history", []))
        rng.bit_generator.state = saved["numpy_rng_state"]
        restore_device_rng(saved, device)
        print(f"Resumed after completed epoch {start_epoch - 1}")
    elif args.init_from is not None:
        saved = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if saved.get("dataset") != metadata.get("dataset"):
            raise ValueError("The --init-from checkpoint dataset does not match the selected prepared data")
        if saved.get("data_fingerprint") != schema_fingerprint:
            raise ValueError("The --init-from checkpoint metadata/ID-map schema does not match")
        if saved.get("prepared_data_fingerprint") != prepared_fingerprint:
            raise ValueError("The --init-from checkpoint was trained on different prepared train/validation/test arrays")
        model.load_state_dict(saved["model_state_dict"])
        source_metrics = saved.get("best_validation_metrics", {})
        baseline_score = float(source_metrics.get("sampled_recall_at_100", saved.get("best_score", float("nan"))))
        if not np.isfinite(baseline_score):
            raise ValueError("The --init-from checkpoint has no finite sampled Recall@100 baseline")
        source_best_epoch = int(saved.get("best_epoch", 0))
        best_score = baseline_score
        best_epoch = 0  # epoch 0 denotes the imported starting checkpoint in this fine-tune run
        no_improve = 0
        baseline_payload = dict(saved)
        baseline_payload.update({
            "best_epoch": 0,
            "source_best_epoch": source_best_epoch,
            "warm_start_source": str(args.init_from.resolve()),
            "warm_start_baseline_score": baseline_score,
            "fine_tune_seed": args.seed,
        })
        atomic_save(baseline_payload, best_path)
        warm_start_info = {
            "source_checkpoint": str(args.init_from.resolve()),
            "source_best_epoch": source_best_epoch,
            "baseline_sampled_recall_at_100": baseline_score,
            "optimizer_state": "reset; fresh AdamW state for fine-tuning",
        }
        print(
            f"Warm-started from best checkpoint (source epoch {source_best_epoch}); "
            f"baseline sampled Recall@100={baseline_score:.4f}. Optimizer state reset."
        )
    else:
        warm_start_info = None

    run_metadata = {
        "status": "running", "started_at_utc": started_at, "seed": args.seed,
        "dataset": metadata.get("dataset"), "data_dir": str(data_dir), "output_dir": str(out),
        "data_fingerprint": schema_fingerprint,
        "prepared_data_fingerprint": prepared_fingerprint,
        "model": "TwoTowerRetrieval", "architecture": "user: 512-256-128; video: 512-256-128; output: normalized 128d",
        "config": {**vars(args), "init_from": str(args.init_from.resolve()) if args.init_from is not None else None}, "rows": {"train": train.rows, "positive_train": int(positive_idx.size), "validation": validation.rows},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "device": str(device)},
        "positive_event_weights": {"is_click": 0.25, "is_like": 1.0, "is_follow": 1.5, "is_comment": 1.25, "is_forward": 1.5, "long_view": 2.0, "is_profile_enter": 0.75, "complete_view": 2.0},
        "short_view_negative_weight": 0.5,
        "negative_sampling": "two random catalogue items per positive by default; known training positives are rejected; explicit is_hate and short_view pairs can be sampled as hard negatives; unobserved items are not confirmed dislikes",
        "eval_seed": args.eval_seed,
        "warm_start": warm_start_info,
        "validation_protocol": f"sampled retrieval: up to {args.eval_queries} positive queries, each ranked against {args.eval_negatives} sampled catalogue negatives using fixed eval_seed={args.eval_seed}; not full-catalogue recall",
    }
    write_json(run_metadata, out / "run_metadata.json")

    if start_epoch <= args.epochs:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            permutation = rng.permutation(positive_idx)
            loss_total = 0.0
            pairs_seen = 0
            batch_count = math.ceil(len(permutation) / args.batch_size)
            progress = tqdm(range(0, len(permutation), args.batch_size), total=batch_count,
                            desc=f"Retrieval epoch {epoch}/{args.epochs}")
            for start in progress:
                row_ids = permutation[start:start + args.batch_size]
                cat = train.categorical[row_ids]
                num = train.numeric[row_ids]
                u_idx = cat[:, 0].astype(np.int64, copy=False)
                pos_v_idx = cat[:, 1].astype(np.int32, copy=False)
                neg_v_idx = sample_negative_video_indices(
                    u_idx, pos_v_idx, n_video_indices, train_positive_keys,
                    hard_negatives_by_user, rng, args.negatives_per_positive, args.hard_negative_sample_rate,
                )

                u_cat = torch.from_numpy(np.ascontiguousarray(cat[:, TwoTowerRetrieval.USER_COLUMNS])).to(device)
                u_num = torch.from_numpy(np.ascontiguousarray(num[:, TwoTowerRetrieval.USER_NUMERIC_COLUMNS])).to(device)
                pos_cat = torch.from_numpy(np.ascontiguousarray(catalog.categorical[pos_v_idx][:, TwoTowerRetrieval.VIDEO_CATALOG_COLUMNS])).to(device)
                pos_num = torch.from_numpy(np.ascontiguousarray(catalog.numeric[pos_v_idx][:, TwoTowerRetrieval.VIDEO_CATALOG_NUMERIC_COLUMNS])).to(device)
                neg_flat = neg_v_idx.reshape(-1)
                neg_cat = torch.from_numpy(np.ascontiguousarray(catalog.categorical[neg_flat][:, TwoTowerRetrieval.VIDEO_CATALOG_COLUMNS])).to(device)
                neg_num = torch.from_numpy(np.ascontiguousarray(catalog.numeric[neg_flat][:, TwoTowerRetrieval.VIDEO_CATALOG_NUMERIC_COLUMNS])).to(device)
                weights = torch.from_numpy(np.ascontiguousarray(
                    positive_weights[np.searchsorted(positive_idx, row_ids)]
                )).to(device)

                optimizer.zero_grad(set_to_none=True)
                user_vec = model.user_tower(u_cat, u_num)
                positive_vec = model.video_tower(pos_cat, pos_num)
                negative_vec = model.video_tower(neg_cat, neg_num).view(len(row_ids), args.negatives_per_positive, -1)
                positive_score = torch.sum(user_vec * positive_vec, dim=1).unsqueeze(1)
                negative_score = torch.sum(user_vec.unsqueeze(1) * negative_vec, dim=2)
                pairwise_loss = F.softplus((negative_score - positive_score) / args.temperature).mean(dim=1)
                loss = torch.sum(pairwise_loss * weights) / weights.sum().clamp(min=1e-6)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                n = len(row_ids)
                loss_total += float(loss.item()) * n
                pairs_seen += n
                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = loss_total / max(pairs_seen, 1)
            val_metrics = evaluate_sampled_recall(model, validation, eval_rows, eval_candidates,
                                                  catalog, device, args.eval_query_chunk_size)
            report_mps_memory(device, f"after epoch {epoch} validation")
            epoch_seconds = time.perf_counter() - epoch_start
            record = {"epoch": epoch, "train_loss": train_loss, "epoch_seconds": epoch_seconds,
                      "validation": val_metrics}
            history.append(record)
            print(f"\nEpoch {epoch}: loss={train_loss:.5f}, sampled_recall@20={val_metrics['sampled_recall_at_20']:.4f}, "
                  f"sampled_recall@100={val_metrics['sampled_recall_at_100']:.4f}, "
                  f"MRR={val_metrics['sampled_mrr']:.4f}, time={epoch_seconds:.1f}s", flush=True)

            score = float(val_metrics["sampled_recall_at_100"])
            if np.isfinite(score) and score > best_score + args.min_delta:
                best_score, best_epoch, no_improve = score, epoch, 0
                atomic_save({
                    "model_state_dict": model.state_dict(), "dataset": metadata.get("dataset"),
                    "data_fingerprint": schema_fingerprint,
                    "prepared_data_fingerprint": prepared_fingerprint,
                    "categorical_sizes": metadata["categorical_sizes"],
                    "user_categorical_sizes": model.user_categorical_sizes,
                    "video_categorical_sizes": model.video_categorical_sizes,
                    "user_embedding_dims": model.user_embedding_dims,
                    "video_embedding_dims": model.video_embedding_dims,
                    "embedding_dim": 128, "seed": args.seed, "best_epoch": epoch,
                    "best_validation_metrics": val_metrics, "model_version": 1,
                    "architecture": "two_tower_512_256_128",
                }, best_path)
                print(f"Saved best retrieval checkpoint: {best_path}")
            else:
                no_improve += 1

            write_json(history, history_path)
            atomic_save({
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "dataset": metadata.get("dataset"), "data_fingerprint": schema_fingerprint,
                "prepared_data_fingerprint": prepared_fingerprint,
                "seed": args.seed, "epoch": epoch,
                "best_epoch": best_epoch, "best_score": best_score, "no_improve": no_improve,
                "history": history, "numpy_rng_state": rng.bit_generator.state,
                "resume_config": {
                    "batch_size": args.batch_size,
                    "negatives_per_positive": args.negatives_per_positive,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "temperature": args.temperature,
                    "hard_negative_sample_rate": args.hard_negative_sample_rate,
                    "max_positive_rows": args.max_positive_rows,
                    "eval_queries": args.eval_queries,
                    "eval_negatives": args.eval_negatives,
                    "eval_query_chunk_size": args.eval_query_chunk_size,
                    "eval_seed": args.eval_seed,
                    "early_stopping_patience": args.early_stopping_patience,
                    "min_delta": args.min_delta,
                },
                **capture_device_rng(device),
            }, latest_path)
            if args.early_stopping_patience and no_improve >= args.early_stopping_patience:
                print(f"Early stopping after {no_improve} epochs without sampled Recall@100 improvement")
                break
    else:
        print(f"Requested maximum epoch {args.epochs} is already completed in the checkpoint")

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved; inspect validation query construction and training logs")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final = {
        "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "best_epoch": best_checkpoint["best_epoch"],
        "best_validation_metrics": best_checkpoint["best_validation_metrics"],
        "test_evaluation": "pending" if not args.evaluate_test else "completed",
    }
    run_metadata.update(final)
    write_json(run_metadata, out / "run_metadata.json")

    if args.evaluate_test:
        test = load_split(data_dir, "test")
        validation_positive_idx, _ = positive_rows(validation, metadata)
        all_train_validation_keys = np.unique(np.concatenate([
            train_positive_keys, pair_keys(validation, validation_positive_idx, n_video_indices)
        ]))
        test_rows, test_candidates = build_validation_queries(
            test, metadata, catalog, all_train_validation_keys,
            seed=args.eval_seed + 19001, query_count=args.eval_queries, negative_count=args.eval_negatives,
        )
        test_metrics = evaluate_sampled_recall(model, test, test_rows, test_candidates,
                                               catalog, device, args.eval_query_chunk_size)
        write_json({"best_epoch": best_checkpoint["best_epoch"], "sampled_test_metrics": test_metrics,
                    "protocol": run_metadata["validation_protocol"]}, out / "test_metrics.json")
        run_metadata["test_evaluation"] = "completed"
        write_json(run_metadata, out / "run_metadata.json")
        print("Sampled held-out test metrics:", json.dumps(test_metrics, indent=2))

    print(f"Best model: {best_path}")
    print(f"Run metadata: {out / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
