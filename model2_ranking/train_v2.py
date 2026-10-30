"""Train and evaluate the metadata-enriched ReelsRankerV2.

Features:
- Saves best_model.pt for inference and latest_checkpoint.pt for resuming.
- --resume loads the latest resumable checkpoint, or falls back to best_model.pt.
- A Ctrl+C during training saves a resumable mid-epoch checkpoint.
- Optional CPU thread limiting and short cooldown pauses help control sustained laptop load.
- Early stopping avoids spending epochs after validation AUC plateaus.
"""
from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from model2_ranking.model_v2 import ReelsRankerV2


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch generators for repeatable experiments.

    MPS kernels are not guaranteed to be bitwise deterministic, so a fixed seed
    improves comparability but does not promise identical results on every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SplitArrays:
    """Load prepared arrays on CPU; transfer only each batch to the GPU."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            # Keep categorical features in their compact stored dtype (normally int32).
            # ReelsRankerV2 casts each batch to long only when doing embedding lookups.
            self.categorical = torch.from_numpy(data["categorical"])
            self.numeric = torch.from_numpy(data["numeric"].astype(np.float32, copy=False))
            self.labels = torch.from_numpy(data["labels"].astype(np.float32, copy=False))
            self.watch = torch.from_numpy(data["watch_log_seconds"].astype(np.float32, copy=False))
        self.n = int(self.categorical.shape[0])


def iter_batches(split, batch_size: int, device: torch.device, shuffle: bool):
    order = torch.randperm(split.n) if shuffle else torch.arange(split.n)
    for start in range(0, split.n, batch_size):
        idx = order[start : start + batch_size]
        yield (
            split.categorical[idx].to(device, non_blocking=True),
            split.numeric[idx].to(device, non_blocking=True),
            split.labels[idx].to(device, non_blocking=True),
            split.watch[idx].to(device, non_blocking=True),
        )


def target_pos_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    positives = labels.sum(0)
    negatives = labels.shape[0] - positives
    return (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=20.0).to(device)


def compute_loss(logits, watch_pred, labels, watch_target, pos_weights):
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weights) + 0.20 * F.smooth_l1_loss(
        watch_pred, watch_target
    )


@torch.no_grad()
def evaluate(model, split, batch_size, device, pos_weights, label_names):
    model.eval()
    loss_total = 0.0
    rows = 0
    pred_parts = []
    truth_parts = []
    for cat, numeric, labels, watch in iter_batches(split, batch_size, device, False):
        logits, watch_pred = model(cat, numeric)
        loss = compute_loss(logits, watch_pred, labels, watch, pos_weights)
        n = int(cat.shape[0])
        loss_total += float(loss.item()) * n
        rows += n
        pred_parts.append(torch.sigmoid(logits).cpu().numpy())
        truth_parts.append(labels.cpu().numpy())
    predictions = np.concatenate(pred_parts)
    truth = np.concatenate(truth_parts)
    aucs = {}
    for i, name in enumerate(label_names):
        aucs[name] = (
            float(roc_auc_score(truth[:, i], predictions[:, i]))
            if np.unique(truth[:, i]).size >= 2
            else float("nan")
        )
    valid = [value for value in aucs.values() if np.isfinite(value)]
    mean_auc = float(np.mean(valid)) if valid else float("nan")
    return loss_total / max(rows, 1), {"mean_auc": mean_auc, **aucs}


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write to a temporary file, then replace the destination atomically."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def atomic_write_json(payload: Any, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temp_path, path)


def rng_snapshot(device: torch.device) -> dict[str, Any]:
    state = {"torch_rng_state": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        state["mps_rng_state"] = torch.mps.get_rng_state()
    return state


def restore_rng(checkpoint: dict[str, Any], device: torch.device) -> None:
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint.get("cuda_rng_states"):
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
    elif device.type == "mps" and checkpoint.get("mps_rng_state") is not None:
        torch.mps.set_rng_state(checkpoint["mps_rng_state"].cpu())



def report_mps_memory(device: torch.device, label: str) -> None:
    """Report MPS allocator/driver memory when the backend exposes these APIs."""
    if device.type != "mps" or not hasattr(torch, "mps"):
        return
    try:
        torch.mps.synchronize()
        gib = 1024 ** 3
        current = torch.mps.current_allocated_memory() / gib
        driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
        recommended_fn = getattr(torch.mps, "recommended_max_memory", None)
        extra = []
        if callable(driver_fn):
            extra.append(f"driver={driver_fn()/gib:.2f} GiB")
        if callable(recommended_fn):
            extra.append(f"recommended_limit={recommended_fn()/gib:.2f} GiB")
        suffix = (", " + ", ".join(extra)) if extra else ""
        print(f"MPS memory ({label}): current={current:.2f} GiB{suffix}", flush=True)
    except Exception as exc:
        print(f"MPS memory report unavailable ({label}): {exc}", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/pure_v2")
    parser.add_argument("--output-dir", default="artifacts/pure_v2")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum total epoch number, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=4096, help="Separate, usually smaller batch size for validation/test")
    parser.add_argument("--skip-test-eval", action="store_true", help="Do not evaluate the held-out test split during experiments; use validation metrics for tuning")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Defaults to 32, or uses the saved value when resuming")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Python, NumPy, and PyTorch")
    parser.add_argument("--run-name", default=None, help="Optional human-readable experiment name saved in metadata/checkpoints")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume latest checkpoint, or initialize from best_model.pt if needed")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Limit PyTorch CPU threads to reduce contention; 0 keeps PyTorch's default")
    parser.add_argument("--cooldown-every-batches", type=int, default=100, help="Synchronize and pause every N training batches; 0 disables pauses")
    parser.add_argument("--cooldown-seconds", type=float, default=2.0, help="Pause duration for --cooldown-every-batches")
    parser.add_argument("--early-stopping-patience", type=int, default=4, help="Stop after this many completed epochs without improved validation AUC; 0 disables")
    parser.add_argument("--min-delta", type=float, default=0.0001, help="Minimum validation AUC improvement to count as progress")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("--epochs, --batch-size, and --eval-batch-size must be positive")
    if args.cpu_threads < 0 or args.cooldown_every_batches < 0 or args.cooldown_seconds < 0:
        parser.error("CPU threads, cooldown interval, and cooldown seconds cannot be negative")
    if args.early_stopping_patience < 0 or args.min_delta < 0:
        parser.error("Early-stopping patience and min-delta cannot be negative")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    # Apply the seed before model initialization and the first shuffled epoch.
    seed_everything(args.seed)
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started_perf = time.perf_counter()

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(args.cpu_threads, 4)))
        except RuntimeError:
            # PyTorch only permits inter-op thread configuration before parallel work starts.
            pass

    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    protected_outputs = ("best_model.pt", "latest_checkpoint.pt", "training_history.json", "test_metrics.json", "run_metadata.json")
    if not args.resume and out.exists():
        existing = [name for name in protected_outputs if (out / name).exists()]
        if existing:
            raise FileExistsError(
                f"Output directory {out} already contains {existing}. "
                "Choose a new --output-dir for a fresh experiment, or use --resume to continue it."
            )
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print("GPU: Apple Silicon via Metal Performance Shaders (MPS)")
    else:
        print("Accelerator: CPU (CUDA/MPS unavailable)")
    print(f"PyTorch CPU threads: {torch.get_num_threads()}")
    print(f"Experiment seed: {args.seed}")
    print(f"Run name: {args.run_name or out.name}")
    print(f"Training batch size: {args.batch_size}; evaluation batch size: {args.eval_batch_size}")
    if args.skip_test_eval:
        print("Held-out test evaluation: SKIPPED (validation only; recommended while tuning)")
    if args.cooldown_every_batches:
        print(
            f"Thermal pacing: pause {args.cooldown_seconds:g}s every "
            f"{args.cooldown_every_batches} training batches"
        )
    if args.early_stopping_patience:
        print(f"Early stopping: patience={args.early_stopping_patience} epoch(s)")

    print("Loading prepared V2 arrays...")
    train = SplitArrays(data_dir / "train.npz")
    val = SplitArrays(data_dir / "validation.npz")
    test = SplitArrays(data_dir / "test.npz")
    if args.max_train_rows > 0 and train.n > args.max_train_rows:
        k = args.max_train_rows
        for attr in ("categorical", "numeric", "labels", "watch"):
            setattr(train, attr, getattr(train, attr)[:k].contiguous())
        train.n = k
        print(f"Quick-test mode: using {k:,} training rows")

    run_metadata = {
        "status": "running",
        "run_name": args.run_name or out.name,
        "started_at_utc": run_started_at,
        "seed": args.seed,
        "config": {
            "data_dir": str(data_dir),
            "output_dir": str(out),
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "embedding_dim_requested": args.embedding_dim,
            "learning_rate": args.learning_rate,
            "max_train_rows": args.max_train_rows,
            "cpu_threads": args.cpu_threads,
            "cooldown_every_batches": args.cooldown_every_batches,
            "cooldown_seconds": args.cooldown_seconds,
            "early_stopping_patience": args.early_stopping_patience,
            "min_delta": args.min_delta,
            "skip_test_eval": args.skip_test_eval,
        },
        "dataset": metadata.get("dataset"),
        "rows": {"train": train.n, "validation": val.n, "test": test.n},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }
    atomic_write_json(run_metadata, out / "run_metadata.json")

    best_path = out / "best_model.pt"
    latest_path = out / "latest_checkpoint.pt"
    history_path = out / "training_history.json"
    resume_checkpoint = None
    resume_kind = None
    if args.resume:
        if latest_path.exists():
            resume_checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
            resume_kind = "latest"
        elif best_path.exists():
            resume_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            resume_kind = "best_only"
        else:
            raise FileNotFoundError(
                f"--resume was requested, but neither {latest_path} nor {best_path} exists."
            )

    saved_embedding_dim = resume_checkpoint.get("embedding_dim") if resume_checkpoint else None
    if args.embedding_dim is not None and saved_embedding_dim is not None and args.embedding_dim != saved_embedding_dim:
        raise ValueError(
            f"Checkpoint uses embedding_dim={saved_embedding_dim}, but --embedding-dim={args.embedding_dim}. "
            "Use the saved dimension when resuming."
        )
    embedding_dim = args.embedding_dim or saved_embedding_dim or 32
    run_metadata["model"] = {
        "architecture": "512-256-128",
        "embedding_dim": embedding_dim,
        "categorical_fields": len(metadata["categorical_sizes"]),
        "numeric_fields": len(metadata["numeric_features"]),
        "binary_targets": metadata["binary_targets"],
        "watch_time_head": True,
    }
    atomic_write_json(run_metadata, out / "run_metadata.json")

    model = ReelsRankerV2(
        metadata["categorical_sizes"],
        len(metadata["numeric_features"]),
        num_targets=len(metadata["binary_targets"]),
        embedding_dim=embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    pos_weights = target_pos_weights(train.labels, device)
    report_mps_memory(device, "after model/optimizer initialization")

    best_score = -float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if resume_checkpoint is not None:
        saved_seed = resume_checkpoint.get("seed")
        if saved_seed is not None and int(saved_seed) != args.seed:
            raise ValueError(
                f"Checkpoint was created with seed={saved_seed}, but --seed={args.seed}. "
                "Use the original seed when resuming this experiment."
            )
        if saved_seed is None:
            print("WARNING: checkpoint has no recorded seed; exact reproducibility cannot be verified.")
        if resume_checkpoint.get("categorical_sizes") != metadata["categorical_sizes"]:
            raise ValueError("Checkpoint categorical feature sizes do not match the selected dataset metadata.")
        if resume_checkpoint.get("num_numeric_features") != len(metadata["numeric_features"]):
            raise ValueError("Checkpoint numeric feature count does not match the selected dataset metadata.")
        if resume_checkpoint.get("binary_targets") != metadata["binary_targets"]:
            raise ValueError("Checkpoint prediction targets do not match the selected dataset metadata.")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if resume_kind == "latest" and "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            # The command-line learning rate is authoritative on resume. PyTorch's
            # optimizer.load_state_dict otherwise silently restores the old LR.
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate
            completed_epoch = int(resume_checkpoint.get("epoch", 0))
            start_epoch = completed_epoch + 1
            best_score = float(resume_checkpoint.get("best_score", -float("inf")))
            best_epoch = int(resume_checkpoint.get("best_epoch", 0))
            no_improve_epochs = int(resume_checkpoint.get("no_improve_epochs", 0))
            history = list(resume_checkpoint.get("history", []))
            restore_rng(resume_checkpoint, device)
            print(f"Resumed model + optimizer state from {latest_path}; next epoch is {start_epoch}.")
        else:
            best_epoch = int(resume_checkpoint.get("best_epoch", 0))
            fallback_score = resume_checkpoint.get("validation_metrics", {}).get("mean_auc", -float("inf"))
            best_score = float(resume_checkpoint.get("validation_score", fallback_score))
            if not np.isfinite(best_score):
                best_score = -float("inf")
            start_epoch = best_epoch + 1
            if history_path.exists():
                try:
                    loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
                    history = [row for row in loaded_history if int(row.get("epoch", 0)) <= best_epoch]
                except (json.JSONDecodeError, TypeError, ValueError):
                    history = []
            print(
                f"Loaded weights from {best_path}; optimizer state was not saved by the older script, "
                f"so the optimizer will restart and training continues at epoch {start_epoch}."
            )
            print("The saved model weights are reused; this is a warm restart, not an exact optimizer resume.")

    def save_latest(completed_epoch: int, checkpoint_history: list[dict[str, Any]], interrupted_during: int | None = None) -> None:
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "categorical_sizes": metadata["categorical_sizes"],
            "num_numeric_features": len(metadata["numeric_features"]),
            "binary_targets": metadata["binary_targets"],
            "embedding_dim": embedding_dim,
            "epoch": completed_epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "no_improve_epochs": no_improve_epochs,
            "history": checkpoint_history,
            "dataset": metadata["dataset"],
            "model_version": 2,
            "seed": args.seed,
            "run_name": args.run_name or out.name,
            "checkpoint_kind": "latest",
            "interrupted_during_epoch": interrupted_during,
            **rng_snapshot(device),
        }
        atomic_torch_save(payload, latest_path)

    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}, which meets the requested maximum ({args.epochs}).")
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            total = 0.0
            rows_seen = 0
            batches_seen = 0
            bar = tqdm(
                iter_batches(train, args.batch_size, device, True),
                total=(train.n + args.batch_size - 1) // args.batch_size,
                desc=f"Epoch {epoch}/{args.epochs}",
            )
            try:
                for cat, numeric, labels, watch in bar:
                    optimizer.zero_grad(set_to_none=True)
                    logits, watch_pred = model(cat, numeric)
                    loss = compute_loss(logits, watch_pred, labels, watch, pos_weights)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    n = int(cat.shape[0])
                    total += float(loss.item()) * n
                    rows_seen += n
                    batches_seen += 1
                    bar.set_postfix(loss=f"{loss.item():.4f}")
                    if args.cooldown_every_batches and batches_seen % args.cooldown_every_batches == 0:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        elif device.type == "mps" and hasattr(torch, "mps"):
                            torch.mps.synchronize()
                        time.sleep(args.cooldown_seconds)
            except KeyboardInterrupt:
                # Do not save the partially updated model under the previous epoch number.
                # Keep latest_checkpoint.pt at the last fully completed epoch. On resume,
                # the interrupted epoch will be replayed cleanly from that checkpoint.
                completed_epoch = int(history[-1]["epoch"]) if history else 0
                print(
                    f"\nInterrupted during epoch {epoch}. The incomplete epoch was NOT checkpointed. "
                    f"The last completed-epoch checkpoint remains at epoch {completed_epoch}. "
                    "Resume with the same output directory and --resume."
                )
                raise SystemExit(130)
            finally:
                bar.close()

            train_loss = total / max(rows_seen, 1)
            val_loss, metrics = evaluate(model, val, args.eval_batch_size, device, pos_weights, metadata["binary_targets"])
            report_mps_memory(device, f"after epoch {epoch} validation")
            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, mean_val_AUC={metrics['mean_auc']:.4f}")
            print(
                "  Validation AUCs: "
                + ", ".join(
                    f"{key}={value:.3f}"
                    for key, value in metrics.items()
                    if key != "mean_auc" and np.isfinite(value)
                )
            )
            epoch_seconds = time.perf_counter() - epoch_started
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": metrics,
                "epoch_seconds": epoch_seconds,
            })
            score = metrics["mean_auc"] if np.isfinite(metrics["mean_auc"]) else -val_loss
            if score > best_score + args.min_delta:
                best_score = score
                best_epoch = epoch
                no_improve_epochs = 0
                best_payload = {
                    "model_state_dict": model.state_dict(),
                    "categorical_sizes": metadata["categorical_sizes"],
                    "num_numeric_features": len(metadata["numeric_features"]),
                    "binary_targets": metadata["binary_targets"],
                    "embedding_dim": embedding_dim,
                    "best_epoch": epoch,
                    "validation_score": score,
                    "validation_loss": val_loss,
                    "validation_metrics": metrics,
                    "dataset": metadata["dataset"],
                    "model_version": 2,
                    "seed": args.seed,
                    "run_name": args.run_name or out.name,
                }
                atomic_torch_save(best_payload, best_path)
                print(f"  Saved best checkpoint: {best_path}")
            else:
                no_improve_epochs += 1

            atomic_write_json(history, history_path)
            save_latest(epoch, history)
            print(f"  Saved resumable checkpoint: {latest_path}")

            if args.early_stopping_patience and no_improve_epochs >= args.early_stopping_patience:
                print(
                    f"Early stopping: validation AUC did not improve for {no_improve_epochs} completed epochs."
                )
                break

    if not best_path.exists():
        raise FileNotFoundError(f"No best model checkpoint exists at {best_path}; no completed epoch was saved.")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if args.skip_test_eval:
        print("\nHeld-out TEST evaluation skipped by request. Use validation AUC for experiment selection.")
    else:
        test_loss, test_metrics = evaluate(model, test, args.eval_batch_size, device, pos_weights, metadata["binary_targets"])
        result = {
            "best_epoch": checkpoint["best_epoch"],
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "checkpoint": str(best_path.resolve()),
            "device": str(device),
        }
        atomic_write_json(result, out / "test_metrics.json")
        print("\nHeld-out TEST metrics:")
        print(f"  test_loss={test_loss:.4f}")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}" if np.isfinite(value) else f"  {key}: unavailable")
    final_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    run_metadata.update({
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - run_started_perf,
        "epochs_recorded": len(history),
        "best_epoch": int(final_checkpoint.get("best_epoch", 0)),
        "best_validation_score": float(final_checkpoint.get("validation_score", float("nan"))),
        "best_validation_loss": float(final_checkpoint.get("validation_loss", float("nan"))),
        "best_validation_metrics": final_checkpoint.get("validation_metrics", {}),
        "test_evaluation": "skipped" if args.skip_test_eval else "completed",
    })
    atomic_write_json(run_metadata, out / "run_metadata.json")
    print(f"\nBest model saved at: {best_path.resolve()}")


if __name__ == "__main__":
    main()
