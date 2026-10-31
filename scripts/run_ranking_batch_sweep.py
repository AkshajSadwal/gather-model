#!/usr/bin/env python3
"""Run sequential, reproducible ranking trials on one device.

The default keeps the agreed batch size at 4096 and repeats it with three
seeds. Other batch sizes can be supplied explicitly. Test evaluation is skipped.
Each run gets an isolated output directory and a separate log file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.data_version import prepared_data_fingerprint, schema_fingerprint


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temp, path)


def load_run_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_trial_row(batch_size: int, seed: int, run_name: str, run_dir: Path,
                    log_path: Path, exit_code: int, meta: dict[str, Any]) -> dict[str, Any]:
    metrics = meta.get("best_validation_metrics") or {}
    row: dict[str, Any] = {
        "run_name": run_name,
        "batch_size": batch_size,
        "seed": seed,
        "status": "completed" if exit_code == 0 and meta.get("status") == "completed" else "failed",
        "exit_code": exit_code,
        "best_epoch": meta.get("best_epoch"),
        "best_validation_auc": finite_float(metrics.get("mean_auc")),
        "best_validation_loss": finite_float(meta.get("best_validation_loss")),
        "elapsed_seconds": finite_float(meta.get("elapsed_seconds")),
        "run_dir": str(run_dir),
        "log_file": str(log_path),
    }
    for name, value in metrics.items():
        if name != "mean_auc":
            row[f"auc_{name}"] = finite_float(value)
    return row


def write_reports(sweep_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(sweep_dir / "trial_results.json", rows)
    columns = sorted({key for row in rows for key in row})
    if columns:
        with (sweep_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "completed" and row.get("best_validation_auc") is not None:
            grouped.setdefault(int(row["batch_size"]), []).append(row)

    comparison: list[dict[str, Any]] = []
    for batch_size in sorted({int(row["batch_size"]) for row in rows}):
        group = grouped.get(batch_size, [])
        aucs = [float(row["best_validation_auc"]) for row in group]
        times = [float(row["elapsed_seconds"]) for row in group if row.get("elapsed_seconds") is not None]
        epochs = [float(row["best_epoch"]) for row in group if row.get("best_epoch") is not None]
        summary_row = {
            "batch_size": batch_size,
            "completed_runs": len(group),
            "attempted_runs": sum(1 for row in rows if int(row["batch_size"]) == batch_size),
            "mean_best_validation_auc": statistics.mean(aucs) if aucs else None,
            "std_best_validation_auc": statistics.stdev(aucs) if len(aucs) > 1 else (0.0 if aucs else None),
            "min_best_validation_auc": min(aucs) if aucs else None,
            "max_best_validation_auc": max(aucs) if aucs else None,
            "mean_elapsed_seconds": statistics.mean(times) if times else None,
            "mean_best_epoch": statistics.mean(epochs) if epochs else None,
        }
        metric_names = sorted({key for row in group for key in row if key.startswith("auc_")})
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in group if row.get(metric_name) is not None]
            summary_row[f"mean_{metric_name}"] = statistics.mean(values) if values else None
            summary_row[f"std_{metric_name}"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        comparison.append(summary_row)

    write_json(sweep_dir / "batch_comparison.json", comparison)
    with (sweep_dir / "batch_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(comparison[0].keys()) if comparison else ["batch_size", "completed_runs"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v2")
    parser.add_argument("--output-root", default="artifacts/ranking/batch_sweeps")
    parser.add_argument("--sweep-name", default=None, help="Defaults to a timestamped unique name")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4096], help="Defaults to the agreed 4096 batch size; optional overrides are for controlled experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--cooldown-every-batches", type=int, default=0)
    parser.add_argument("--cooldown-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if not args.batch_sizes or any(size < 1 for size in args.batch_sizes):
        parser.error("--batch-sizes must contain positive integers")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain nonnegative integers")
    if len(set(args.batch_sizes)) != len(args.batch_sizes) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Batch sizes and seeds must not contain duplicates")
    if args.epochs < 1 or args.eval_batch_size < 1:
        parser.error("--epochs and --eval-batch-size must be positive")

    sweep_name = args.sweep_name or datetime.now().strftime("ranking_batch_sweep_%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_root) / sweep_name
    if sweep_dir.exists():
        parser.error(f"Sweep directory already exists: {sweep_dir}. Choose another --sweep-name.")
    sweep_dir.mkdir(parents=True)

    log_dir = Path("logs/ranking_batch_sweeps") / sweep_name
    log_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    sweep_config = {
        "sweep_name": sweep_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "schema_fingerprint": schema_fingerprint(data_dir),
        "prepared_data_fingerprint": prepared_data_fingerprint(data_dir),
        "batch_sizes": args.batch_sizes,
        "seeds": args.seeds,
        "epochs_max": args.epochs,
        "eval_batch_size": args.eval_batch_size,
        "embedding_dim": args.embedding_dim,
        "learning_rate": args.learning_rate,
        "cpu_threads": args.cpu_threads,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "cooldown_every_batches": args.cooldown_every_batches,
        "test_evaluation": "skipped during tuning",
        "execution": "sequential; one training process at a time",
    }
    write_json(sweep_dir / "sweep_config.json", sweep_config)

    rows: list[dict[str, Any]] = []
    total = len(args.batch_sizes) * len(args.seeds)
    run_index = 0
    interrupted = False
    try:
        for batch_size in args.batch_sizes:
            for seed in args.seeds:
                run_index += 1
                run_name = f"batch_{batch_size}_seed_{seed}"
                run_dir = sweep_dir / run_name
                log_path = log_dir / f"{run_name}.log"
                command = [
                    sys.executable, "-u", "-m", "model2_ranking.train_v2",
                    "--data-dir", args.data_dir,
                    "--output-dir", str(run_dir),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(batch_size),
                    "--eval-batch-size", str(args.eval_batch_size),
                    "--embedding-dim", str(args.embedding_dim),
                    "--learning-rate", str(args.learning_rate),
                    "--cpu-threads", str(args.cpu_threads),
                    "--early-stopping-patience", str(args.early_stopping_patience),
                    "--min-delta", str(args.min_delta),
                    "--cooldown-every-batches", str(args.cooldown_every_batches),
                    "--cooldown-seconds", str(args.cooldown_seconds),
                    "--seed", str(seed),
                    "--run-name", run_name,
                    "--skip-test-eval",
                ]
                print(f"\\n[{run_index}/{total}] Starting {run_name}", flush=True)
                print(f"Log: {log_path}", flush=True)
                exit_code = 1
                try:
                    with log_path.open("w", encoding="utf-8") as log_handle:
                        process = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            bufsize=1,
                        )
                        assert process.stdout is not None
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log_handle.write(line)
                        exit_code = process.wait()
                except KeyboardInterrupt:
                    interrupted = True
                    if "process" in locals() and process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    exit_code = 130

                meta = load_run_metadata(run_dir / "run_metadata.json")
                row = build_trial_row(batch_size, seed, run_name, run_dir, log_path, exit_code, meta)
                rows.append(row)
                write_reports(sweep_dir, rows)
                print(f"[{run_index}/{total}] {run_name}: {row['status']} (exit={exit_code})", flush=True)
                if exit_code != 0 or interrupted:
                    print("Stopping sweep after failed/interrupted run. Completed results are preserved.", flush=True)
                    break
            if exit_code != 0 or interrupted:
                break
    except KeyboardInterrupt:
        interrupted = True

    write_reports(sweep_dir, rows)
    sweep_config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    sweep_config["status"] = "interrupted" if interrupted else (
        "completed" if len(rows) == total and all(r["status"] == "completed" for r in rows) else "incomplete"
    )
    sweep_config["runs_completed_or_attempted"] = len(rows)
    write_json(sweep_dir / "sweep_config.json", sweep_config)

    print(f"\\nSweep status: {sweep_config['status']}")
    print(f"Results: {sweep_dir / 'batch_comparison.csv'}")
    print(f"Per-run results: {sweep_dir / 'trial_results.csv'}")
    print(f"Logs: {log_dir}")
    return 130 if interrupted else (0 if sweep_config["status"] == "completed" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
