#!/usr/bin/env python3
"""Run sequential, seeded Model 1 retrieval trials at a fixed batch size."""
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

from common.data_version import prepared_data_fingerprint


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temp, path)


def finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def write_reports(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / "trial_results.json", rows)
    columns = sorted({key for row in rows for key in row})
    with (out_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    complete = [row for row in rows if row.get("status") == "completed"]
    summary = {
        "batch_size": rows[0]["batch_size"] if rows else None,
        "runs_attempted": len(rows),
        "runs_completed": len(complete),
        "mean_sampled_recall_at_100": statistics.mean([row["sampled_recall_at_100"] for row in complete]) if complete else None,
        "std_sampled_recall_at_100": statistics.stdev([row["sampled_recall_at_100"] for row in complete]) if len(complete) > 1 else (0.0 if complete else None),
        "mean_sampled_recall_at_20": statistics.mean([row["sampled_recall_at_20"] for row in complete]) if complete else None,
        "mean_sampled_mrr": statistics.mean([row["sampled_mrr"] for row in complete]) if complete else None,
        "mean_elapsed_seconds": statistics.mean([row["elapsed_seconds"] for row in complete]) if complete else None,
    }
    write_json(out_dir / "repeat_summary.json", summary)
    with (out_dir / "repeat_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v2")
    parser.add_argument("--output-root", default="artifacts/retrieval/repeated_runs")
    parser.add_argument("--run-group", default=None, help="Defaults to a timestamped group name")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026])
    parser.add_argument("--eval-seed", type=int, default=314159,
                        help="Fixed evaluation-sample seed shared by all training-seed repeats")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-queries", type=int, default=512)
    parser.add_argument("--eval-negatives", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-positive-rows", type=int, default=0, help="Optional smoke-test cap; keep 0 for full trials")
    args = parser.parse_args()
    if not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique nonnegative integers")
    if args.eval_seed < 0:
        parser.error("--eval-seed must be nonnegative")
    if min(args.batch_size, args.epochs, args.negatives_per_positive, args.eval_queries, args.eval_negatives) < 1:
        parser.error("batch size, epochs, negatives, and evaluation sizes must be positive")

    group = args.run_group or datetime.now().strftime("retrieval_repeats_%Y%m%d_%H%M%S")
    group_dir = Path(args.output_root) / group
    if group_dir.exists():
        parser.error(f"Run group already exists: {group_dir}; choose another --run-group")
    group_dir.mkdir(parents=True)
    log_dir = Path("logs/retrieval_repeats") / group
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    group_config = {
        "run_group": group, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_seed": args.eval_seed,
        "data_dir": str(data_dir), "prepared_data_fingerprint": prepared_data_fingerprint(data_dir), "seeds": args.seeds,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "negatives_per_positive": args.negatives_per_positive,
        "learning_rate": args.learning_rate, "weight_decay": 1e-5, "temperature": 0.10,
        "hard_negative_sample_rate": 0.25, "eval_queries": args.eval_queries,
        "eval_negatives": args.eval_negatives,
        "early_stopping_patience": args.early_stopping_patience,
        "max_positive_rows": args.max_positive_rows,
        "execution": "sequential, one training process at a time; validation-only selection",
    }
    write_json(group_dir / "run_group.json", group_config)

    rows: list[dict[str, Any]] = []
    interrupted = False
    for run_no, seed in enumerate(args.seeds, start=1):
        run_name = f"seed_{seed}"
        run_dir = group_dir / run_name
        log_path = log_dir / f"{run_name}.log"
        command = [
            sys.executable, "-u", "-m", "model1_retrieval.train",
            "--data-dir", args.data_dir,
            "--output-dir", str(run_dir),
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--negatives-per-positive", str(args.negatives_per_positive),
            "--learning-rate", str(args.learning_rate),
            "--eval-queries", str(args.eval_queries),
            "--eval-negatives", str(args.eval_negatives),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(seed),
            "--eval-seed", str(args.eval_seed),
        ]
        if args.max_positive_rows:
            command += ["--max-positive-rows", str(args.max_positive_rows)]
        print(f"\n[{run_no}/{len(args.seeds)}] Model 1 retrieval {run_name}")
        print(f"Log: {log_path}", flush=True)
        exit_code = 1
        process = None
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace", bufsize=1)
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_handle.write(line)
                exit_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            exit_code = 130

        meta_path = run_dir / "run_metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            meta = {}
        metrics = meta.get("best_validation_metrics") or {}
        row = {
            "seed": seed, "batch_size": args.batch_size,
            "status": "completed" if exit_code == 0 and meta.get("status") == "completed" else "failed",
            "exit_code": exit_code, "best_epoch": meta.get("best_epoch"),
            "sampled_recall_at_20": finite(metrics.get("sampled_recall_at_20")),
            "sampled_recall_at_100": finite(metrics.get("sampled_recall_at_100")),
            "sampled_mrr": finite(metrics.get("sampled_mrr")),
            "elapsed_seconds": finite(meta.get("elapsed_seconds")),
            "run_dir": str(run_dir), "log_file": str(log_path),
        }
        rows.append(row)
        write_reports(group_dir, rows)
        print(f"[{run_no}/{len(args.seeds)}] {run_name}: {row['status']}", flush=True)
        if exit_code != 0 or interrupted:
            print("Stopping after a failed/interrupted run; completed results remain saved.")
            break

    group_config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    group_config["status"] = "interrupted" if interrupted else ("completed" if len(rows) == len(args.seeds) and all(row["status"] == "completed" for row in rows) else "incomplete")
    group_config["runs_attempted"] = len(rows)
    write_json(group_dir / "run_group.json", group_config)
    print(f"\nRepeat status: {group_config['status']}")
    print(f"Summary: {group_dir / 'repeat_summary.csv'}")
    return 130 if interrupted else (0 if group_config["status"] == "completed" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
