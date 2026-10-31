#!/usr/bin/env python3
"""Run one full-data ranking trial followed by one retrieval trial, sequentially.

Both models use batch size 4096 by default and a maximum of 20 epochs. The jobs
are intentionally not concurrent: they share the M4's 16 GB unified memory.
Test splits are not evaluated during this validation-based training run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.data_version import prepared_data_fingerprint, schema_fingerprint


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temp, path)


def run_stage(name: str, command: list[str], log_path: Path, project_root: Path) -> tuple[int, float]:
    print("\n" + "=" * 78, flush=True)
    print(f"Starting {name}", flush=True)
    print("Command: " + " ".join(command), flush=True)
    print(f"Log: {log_path}", flush=True)
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                # Keep stderr attached to the parent terminal. tqdm writes its
                # progress bar to stderr and needs a TTY to redraw one line
                # instead of emitting a new line for every refresh. Child
                # stdout remains captured and written to the stage log below.
                stderr=None,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
            code = process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    return code, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v2")
    parser.add_argument("--output-root", default="artifacts/two_stage_runs")
    parser.add_argument("--run-name", default=None, help="Unique run folder name; defaults to a timestamp")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs per model; early stopping may stop sooner")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=314159)
    parser.add_argument("--eval-queries", type=int, default=512)
    parser.add_argument("--eval-negatives", type=int, default=500)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.eval_batch_size, args.embedding_dim,
           args.eval_queries, args.eval_negatives, args.negatives_per_positive) < 1:
        parser.error("epochs, batch sizes, embedding dimension and evaluation sizes must be positive")
    if min(args.seed, args.eval_seed, args.cpu_threads, args.early_stopping_patience) < 0:
        parser.error("seeds, CPU threads and early-stopping patience must be nonnegative")
    if args.learning_rate <= 0 or args.min_delta < 0:
        parser.error("learning rate must be positive and min-delta nonnegative")

    project_root = Path.cwd()
    required = [
        project_root / "model1_retrieval" / "train.py",
        project_root / "model2_ranking" / "train_v2.py",
        project_root / "common" / "data_version.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Run this command from the project root; missing: " + ", ".join(missing))

    data_dir = (project_root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    if not (data_dir / "metadata.json").exists():
        parser.error(f"Prepared dataset metadata not found: {data_dir / 'metadata.json'}")

    run_name = args.run_name or datetime.now().strftime("two_stage_seed_%Y%m%d_%H%M%S")
    run_root = (project_root / args.output_root / run_name).resolve()
    if run_root.exists():
        parser.error(f"Run directory already exists: {run_root}. Choose another --run-name.")
    (run_root / "logs").mkdir(parents=True, exist_ok=False)
    ranking_out = run_root / "ranking"
    retrieval_out = run_root / "retrieval"

    config = {
        "run_name": run_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "schema_fingerprint": schema_fingerprint(data_dir),
        "prepared_data_fingerprint": prepared_data_fingerprint(data_dir),
        "max_epochs_per_model": args.epochs,
        "batch_size": args.batch_size,
        "ranking_eval_batch_size": args.eval_batch_size,
        "ranking_embedding_dim": args.embedding_dim,
        "learning_rate": args.learning_rate,
        "training_seed": args.seed,
        "retrieval_evaluation_seed": args.eval_seed,
        "retrieval_eval_queries": args.eval_queries,
        "retrieval_eval_negatives": args.eval_negatives,
        "retrieval_negatives_per_positive": args.negatives_per_positive,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "cpu_threads": args.cpu_threads,
        "execution": "sequential; ranking completes before retrieval starts; test evaluation skipped",
        "status": "running",
        "stages": {},
    }
    manifest_path = run_root / "pipeline_metadata.json"
    atomic_json(manifest_path, config)

    ranking_command = [
        sys.executable, "-u", "-m", "model2_ranking.train_v2",
        "--data-dir", str(data_dir),
        "--output-dir", str(ranking_out),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--embedding-dim", str(args.embedding_dim),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
        "--run-name", f"{run_name}_ranking",
        "--cpu-threads", str(args.cpu_threads),
        "--cooldown-every-batches", "0",
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--min-delta", str(args.min_delta),
        "--skip-test-eval",
    ]
    retrieval_command = [
        sys.executable, "-u", "-m", "model1_retrieval.train",
        "--data-dir", str(data_dir),
        "--output-dir", str(retrieval_out),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--negatives-per-positive", str(args.negatives_per_positive),
        "--learning-rate", str(args.learning_rate),
        "--eval-queries", str(args.eval_queries),
        "--eval-negatives", str(args.eval_negatives),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--min-delta", str(args.min_delta),
        "--seed", str(args.seed),
        "--eval-seed", str(args.eval_seed),
        "--cpu-threads", str(args.cpu_threads),
    ]

    stages = [
        ("model2_ranking", ranking_command),
        ("model1_retrieval", retrieval_command),
    ]
    try:
        for stage_name, command in stages:
            code, elapsed = run_stage(stage_name, command, run_root / "logs" / f"{stage_name}.log", project_root)
            config["stages"][stage_name] = {
                "exit_code": code,
                "elapsed_seconds": elapsed,
                "output_dir": str(ranking_out if stage_name == "model2_ranking" else retrieval_out),
                "status": "completed" if code == 0 else "failed",
            }
            atomic_json(manifest_path, config)
            if code != 0:
                config["status"] = "failed"
                config["failed_stage"] = stage_name
                config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(manifest_path, config)
                print(f"\nStopping: {stage_name} exited with code {code}.", flush=True)
                print(f"Completed outputs are preserved in: {run_root}", flush=True)
                return code
    except KeyboardInterrupt:
        config["status"] = "interrupted"
        config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, config)
        print(f"\nPipeline interrupted. Completed outputs are preserved in: {run_root}", flush=True)
        return 130

    config["status"] = "completed"
    config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, config)
    print("\nBoth model trainings completed.")
    print(f"Ranking checkpoint: {ranking_out / 'best_model.pt'}")
    print(f"Retrieval checkpoint: {retrieval_out / 'best_model.pt'}")
    print(f"Pipeline metadata: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
