#!/usr/bin/env python3
"""Fine-tune both Reels recommendation models concurrently from their best checkpoints.

Each model gets its own process, output folder, log, and fixed seed. The prior
best checkpoint is copied into the new run as its baseline, and only validation
improvements replace it. Early stopping is disabled so all requested epochs run.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import re
import selectors
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time
import fcntl
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


DASHBOARD_ACTIVE = False


def enter_dashboard() -> bool:
    """Switch to the terminal's alternate screen so refreshes never scroll."""
    global DASHBOARD_ACTIVE
    if not sys.stdout.isatty() or os.environ.get("TERM", "") == "dumb":
        DASHBOARD_ACTIVE = False
        return False
    # 1049 keeps the dashboard separate from normal scrollback. Hides the
    # cursor; every refresh then redraws the same fixed screen in place.
    sys.stdout.write("\033[?1049h\033[?25l\033[2J\033[H")
    sys.stdout.flush()
    DASHBOARD_ACTIVE = True
    return True


def leave_dashboard() -> None:
    """Restore the user's normal terminal screen and cursor."""
    global DASHBOARD_ACTIVE
    if DASHBOARD_ACTIVE:
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
    DASHBOARD_ACTIVE = False


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temp, path)


def clean_status(value: str) -> str:
    value = ANSI_RE.sub("", value)
    value = CONTROL_RE.sub("", value)
    return " ".join(value.split()).strip()


@dataclass
class Job:
    key: str
    label: str
    command: list[str]
    output_dir: Path
    log_path: Path
    seed: int
    process: subprocess.Popen[bytes] | None = None
    master_fd: int | None = None
    log_handle: Any = None
    pending: str = ""
    latest_status: str = "Waiting to start"
    return_code: int | None = None
    started_at: float | None = None
    elapsed_seconds: float | None = None
    eof: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Model 2 ranking and Model 1 retrieval concurrently from a prior run's best checkpoints."
    )
    parser.add_argument("--source-run", required=True, help="Previous parallel run folder containing model1_retrieval/best_model.pt and model2_ranking/best_model.pt")
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v2")
    parser.add_argument("--output-root", default="artifacts/two_stage_runs_parallel")
    parser.add_argument("--run-name", default=None, help="Unique output folder; defaults to a timestamp")
    parser.add_argument("--epochs", type=int, default=10, help="Number of additional fine-tuning epochs for each model")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size for both models")
    parser.add_argument("--eval-batch-size", type=int, default=4096, help="Model 2 validation batch size")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="Conservative fine-tuning learning rate; optimizer state is restarted from the best weights")
    parser.add_argument("--ranking-seed", type=int, default=42)
    parser.add_argument("--retrieval-seed", type=int, default=43)
    parser.add_argument("--eval-seed", type=int, default=314159,
                        help="Fixed retrieval validation-sampling seed; independent of the training seeds")
    parser.add_argument("--eval-queries", type=int, default=512)
    parser.add_argument("--eval-negatives", type=int, default=500)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="0 disables early stopping so every requested fine-tuning epoch runs")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Save any strict improvement over the starting best validation score")
    parser.add_argument("--cpu-threads", type=int, default=2,
                        help="CPU threads per process; 2 each limits contention during concurrent MPS training")
    args = parser.parse_args()

    positive_ints = [args.epochs, args.batch_size, args.eval_batch_size, args.embedding_dim,
                     args.eval_queries, args.eval_negatives, args.negatives_per_positive]
    if min(positive_ints) < 1:
        parser.error("epochs, batch sizes, embedding dimension and evaluation sizes must be positive")
    if min(args.ranking_seed, args.retrieval_seed, args.eval_seed, args.cpu_threads,
           args.early_stopping_patience) < 0:
        parser.error("seeds, CPU threads and early-stopping patience must be nonnegative")
    if args.ranking_seed == args.retrieval_seed:
        parser.error("Use different --ranking-seed and --retrieval-seed values so the runs have independent random initializations")
    if args.learning_rate <= 0 or args.min_delta < 0:
        parser.error("learning rate must be positive and min-delta nonnegative")
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}):
        parser.error("--run-name must be a simple folder name, not a path")
    return args


def make_jobs(args: argparse.Namespace, project_root: Path, run_root: Path, data_dir: Path, source_run: Path) -> list[Job]:
    ranking_out = run_root / "model2_ranking"
    retrieval_out = run_root / "model1_retrieval"
    ranking_init = source_run / "model2_ranking" / "best_model.pt"
    retrieval_init = source_run / "model1_retrieval" / "best_model.pt"
    ranking_cmd = [
        sys.executable, "-u", "-m", "model2_ranking.train_v2",
        "--data-dir", str(data_dir),
        "--output-dir", str(ranking_out),
        "--init-from", str(ranking_init),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--embedding-dim", str(args.embedding_dim),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.ranking_seed),
        "--run-name", f"{run_root.name}_model2_ranking",
        "--cpu-threads", str(args.cpu_threads),
        "--cooldown-every-batches", "0",
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--min-delta", str(args.min_delta),
        "--skip-test-eval",
    ]
    retrieval_cmd = [
        sys.executable, "-u", "-m", "model1_retrieval.train",
        "--data-dir", str(data_dir),
        "--output-dir", str(retrieval_out),
        "--init-from", str(retrieval_init),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--negatives-per-positive", str(args.negatives_per_positive),
        "--learning-rate", str(args.learning_rate),
        "--eval-queries", str(args.eval_queries),
        "--eval-negatives", str(args.eval_negatives),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--min-delta", str(args.min_delta),
        "--seed", str(args.retrieval_seed),
        "--eval-seed", str(args.eval_seed),
        "--cpu-threads", str(args.cpu_threads),
    ]
    log_dir = run_root / "logs"
    return [
        Job("model2_ranking", "MODEL 2 — RANKING", ranking_cmd, ranking_out,
            log_dir / "model2_ranking.log", args.ranking_seed),
        Job("model1_retrieval", "MODEL 1 — RETRIEVAL", retrieval_cmd, retrieval_out,
            log_dir / "model1_retrieval.log", args.retrieval_seed),
    ]


def start_job(job: Job, project_root: Path, columns: int) -> None:
    master_fd, slave_fd = pty.openpty()
    # Give tqdm a useful width even though the child has a private PTY.
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, max(80, columns), 0, 0))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_handle = job.log_path.open("wb")
    try:
        job.process = subprocess.Popen(
            job.command,
            cwd=project_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
        )
    except Exception:
        os.close(master_fd)
        os.close(slave_fd)
        job.log_handle.close()
        raise
    os.close(slave_fd)
    os.set_blocking(master_fd, False)
    job.master_fd = master_fd
    job.started_at = time.perf_counter()
    job.latest_status = f"Started (PID {job.process.pid})"


def consume_output(job: Job, data: bytes) -> None:
    if job.log_handle is not None:
        job.log_handle.write(data)
        job.log_handle.flush()
    text = data.decode("utf-8", errors="replace")
    parts = re.split(r"[\r\n]+", job.pending + text)
    job.pending = parts.pop()[-8192:] if parts else ""
    for part in parts:
        status = clean_status(part)
        if status:
            job.latest_status = status[-500:]
    # A partial progress update is useful immediately; tqdm commonly ends it with CR.
    if "\r" in text:
        partial = clean_status(job.pending)
        if partial:
            job.latest_status = partial[-500:]


def close_job_pty(job: Job, selector: selectors.BaseSelector) -> None:
    if job.master_fd is not None:
        try:
            selector.unregister(job.master_fd)
        except Exception:
            pass
        try:
            os.close(job.master_fd)
        except OSError:
            pass
        job.master_fd = None
    job.eof = True
    remaining = clean_status(job.pending)
    if remaining:
        job.latest_status = remaining[-500:]
    job.pending = ""


def render_dashboard(jobs: list[Job], run_root: Path, started: float) -> None:
    # Never emit repeated dashboard snapshots into a redirected/non-interactive
    # stream. In a real terminal, redraw inside the alternate screen at the
    # same cursor position, keeping exactly two model panels visible.
    if not DASHBOARD_ACTIVE:
        return
    width = shutil.get_terminal_size((150, 30)).columns
    max_width = max(50, width - 4)
    elapsed = time.perf_counter() - started
    title = f"PARALLEL TRAINING RUNNING  |  elapsed {elapsed / 60:.1f} min"
    lines = [title, "Model 2 Ranking and Model 1 Retrieval are training concurrently.", ""]
    for job in jobs:
        if job.return_code is None:
            state = "RUNNING" if job.process and job.process.poll() is None else "STOPPING"
        else:
            state = "COMPLETED" if job.return_code == 0 else f"FAILED (exit {job.return_code})"
        lines.append(f"{job.label}  [{state}]  seed={job.seed}")
        lines.append(job.latest_status or "(waiting for progress)")
        lines.append("")
    lines.extend([f"Run folder: {run_root}", "Logs: " + str(run_root / "logs"), "Press Ctrl+C to interrupt both jobs safely."])
    # Cursor-home + clear-to-end redraws the same fixed screen. The alternate
    # screen prevents each refresh from entering terminal scrollback.
    sys.stdout.write("\033[H\033[2J")
    for line in lines:
        sys.stdout.write(line[:max_width] + "\n")
    sys.stdout.flush()


def interrupt_jobs(jobs: list[Job], manifest_path: Path, manifest: dict[str, Any]) -> None:
    for job in jobs:
        if job.process is not None and job.process.poll() is None:
            try:
                os.killpg(job.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for job in jobs:
        if job.process is not None and job.process.poll() is None:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                job.return_code = job.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(job.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    job.return_code = job.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(job.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    job.return_code = job.process.wait()
        elif job.process is not None:
            job.return_code = job.process.returncode
        if job.started_at is not None and job.elapsed_seconds is None:
            job.elapsed_seconds = time.perf_counter() - job.started_at
        manifest["stages"][job.key].update({
            "status": "interrupted" if job.return_code not in (0, None) else "stopped",
            "exit_code": job.return_code,
            "elapsed_seconds": job.elapsed_seconds,
        })
    manifest["status"] = "interrupted"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()
    # Make project packages importable when this file is run as a script.
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
    from common.data_version import prepared_data_fingerprint, schema_fingerprint
    required = [
        project_root / "model1_retrieval" / "train.py",
        project_root / "model2_ranking" / "train_v2.py",
        project_root / "common" / "data_version.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("ERROR: Run this script from the project root; missing: " + ", ".join(missing), file=sys.stderr)
        return 2

    data_arg = Path(args.data_dir)
    data_dir = (project_root / data_arg).resolve() if not data_arg.is_absolute() else data_arg.resolve()
    required_data = ["metadata.json", "id_maps.npz", "train.npz", "validation.npz", "test.npz"]
    missing_data = [name for name in required_data if not (data_dir / name).is_file()]
    if missing_data:
        print(f"ERROR: Prepared data is incomplete in {data_dir}. Missing: {', '.join(missing_data)}", file=sys.stderr)
        return 2

    run_name = args.run_name or f"parallel_finetune{args.epochs}_r{args.ranking_seed}_m1s{args.retrieval_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root_arg = Path(args.output_root)
    run_root_base = (project_root / run_root_arg).resolve() if not run_root_arg.is_absolute() else run_root_arg.resolve()
    run_root = run_root_base / run_name
    if run_root.exists():
        print(f"ERROR: Run folder already exists: {run_root}. Choose a new --run-name.", file=sys.stderr)
        return 2

    source_arg = Path(args.source_run)
    source_run = (project_root / source_arg).resolve() if not source_arg.is_absolute() else source_arg.resolve()
    source_checkpoints = {
        "model2_ranking": source_run / "model2_ranking" / "best_model.pt",
        "model1_retrieval": source_run / "model1_retrieval" / "best_model.pt",
    }
    missing_checkpoints = [str(path) for path in source_checkpoints.values() if not path.is_file()]
    if missing_checkpoints:
        print("ERROR: Source run is missing best checkpoints: " + ", ".join(missing_checkpoints), file=sys.stderr)
        return 2

    # Hash/verify the prepared data before starting either worker.
    schema_hash = schema_fingerprint(data_dir)
    prepared_hash = prepared_data_fingerprint(data_dir)
    run_root.mkdir(parents=True, exist_ok=False)
    jobs = make_jobs(args, project_root, run_root, data_dir, source_run)
    started_wall = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "run_name": run_name,
        "started_at_utc": started_wall,
        "run_type": "warm_start_fine_tuning",
        "source_run": str(source_run),
        "source_best_checkpoints": {key: str(value) for key, value in source_checkpoints.items()},
        "fine_tuning_epochs_per_model": args.epochs,
        "data_dir": str(data_dir),
        "schema_fingerprint": schema_hash,
        "prepared_data_fingerprint": prepared_hash,
        "execution": "parallel; Model 2 ranking and Model 1 retrieval run as separate concurrent Python processes",
        "max_epochs_per_model": args.epochs,
        "training_mode": "initialize from each source run's best_model.pt; fresh AdamW optimizer state; only validation improvements replace the copied baseline best checkpoint",
        "training_batch_size_both_models": args.batch_size,
        "ranking_eval_batch_size": args.eval_batch_size,
        "ranking_embedding_dim": args.embedding_dim,
        "learning_rate": args.learning_rate,
        "training_seeds": {
            "model2_ranking": args.ranking_seed,
            "model1_retrieval": args.retrieval_seed,
        },
        "retrieval_evaluation_seed": args.eval_seed,
        "retrieval_eval_queries": args.eval_queries,
        "retrieval_eval_negatives": args.eval_negatives,
        "retrieval_negatives_per_positive": args.negatives_per_positive,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "cpu_threads_per_process": args.cpu_threads,
        "test_evaluation": "skipped for Model 2 during training; retrieval test evaluation is off",
        "status": "starting",
        "stages": {
            job.key: {
                "status": "pending",
                "seed": job.seed,
                "output_dir": str(job.output_dir),
                "log_path": str(job.log_path),
                "command": job.command,
            }
            for job in jobs
        },
    }
    manifest_path = run_root / "pipeline_metadata.json"
    atomic_json(manifest_path, manifest)

    selector = selectors.DefaultSelector()
    started = time.perf_counter()
    columns = shutil.get_terminal_size((150, 30)).columns
    try:
        # Launch both before entering the monitor loop: this is genuinely concurrent.
        for job in jobs:
            start_job(job, project_root, columns)
            assert job.master_fd is not None
            selector.register(job.master_fd, selectors.EVENT_READ, job)
            manifest["stages"][job.key].update({
                "status": "running",
                "pid": job.process.pid if job.process else None,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        manifest["status"] = "running"
        atomic_json(manifest_path, manifest)
        dashboard_enabled = enter_dashboard()
        if dashboard_enabled:
            render_dashboard(jobs, run_root, started)
        else:
            print("Both training processes have started concurrently.", flush=True)
            print("Interactive dashboard is unavailable because this output is not a TTY; follow the two log files instead:", flush=True)
            for job in jobs:
                print(f"  {job.label}: {job.log_path}", flush=True)

        last_dashboard = 0.0
        last_manifest = 0.0
        while any(job.process is not None and job.process.poll() is None for job in jobs) or any(
            job.master_fd is not None for job in jobs
        ):
            events = selector.select(timeout=0.2)
            for key, _ in events:
                job: Job = key.data
                try:
                    data = os.read(key.fd, 65536)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        close_job_pty(job, selector)
                    else:
                        raise
                    continue
                if not data:
                    close_job_pty(job, selector)
                else:
                    consume_output(job, data)

            now = time.monotonic()
            for job in jobs:
                if job.return_code is None and job.process is not None:
                    code = job.process.poll()
                    if code is not None:
                        job.return_code = code
                        job.elapsed_seconds = time.perf_counter() - (job.started_at or time.perf_counter())
                        manifest["stages"][job.key].update({
                            "status": "completed" if code == 0 else "failed",
                            "exit_code": code,
                            "elapsed_seconds": job.elapsed_seconds,
                            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        })
                        if job.master_fd is not None:
                            # PTY can still contain final output; it will be drained on later selector iterations.
                            pass
            if now - last_dashboard >= 0.35:
                render_dashboard(jobs, run_root, started)
                last_dashboard = now
            if now - last_manifest >= 3.0:
                atomic_json(manifest_path, manifest)
                last_manifest = now

        # Flush any final partial status and set aggregate run result.
        for job in jobs:
            if job.return_code is None and job.process is not None:
                job.return_code = job.process.wait()
            if job.log_handle is not None:
                job.log_handle.flush()
                job.log_handle.close()
                job.log_handle = None
            if job.master_fd is not None:
                close_job_pty(job, selector)
        failed = [job.key for job in jobs if job.return_code != 0]
        manifest["status"] = "failed" if failed else "completed"
        manifest["failed_stages"] = failed
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
        leave_dashboard()
        dashboard_enabled = False
        print("\nBoth training jobs have finished." if not failed else f"\nTraining finished with failures: {', '.join(failed)}")
        for job in jobs:
            print(f"{job.label}: {job.output_dir / 'best_model.pt'}")
        print(f"Run metadata: {manifest_path}")
        return 0 if not failed else 1

    except KeyboardInterrupt:
        leave_dashboard()
        print("\nCtrl+C received. Sending an interrupt to both training jobs...", flush=True)
        interrupt_jobs(jobs, manifest_path, manifest)
        return 130
    except Exception as exc:
        leave_dashboard()
        print(f"\nLauncher error: {exc}. Stopping any running workers...", file=sys.stderr, flush=True)
        interrupt_jobs(jobs, manifest_path, manifest)
        raise
    finally:
        for job in jobs:
            if job.master_fd is not None:
                close_job_pty(job, selector)
            if job.log_handle is not None:
                job.log_handle.close()
        selector.close()
        leave_dashboard()


if __name__ == "__main__":
    raise SystemExit(main())
