# Run Model 1 and Model 2 at the same time

This patch adds `scripts/run_parallel_training.py`. It starts **two separate Python training processes concurrently**:

- Model 2 — Ranking (`model2_ranking.train_v2`)
- Model 1 — Retrieval (`model1_retrieval.train`)

Each process gets its own checkpoint/output directory, log file, and training seed. The launcher keeps each child's `tqdm` output attached to a private pseudo-terminal and displays a two-job dashboard, so progress updates do not flood the terminal with one line per refresh.

## Install the patch

Extract the ZIP into the project root (`reels-recommender-m4`) and allow it to merge/replace the files. It adds:

- `scripts/run_parallel_training.py`
- `README_PARALLEL_TRAINING.md`

It also updates `model2_ranking/train_v2.py` with the validation-only test-split metadata fix. This prevents the earlier `NoneType` error when `--skip-test-eval` is used and avoids loading the held-out test arrays during ranking training, reducing memory pressure.

## Run it

From Terminal:

```bash
cd ~/Projects/reels-recommender-m4
source .venv/bin/activate
python scripts/run_parallel_training.py
```

Defaults:

| Setting | Value |
|---|---:|
| Execution | Both models concurrently |
| Maximum epochs | 20 each (early stopping may stop sooner) |
| Training batch size | 4096 each |
| Model 2 ranking seed | 42 |
| Model 1 retrieval seed | 43 |
| Retrieval validation sampling seed | 314159 |
| Model 2 embedding dimension | 32 |
| CPU threads | 2 per process |
| Test evaluation | Skipped during training |

The training seeds are intentionally different. A seed controls random initialization and training-time randomness; different seeds give independent runs, but do not guarantee a higher validation score. The retrieval evaluation seed stays fixed so validation comparisons use the same sampled queries/negatives.

## Output locations

Each run gets a unique folder under:

```text
artifacts/two_stage_runs_parallel/<run-name>/
```

Inside it:

```text
model2_ranking/best_model.pt
model1_retrieval/best_model.pt
logs/model2_ranking.log
logs/model1_retrieval.log
pipeline_metadata.json
```

The old `scripts/run_two_stage_training.py`, if present, is not changed; it remains the sequential launcher. Use `run_parallel_training.py` for simultaneous training.

## Memory warning for a 16 GB Apple Silicon Mac

Both processes use the same unified memory and MPS accelerator. Running both at batch size 4096 at once is memory-intensive; the operating system, prepared arrays, model parameters, gradients, and optimizer states all share the 16 GB pool. The launcher does not silently lower batch size. If macOS reports memory pressure, either process exits with an MPS out-of-memory error, or the machine becomes unstable, rerun with smaller batches:

```bash
python scripts/run_parallel_training.py --batch-size 2048 --eval-batch-size 1024
```

Lower batches reduce memory pressure but may change training dynamics and throughput. Compare validation metrics before deciding which run is better.

Press `Ctrl+C` in the launcher terminal to send an interrupt to both child processes. Logs and any checkpoints already saved remain in the run folder. Each model's own training script saves resumable `latest_checkpoint.pt` files at epoch boundaries (and Model 2 also handles an interrupt checkpoint).
