# Reproducible ranking experiment patch

This patch adds reproducibility and sweep reporting to the existing V2 ranking trainer.
It does not change the model architecture, target definitions, loss function, or optimizer.

## Changes

- `model2_ranking/train_v2.py`
  - Adds `--seed` (default 42) and `--run-name`.
  - Seeds Python, NumPy, and PyTorch before model initialization and data shuffling.
  - Saves the seed and run name in best/latest checkpoints and run metadata.
  - Rejects resuming a checkpoint with a different recorded seed.
  - Writes `run_metadata.json` with settings, dataset sizes, software/device details,
    validation results, and elapsed time.
  - Records epoch duration in `training_history.json`.
  - Refuses to silently overwrite existing checkpoint/result files on a fresh run.
- `scripts/run_ranking_batch_sweep.py`
  - Runs 4,096 / 8,192 / 16,384 batches with seeds 17 / 42 / 2026 by default.
  - Runs one process at a time and skips held-out test evaluation.
  - Writes per-run logs, `trial_results.csv/json`, and `batch_comparison.csv/json`.
- `docs/EXPERIMENT_PROTOCOL.md`
  - Describes the comparison and selection rules.

MPS does not promise bitwise deterministic execution for every kernel. Fixed seeds improve
comparability but do not guarantee identical results across all runs or PyTorch versions.
