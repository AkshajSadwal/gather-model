# Reproducible ranking experiment patch

This patch adds seeded experiment metadata and repeated-run reporting to the V2 ranking trainer. It does not change the ranking model architecture, targets, loss, or optimizer.

## Changes

- `model2_ranking/train_v2.py`
  - Adds `--seed` (default 42) and `--run-name`.
  - Seeds Python, NumPy, and PyTorch before model initialization and data shuffling.
  - Saves seed/run name in checkpoints and `run_metadata.json`.
  - Records per-epoch duration and rejects accidental output-directory reuse.
  - Uses project-layout defaults and a 4,096 training batch size.
- `scripts/run_ranking_batch_sweep.py`
  - Runs three sequential 4,096-batch repeats with seeds 17, 42, and 2026 by default.
  - Skips held-out test evaluation during tuning and writes comparison reports.
  - Other batch sizes remain available only as explicit overrides.
- `docs/EXPERIMENT_PROTOCOL.md`
  - Defines the fixed-batch repeatability protocol.

MPS does not promise bitwise deterministic execution for every kernel. Fixed seeds improve comparability but do not guarantee identical results across all runs or PyTorch versions.


## Additional safeguards in the current source update

- `common/data_version.py` fingerprints metadata, ID maps, and all three prepared splits. Both trainers save the prepared-data fingerprint in run metadata/checkpoints, and new checkpoints can be checked against the same data version.
- `scripts/run_ranking_batch_sweep.py` reports mean and standard deviation for each action target across seeds, in addition to the mean AUC.
- The ranking recommender outputs the model's predicted watch time for analysis. Its weighted action-probability score remains a hand-set baseline heuristic and is not presented as calibrated utility.
