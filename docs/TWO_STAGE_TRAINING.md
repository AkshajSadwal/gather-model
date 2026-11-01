# Two-stage training and checkpoint lifecycle

The project currently has trained checkpoints for both stages. The selected pair is documented in the root README and in `docs/MODEL1_RETRIEVAL.md` / `docs/MODEL2_RANKING.md`.

## Do not retrain just to integrate the API

The current phase is API integration. Re-training is not required to serve the selected offline models. Keep the selected checkpoint paths and feature schema fixed while implementing the service.

## Training entry points

Inspect the exact command-line options in the checked-out source before starting a new run:

```bash
python -m model1_retrieval.train_v3 --help
python -m model2_ranking.train_v2 --help
python scripts/run_parallel_training.py --help
```

Use a new `--output-dir` for every experiment. Preserve `best_model.pt`, run metadata and training history for any experiment under review. Do not resume from `latest_checkpoint.pt` unless resuming that exact run intentionally.

## Checkpoint promotion

A candidate checkpoint is not promoted because its training loss decreases. Compare the new checkpoint on the same V3 validation data and evaluation protocol. For retrieval, build a matching index and evaluate active-pool retrieval and the hybrid candidate union. For ranking, keep candidate slates fixed and evaluate paired ranking metrics against the current selected score. Only after a decision is frozen may a new holdout test or live experiment be run.

## Current selected pair

- Model 1: `artifacts/retrieval/v3_linear_output_seed42_14ep/best_model.pt`
- Model 1 index: `data/indexes/retrieval/1k_v3_v3_linear_output_seed42_14ep/`
- Model 2: `artifacts/two_stage_runs_parallel/parallel_v3catalog_warmstart_r42_20261101_203431/model2_ranking/warmstart_v3_catalog_baseline.pt`
- Model 2 score: `p_is_like`

## Deferred

Online/continual learning, periodic retraining automation, model registry, drift alerts and rollback automation are not implemented. User events can later be collected continuously, but model weights should be updated through controlled training/evaluation and checkpoint promotion, not modified implicitly during each API request.
