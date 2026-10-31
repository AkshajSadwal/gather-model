# Fine-tune both Reels models from their best checkpoints

This patch adds a warm-start option to both trainers and a parallel 10-epoch
fine-tuning launcher. It does **not** overwrite the previous training run.

## What it does

- Loads Model 2 from `model2_ranking/best_model.pt` in the source run.
- Loads Model 1 from `model1_retrieval/best_model.pt` in the source run.
- Starts each trainer in a new, timestamped output directory.
- Trains both models concurrently for 10 additional epochs by default.
- Disables early stopping (`patience=0`), so all 10 epochs run.
- Starts each fresh AdamW optimizer at learning rate `0.0001`. The source best
  checkpoints contain model weights and validation metrics, not optimizer state.
- Copies the source best checkpoint into the new run as the initial best.
  A later epoch replaces it only if its validation score is strictly better
  (`min_delta=0`). Non-improving epochs do not replace the best checkpoint.
- Keeps the existing dashboard, separate logs, and separate output folders.
- Skips held-out test evaluation during tuning; test results should be measured
  once after selecting the best fine-tuned checkpoints.

## Apply

From the project root on macOS:

```bash
unzip -o ~/Downloads/parallel-best-checkpoint-finetune10.zip -d .
source .venv/bin/activate
```

## Run using the completed parallel run from 2026-10-30

```bash
python scripts/run_parallel_finetune.py \
  --source-run artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836 \
  --epochs 10 \
  --batch-size 4096 \
  --learning-rate 0.0001 \
  --early-stopping-patience 0 \
  --min-delta 0
```

The source run must contain:

```text
artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/
  model1_retrieval/best_model.pt
  model2_ranking/best_model.pt
```

## Output

A new directory is created under `artifacts/two_stage_runs_parallel/`, with
separate model folders, logs, and `pipeline_metadata.json`. The new
`best_model.pt` initially contains the source best weights; it is replaced only
when validation improves. The original source run is never modified.

## Resume behavior

If interrupted, `latest_checkpoint.pt` is saved after each completed epoch.
Use each trainer's existing `--resume` flow with its own output directory to
continue that fine-tuning run. Do not re-run the parallel launcher with the same
run name over an existing output directory.
