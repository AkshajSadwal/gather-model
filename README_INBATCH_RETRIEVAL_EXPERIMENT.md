# In-batch retrieval experiment (KuaiRand-1K v3)

This patch adds an optional cosine learning-rate schedule and makes warm-start checkpoint selection fair across prepared-data revisions.

## Important behaviour

- `--negative-mode in-batch` uses the other positive items in the batch as candidates and masks all known positive user-item pairs. Multiple positives for the same user are treated as positives, not false negatives.
- The full training catalogue is the v3 catalogue (`catalog_maps.npz`), while model video-ID embeddings still use training-only IDs. OOV training IDs can therefore use metadata.
- `--init-from` checks the model schema and categorical embedding sizes. It may warm-start from the earlier v2 checkpoint because the v3 preparation keeps the training ID maps and feature schema unchanged.
- On warm start, the trainer evaluates the imported weights against the v3 validation split and v3 sampled candidate set. That current-data result, not the old v2 saved metric, is the checkpoint-selection baseline.
- The optimizer state is reset for the new objective. The imported learned model weights are retained.
- `--lr-scheduler cosine` decays the learning rate from `--learning-rate` to `--min-learning-rate` over the configured maximum epochs.
- The original sampled-negative objective remains available as `--negative-mode sampled`.

## Step 1: smoke test on the Mac

Run from the project root with the venv active:

```bash
python -m model1_retrieval.train \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --output-dir artifacts/retrieval/1k_v3_inbatch_smoke \
  --init-from artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
  --negative-mode in-batch \
  --epochs 1 \
  --batch-size 4096 \
  --max-positive-rows 32768 \
  --learning-rate 0.0003 \
  --temperature 0.10 \
  --eval-queries 512 \
  --eval-negatives 500 \
  --early-stopping-patience 0 \
  --seed 43 \
  --eval-seed 314159 \
  --cpu-threads 2
```

Confirm that output includes:

- `Device: Apple Silicon MPS`
- `negative mode: in-batch`
- `re-evaluated current-data baseline sampled Recall@100=...`
- `Epoch 1: ...`
- `Best model: ...`

Watch the MPS memory report. If the MPS run runs out of memory, repeat the smoke test at `--batch-size 2048` and report the issue before full training.

## Step 2: full 10-epoch in-batch experiment

Only after the smoke test succeeds:

```bash
caffeinate -i python -m model1_retrieval.train \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --output-dir artifacts/retrieval/1k_v3_inbatch_b4096_seed43 \
  --init-from artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
  --negative-mode in-batch \
  --epochs 10 \
  --batch-size 4096 \
  --learning-rate 0.0003 \
  --lr-scheduler cosine \
  --min-learning-rate 0.00005 \
  --temperature 0.10 \
  --hard-negative-sample-rate 0.25 \
  --eval-queries 512 \
  --eval-negatives 500 \
  --early-stopping-patience 0 \
  --min-delta 0.0001 \
  --seed 43 \
  --eval-seed 314159 \
  --cpu-threads 2
```

`--early-stopping-patience 0` means complete all 10 epochs. `best_model.pt` is only replaced when sampled validation Recall@100 improves over the current-data warm-start baseline by more than `min_delta`. Otherwise the best checkpoint remains the imported starting weights. `latest_checkpoint.pt` records the final epoch for recovery.

The experiment runs Model 1 only; do not launch Model 2 concurrently on the 16 GB M4 during this test.

## Step 3: index and full-catalogue validation

```bash
python -m model1_retrieval.build_index \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --model-path artifacts/retrieval/1k_v3_inbatch_b4096_seed43/best_model.pt \
  --output-dir data/indexes/retrieval/1k_v3_inbatch_seed43 \
  --batch-size 4096

python -m scripts.evaluate_retrieval \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --model-path artifacts/retrieval/1k_v3_inbatch_b4096_seed43/best_model.pt \
  --index-dir data/indexes/retrieval/1k_v3_inbatch_seed43 \
  --split validation \
  --queries 5000 \
  --seed 314159 \
  --output reports/experiments/retrieval_validation_1k_v3_inbatch_seed43.json
```

Compare against the existing baseline:

- Checkpoint: `artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt`
- Index: `data/indexes/retrieval/1k_v3_catalog_original`
- Report: `reports/experiments/retrieval_validation_1k_v3_catalog.json`

Use the same data version, query seed and query count. Do not run the held-out test split during tuning.
