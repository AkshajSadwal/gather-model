# Retrieval in-batch regression: audit and safer hybrid experiment

This patch follows the 2026-11-01 review of the full in-batch retrieval run.

## What the audit found

The uploaded code's in-batch candidate builder deduplicates candidate video IDs and marks a candidate positive for a user when the complete known-positive training pair table contains that user-video pair. I did not find an obvious false-negative masking defect in that path.

The experiment changed the warm-started model from pairwise softplus training to an entirely different multi-positive contrastive objective. The 32,768-row smoke run made only 8 optimizer updates; the full dataset makes 955 updates per epoch. The full run used 0.0003 for the new objective and sampled Recall@100 deteriorated from the re-evaluated v3 baseline 0.7891 to 0.4844 after the first epoch. This strongly suggests warm-start forgetting / optimization mismatch. It is a likely diagnosis, not a proven single cause.

The smoke improvement 0.7891 -> 0.7988 on 512 queries was only about five additional successes, so it was too small to establish an improvement.

The full-catalogue output matched the original because `best_model.pt` stayed at the imported starting checkpoint. The last in-batch weights were in `latest_checkpoint.pt`.

## What this patch changes

- Adds `--negative-mode hybrid`.
- Hybrid loss keeps the existing sampled pairwise loss as the main objective and blends in a small multi-positive in-batch auxiliary loss.
- Default `--inbatch-loss-weight` is 0.05, i.e. 95% pairwise + 5% in-batch.
- Training history records the pairwise and in-batch component losses independently.
- Existing `sampled` and `in-batch` modes remain unchanged.
- Resume configuration validates the hybrid coefficient without invalidating older non-hybrid resume configurations.
- Updates `PROJECT_ISSUES_AND_RESOLUTIONS.md` with the failed experiment, code audit and new experiment proposal.

This is an experimental safer path, not a guarantee of better full-catalogue retrieval.

## Install

From the project root:

```bash
cd ~/Projects/reels-recommender-m4
unzip -o ~/Downloads/retrieval-hybrid-loss-fix.zip -d .
source .venv/bin/activate

python -m py_compile model1_retrieval/train.py tests/test_retrieval_hybrid.py
python -m unittest discover -s tests -v
```

The original checkpoint, old indexes, prepared arrays and prior experiment folders are not changed by this patch.

## First run: medium-size smoke test

Do not repeat the full 10-epoch in-batch-only run. First test a 262,144-row subset (64 updates) with a conservative warm-start learning rate:

```bash
python -m model1_retrieval.train \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --output-dir artifacts/retrieval/1k_v3_hybrid_smoke_262k \
  --init-from artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
  --negative-mode hybrid \
  --inbatch-loss-weight 0.05 \
  --epochs 1 \
  --batch-size 4096 \
  --max-positive-rows 262144 \
  --learning-rate 0.00003 \
  --temperature 0.10 \
  --hard-negative-sample-rate 0.25 \
  --eval-queries 2048 \
  --eval-negatives 500 \
  --early-stopping-patience 0 \
  --min-delta 0.0001 \
  --seed 43 \
  --eval-seed 314159 \
  --cpu-threads 2
```

The trainer re-evaluates the imported checkpoint on the same current v3 validation query construction before training, and prints its sampled Recall@100 baseline. It then trains one epoch and records sampled Recall@100 and both loss components.

This is still a smoke test. It does not replace full-catalogue evaluation. If sampled Recall@100 falls materially from the printed starting baseline, do not launch a full run; send the output for review. A small change on 2,048 queries is less noisy than the 512-query test but is not a proof of catalogue improvement.

## If the smoke test is stable

Only after reviewing the smoke output, a short full-data hybrid experiment can be run in a new folder:

```bash
caffeinate -i python -m model1_retrieval.train \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --output-dir artifacts/retrieval/1k_v3_hybrid_b4096_seed43 \
  --init-from artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
  --negative-mode hybrid \
  --inbatch-loss-weight 0.05 \
  --epochs 3 \
  --batch-size 4096 \
  --learning-rate 0.00003 \
  --lr-scheduler cosine \
  --min-learning-rate 0.00001 \
  --temperature 0.10 \
  --hard-negative-sample-rate 0.25 \
  --eval-queries 2048 \
  --eval-negatives 500 \
  --early-stopping-patience 1 \
  --min-delta 0.0001 \
  --seed 43 \
  --eval-seed 314159 \
  --cpu-threads 2
```

After any successful training, evaluate the selected checkpoint with the 5,000-query full-catalogue evaluator and compare Recall@20/100/500/1000, MRR and the same popularity/shuffled-user baselines. Do not select a model on training loss alone.

## Verification in the patch-building environment

- Python compilation passed for the changed trainer and new test.
- Three new unit tests passed for hybrid loss mixing, gradients, shape checks and coefficient validation.
- A tiny synthetic one-epoch run completed for all three modes (`sampled`, `in-batch`, `hybrid`), including validation and checkpoint creation.
- These checks validate the code path only. The patch has not been run against the user's full KuaiRand data or MPS device.
