# Full-catalogue retrieval diagnostics patch

This patch adds `scripts/diagnose_retrieval.py`. It is a validation-only diagnostic; it does not train a model, modify checkpoints, rebuild indexes, or change the existing benchmark evaluator.

## What it reports

- Full-catalogue model ranks using the same validation-query sampling seed and exact catalogue scan used by `scripts.evaluate_retrieval.py`.
- Raw ranks and ranks after filtering the user's other known-positive training videos from the candidate set. The held-out target is always retained, including when the same user-video pair appeared in training.
- Popularity baseline with the same raw/filtered candidate-set definitions.
- Item cohorts: metadata-backed unseen training video IDs and known IDs grouped by training interaction exposure counts (1–4, 5–19, 20–99, 100+).
- Event cohorts: click, like, follow, comment, forward, long-view and complete-view positives.
- User-cluster bootstrap confidence intervals for Recall@K, because multiple sampled query rows can belong to the same user.

## Install

From the project root:

```bash
unzip -o ~/Downloads/retrieval-diagnostics-patch.zip -d .
source .venv/bin/activate
python -m py_compile scripts/diagnose_retrieval.py tests/test_retrieval_diagnostics.py
python -m unittest discover -s tests -v
```

## Run on the original retrieval checkpoint

```bash
caffeinate -i python -m scripts.diagnose_retrieval \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --model-path artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
  --index-dir data/indexes/retrieval/1k_v3_catalog_original \
  --queries 5000 \
  --ks 20 100 500 1000 \
  --seed 314159 \
  --bootstrap 1000 \
  --output reports/experiments/retrieval_diagnosis_original_validation.json
```

## Run on the hybrid epoch-1 candidate

```bash
caffeinate -i python -m scripts.diagnose_retrieval \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --model-path artifacts/retrieval/1k_v3_hybrid_b4096_w005_seed43/epoch1_eval_checkpoint.pt \
  --index-dir data/indexes/retrieval/1k_v3_hybrid_b4096_w005_seed43_epoch1 \
  --queries 5000 \
  --ks 20 100 500 1000 \
  --seed 314159 \
  --bootstrap 1000 \
  --output reports/experiments/retrieval_diagnosis_hybrid_w005_epoch1.json
```

The two commands use the same data, query seed, query count, cutoffs and evaluation definitions. Each produces its own JSON report.

## Important interpretation

- The existing `scripts.evaluate_retrieval.py` output remains the unfiltered benchmark and should be retained for historical comparability.
- Filtered results answer a different question: how well does retrieval rank the held-out target after removing other known-positive training items for that user?
- Neither metric is a perfect production simulation. The current user representation is based on user ID/profile and request context; it does not yet encode a sequence of recent watched videos.
- This diagnostic does not alter model weights or prove a cause. Use the cohort results to decide the next controlled model experiment.
