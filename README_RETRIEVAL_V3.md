# Retrieval v3 — pool-softmax training with gated video IDs

Model 1 retrieval was weak (full-catalogue Recall@1000 about 1%, barely above popularity). The saved
reports point to the training and selection design, not to model size. Details and evidence:
section 16 of `PROJECT_ISSUES_AND_RESOLUTIONS.md`.

## What changed

| Change | Files |
|---|---|
| Video IDs seen < 20 times in training (and a random 20% of the rest) are hidden from the video tower, so rare and brand-new videos are scored from content (author, tag, music, duration...). Stored inside the checkpoint; old checkpoints still load. | `model1_retrieval/model.py` |
| Training batches are consecutive-in-time positives; negatives are the other users' positives in the batch plus 4,096 videos from the same day's exposure pool; one softmax per batch with logQ correction; the user's other known positives are excluded, not treated as negatives. | `model1_retrieval/train_v3.py`, `model1_retrieval/losses.py`, `common/pools.py` |
| Checkpoints are selected on "hit rate in the top 1% of the day's pool" instead of Recall@100 against 500 uniformly sampled videos. | `model1_retrieval/train_v3.py` |
| New evaluator: positives ranked against the day's active pool, next to shuffled-user, previous-day popularity and user-author-affinity baselines, with user-cluster confidence intervals. | `scripts/evaluate_retrieval_pools.py` |
| Per-row timestamps recreated from the raw logs, checked row-for-row against your prepared arrays. | `scripts/add_row_times.py` |

`train.py`, `build_index.py`, `retrieve.py` and the existing evaluators are unchanged and still work.

## Run order (Mac, project root, venv active)

```bash
# 0. The torch tests check the new model/loss code against NumPy references. Run them first.
python -m pytest -q tests/test_retrieval_v3.py tests/test_pools.py tests/test_add_row_times.py tests/test_eval_metrics.py

# 1. One-time: create row_times.npz (use the same raw folder you gave prepare_data_v2)
python -m scripts.add_row_times --raw-dir <RAW_KUAIRAND_1K_FOLDER> --data-dir data/processed/ranking/1k_v3_catalog

# 2. Baseline: judge the ORIGINAL model with the pool protocol (about 2-5 min)
python -m scripts.evaluate_retrieval_pools \
  --model-path artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
  --index-dir data/indexes/retrieval/1k_v3_catalog_original \
  --output reports/experiments/pool_eval_original.json

# 3. Smoke test of the new trainer (a few minutes)
python -m model1_retrieval.train_v3 --output-dir artifacts/retrieval/v3_smoke \
  --epochs 1 --max-positive-rows 131072 --eval-queries 1000

# 4. Full training
caffeinate -i python -m model1_retrieval.train_v3 \
  --output-dir artifacts/retrieval/1k_v3_pool_seed42 --epochs 8

# 5. Index and compare (pool protocol, then the old full-catalogue protocol)
python -m model1_retrieval.build_index \
  --model-path artifacts/retrieval/1k_v3_pool_seed42/best_model.pt \
  --output-dir data/indexes/retrieval/1k_v3_pool_seed42
python -m scripts.evaluate_retrieval_pools \
  --model-path artifacts/retrieval/1k_v3_pool_seed42/best_model.pt \
  --index-dir data/indexes/retrieval/1k_v3_pool_seed42 \
  --output reports/experiments/pool_eval_v3_seed42.json
python -m scripts.evaluate_retrieval \
  --model-path artifacts/retrieval/1k_v3_pool_seed42/best_model.pt \
  --index-dir data/indexes/retrieval/1k_v3_pool_seed42 \
  --output reports/experiments/retrieval_validation_v3_seed42.json
```

## How to read the pool report

`mean pct` is the average percentile of the true video among the day's pool: 0 = always first,
0.5 = random. Compare the model with the baselines on the same queries:

* model not clearly better than **shuffled users**: it is not personalising.
* model not better than **popularity (previous window)** or **author affinity**: a simple rule
  beats it; do not ship it. If author affinity wins, the user tower needs a recent-history input
  (next step, below).
* the "cold-start split" shows whether videos unseen in training (about half of the positives) are
  handled.

## Settings worth knowing

| Flag | Default | Meaning |
|---|---|---|
| `--batch-size` | 4096 | positives per batch = in-batch negatives. 8192 doubles the negatives; try it only after the first full run. |
| `--shared-negatives` | 4096 | extra same-day pool negatives per batch |
| `--id-min-count` | 20 | video IDs seen fewer times in training are treated as unknown. Try 5 and 50. |
| `--id-dropout` | 0.2 | extra random hiding of kept IDs while training |
| `--temperature` | 0.07 | try 0.05 and 0.10 |
| `--logq-strength` | 1.0 | 0 switches the popularity correction off (a useful ablation) |
| `--pool-hours` | 24 | length of a time window / pool |

Change one setting per run. Network size is unchanged on purpose: the old training loss fell to 0.22
while validation stalled, which means memorisation, so a bigger network would not help.

## Not done yet

* A recent-watch-history input for the user tower (needs a new, leak-free preparation step).
* Restricting `retrieve.py` to the active pool at serving time (the v3 model is trained and judged
  inside a pool, so serving should filter the same way).
* None of the torch code could be executed where it was written; steps 0 and 3 are the checks.
