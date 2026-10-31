# Evaluation toolkit (Phase 1: measure before changing the models)

These scripts answer "how good is the model really?" with confidence intervals and
baselines. They read checkpoints but never change them. Run from the project root with
the virtual environment active. Validation is the default; `--allow-test` is required to
touch the held-out test split, so keep it for the final report.

## 1. Data diagnostics (no model needed, NumPy only)

    python -m scripts.diagnose_data --output reports/experiments/data_diagnostics.json

Shows how many validation/test rows use videos never seen in training (these lose all
video metadata), how concentrated exposure is on popular videos, how often a
(user, video) pair repeats, event-rate drift between splits, and whether
complete/short/long-view labels mostly track video duration.

## 2. Ranking evaluation

    python -m scripts.evaluate_ranking \
        --model-path artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model2_ranking/best_model.pt \
        --output reports/experiments/ranking_eval_original.json

Run it again with the fine-tuned checkpoint (`parallel_finetune10_.../model2_ranking/best_model.pt`)
to compare. It reports, per behaviour: ROC-AUC, PR-AUC, per-user AUC (GAUC) with a 95% CI over
users, and calibration (`pred/base` near 1.0 and a small ECE mean the probabilities are honest).
It also reports watch-time fit, and NDCG@10/20 when re-ranking each user's logged impressions
(blocks of 30 consecutive impressions), comparing the current hand-weighted score with every
single probability, a popularity baseline and random order. A `*` marks a difference whose
95% interval excludes zero.

Limit: logged impressions were chosen by the original production recommender, so this measures
re-ordering of items that were actually shown, not discovery of unseen items.

## 3. Retrieval evaluation (full catalogue)

    python -m model1_retrieval.build_index \
        --model-path artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
        --output-dir data/indexes/retrieval/parallel_r42
    python -m scripts.evaluate_retrieval \
        --model-path artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
        --index-dir data/indexes/retrieval/parallel_r42 \
        --output reports/experiments/retrieval_eval_original.json

Ranks 5,000 held-out positives against every video in the index. Three rows are compared on
identical queries: the model, the same model with shuffled users (if it is close to the model,
the score is mostly popularity, not personalisation) and a popularity-only baseline.

## Tests

    python -m unittest tests.test_eval_metrics tests.test_ranking_metadata tests.test_retrieval_core
