# Model 2 — multi-task ranking

## Selected checkpoint and score

Selected checkpoint:

```text
artifacts/two_stage_runs_parallel/parallel_v3catalog_warmstart_r42_20261101_203431/model2_ranking/warmstart_v3_catalog_baseline.pt
```

The selected inference-time score for the current top-100 objective is the `is_like` sigmoid output (`p_is_like`). Do not use the old hand-weighted composite in `model2_ranking/recommend_v2.py` as though it were the selected score; that CLI still defines and applies the legacy `SCORE_WEIGHTS` formula.

## Inputs and architecture

- Prepared feature schema: V3 catalogue data, fingerprint `7bbaadb7410036832014bdf7f3a225671f84bf60c8226c2f98ae78ca5ebb67e7`.
- 36 categorical fields and 10 numeric fields.
- Categorical embeddings feed a 512 → 256 → 128 shared MLP with dropout.
- Ten binary behavior outputs: `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`, `is_profile_enter`, `complete_view`, `short_view`.
- One watch-time regression head.
- Embedding dimension 32 in the selected V3-catalogue ranker.

The ranker scores candidates, not the entire 4.37M catalogue in the offline hybrid evaluation. Its feature maps must match the same prepared V3 catalogue metadata used by the checkpoint.

## Why the original checkpoint was kept

The earlier Model 2 run on V2 data reached best validation score 0.82378 at epoch 3 (7 epochs recorded). A fine-tuning experiment on V3 changed the validation score only marginally and showed validation deterioration after the first epoch in the earlier V2 fine-tune. The controlled V3 warm-start fine-tune recorded best score 0.8625345, while the rebaselined original checkpoint scored about 0.862416 on the V3 catalogue audit. The difference is approximately 0.000119; that gain was too small to justify promoting the fine-tuned checkpoint. The rebaselined original checkpoint remains selected.

## Score-formula experiment

The evaluator compares formulas on the same candidate slates by summing raw sigmoid outputs. These are ranking diagnostics, not calibrated relevance probabilities.

On the 20,000-query validation confirmation run, Hit@100 is measured across all 20,000 queries; mean target percentile is conditional on the 7,373 hybrid queries whose target was retrieved:

| Formula | Mean target percentile | Hit@100 |
|---|---:|---:|
| `is_like` | 0.429 | **0.0157** |
| `is_like + is_follow` | **0.427** | 0.0150 |
| `is_like + is_comment` | 0.428 | 0.0149 |
| `is_like + is_follow + is_comment` | **0.426** | 0.0149 |
| Legacy current weights | 0.484 | 0.0100 |

Paired with `is_like`, the three-head formula improved mean percentile by 0.00305 (95% CI [-0.00601, -0.00004]) but did not improve Hit@100; its Hit@100 difference was -0.00075 (95% CI [-0.00184, +0.00024]). Like + Follow improved mean percentile by 0.00192 (95% CI [-0.00350, -0.00040]) but its Hit@100 difference was -0.00065 (95% CI [-0.00145, +0.00010]). Because the primary objective is top-100 placement, `is_like` was frozen as the selected score.

The legacy composite was materially worse than `is_like`: on validation, its paired target-percentile difference was +0.0551 (95% CI [+0.0452, +0.0654]) and its Hit@100 difference was -0.0057 (95% CI [-0.00760, -0.00375]).

## Held-out test score

On 20,000 held-out test queries, the hybrid candidate target was present for 6,700 queries (33.5%). On those covered queries, `is_like` achieved mean target percentile 0.42685 (95% CI [0.41863, 0.43512]). Across all 20,000 test queries, Hit@100 was 0.0128 (95% CI [0.01118, 0.01464]).

The percentile is conditional on retrieval coverage; Hit@100 is unconditional. Keep the two interpretations distinct.

## Serving integration requirements

1. Load the checkpoint and V3 feature metadata once at process startup.
2. Build categorical and numeric input rows using the exact V3 mapping, medians, normalization means/stds and candidate catalogue maps.
3. Run inference in batches and retain the `is_like` output column.
4. Sort candidates by `p_is_like` descending.
5. Apply application-level visibility, safety, availability, duplicate, already-watched, freshness, pagination and rate-limit rules.
6. Do not treat an unobserved KuaiRand candidate as a confirmed dislike.

Online adaptation, scheduled retraining, probability calibration and an app-specific ranking objective are deferred. No API service is included yet.
