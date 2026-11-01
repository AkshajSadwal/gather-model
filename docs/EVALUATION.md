# Evaluation protocol and retained reports

## Protocols are not interchangeable

This project used several evaluation protocols during development. Their metrics must not be compared as if they measured the same task.

1. **Sampled retrieval validation:** a positive is compared with a fixed sample of catalogue negatives. This is useful for quick training checks but is not full-catalogue or active-pool recall.
2. **Full-catalogue ranking:** a held-out positive is ranked against the full 4,371,868-video catalogue. Historical reports exist for this protocol.
3. **24-hour active exposure pools:** the positive competes against videos exposed within the same 24-hour time window. This is the protocol used for final retrieval and hybrid candidate evaluation.
4. **Hybrid reranking:** Model 1 and author-affinity independently contribute per-source top-K lists; the union is deduplicated and Model 2 ranks candidates. Candidate budget is per source, so the union can contain up to 2K.

## 24-hour pool caveat

The evaluator builds pools from all exposures in each 24-hour window. A pool may include a video exposed later than an individual query timestamp in that window. This is a within-window lookahead and may overstate what a strict point-in-time serving system can retrieve. A causal replay using only catalogue items available at each query time is still required before production claims.

## Metric meanings

- **Candidate coverage / recovery:** fraction of all evaluated query targets present in a candidate source or union. A target absent from candidates cannot be recovered by the ranker.
- **Mean target percentile:** position of a target within the candidate set, normalized from 0 (first/best) to 1 (last/worst); 0.5 is random expectation. The score-variant percentile is conditional on the target being in the slate and the slate containing more than one item.
- **Hit@K:** whether the target's rank is at most K. The score-variant `hit_at_100` is evaluated across all queries, so a target omitted by candidate generation counts as a miss.
- **MRR:** reciprocal rank, with unrecovered targets contributing zero.
- **Confidence intervals:** user-cluster bootstrap intervals, to preserve dependence among queries from the same user.

## Main retained reports

| Report | Purpose |
|---|---|
| `retrieval_pools_original_v3_paired.json` | Original V3 active-pool baseline and candidate union |
| `retrieval_pools_linear_output_v3_paired.json` | Linear-output Model 1 active-pool result |
| `hybrid_ranking_original_v3_k3000.json` | Original V3 + author union with frozen Model 2 |
| `hybrid_ranking_linear_output_v3_k3000_score_ablation.json` | Expanded 5,000-query score ablation; supersedes the preliminary same-query report |
| `hybrid_ranking_linear_output_v3_k3000_score_confirmation20k.json` | 20,000-query validation score confirmation and paired comparisons |
| `hybrid_ranking_linear_output_v3_k3000_final_test.json` | Final held-out test, run after formula selection |
| `creator_history_validation_20261101.json` | As-of creator-history experiments; static author affinity retained |
| `ranking_audit_1k_v3_catalog.json` | V3-catalogue Model 2 audit |
| `data_diagnostics_1k_v3_catalog.json` | V3 prepared-data diagnostics |
| `author_budget_1k_3k_5k.json` | Candidate-budget exploration |

## Final held-out test headline

20,000 test queries; Model 1 recovery 20.73%, author-affinity recovery 21.24%, and hybrid-union recovery 33.50%. The selected `is_like` score had a mean target percentile of 0.42685 (95% CI [0.41863, 0.43512]) over 6,700 covered queries and all-query Hit@100 of 1.28% (95% CI [1.118%, 1.464%]).

The top-level `ranker_reranking` and `same_candidate_random_baseline` objects in the final JSON refer to the evaluator's legacy `current_weights` score. The selected `is_like` formula is recorded separately under `score_variants.variants.hybrid_union.head:is_like`. Do not quote the top-level current-weights Hit@100 as the selected `is_like` Hit@100.

## Test-set policy

The final test report is frozen. Do not use it to choose new model architectures, score weights, thresholds, or candidate budgets. Any further changes must be tuned on training/validation data and evaluated on a newly defined future holdout or a live controlled experiment.
