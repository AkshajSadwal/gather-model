# Experiment protocol and promotion rules

## Data identity

Use `data/processed/ranking/1k_v3_catalog/` for the selected pipeline. Metadata records 10,381,632 train rows, 665,706 validation rows and 665,707 test rows, ordered by time. Schema fingerprint: `7bbaadb7410036832014bdf7f3a225671f84bf60c8226c2f98ae78ca5ebb67e7`; prepared-data fingerprint: `a9e0107ff2818fba39e378746664cf1a1e70402af11ca0e62d85097c2da1195f`.

Never mix V2 checkpoints/maps with V3 data. The V3 catalogue includes metadata for many video IDs that were OOV during training, changing evaluation behaviour for cold-start videos.

## Rules for new experiments

1. Change one conceptual factor at a time.
2. Use a new output directory; do not overwrite a selected checkpoint or index.
3. Save run metadata, training history, and a machine-readable evaluation report.
4. Compare against the same query set, seed, candidate pool, data fingerprint and candidate budget.
5. Report candidate coverage separately from conditional ranking quality.
6. Use user-cluster bootstrap confidence intervals for paired comparisons.
7. Select on validation only. Keep the held-out test report frozen.
8. Promote a new checkpoint only after it improves the protocol relevant to serving, not merely training loss or a sampled metric.
9. Keep the source, checkpoint, data maps and index versioned together.

## Current promotion decisions

| Experiment | Result | Decision |
|---|---|---|
| Pool-softmax V3 | Validation active-pool hit@1%=0.1388; indexed pool hit@5%=0.3750 | Retained as previous reference |
| Linear-output V3 | Validation active-pool hit@1%=0.1480; indexed pool hit@5%=0.3886 | Selected Model 1 |
| Seed-43 in-batch / hybrid losses | Did not establish an improvement on the final active-pool protocol | Not promoted |
| Time-decayed author affinity | No reliable candidate-union gain over static author affinity | Keep static author affinity |
| Creator-history retrieval windows | Lifetime estimate was small/uncertain; short windows regressed candidate recovery | Do not replace static author affinity |
| Model 2 V3 fine-tune | About +0.000119 validation AUC over rebaselined baseline | Not promoted |
| Model 2 `is_like` score | Best observed top-100 score in validation confirmation; selected before test | Selected score |

## Final test policy

The final 20,000-query test was already run with the selected pipeline and score. Treat `reports/experiments/hybrid_ranking_linear_output_v3_k3000_final_test.json` as frozen evidence. Do not repeatedly evaluate new variants against this same test split to select winners.
