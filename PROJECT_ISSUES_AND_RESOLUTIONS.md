# Project issues, resolutions, experiment history and current status

Last updated: 2026-11-02

This file is the consolidated engineering history. It replaces the collection of one-off patch READMEs. Values below come from the retained run metadata/reports and the user-provided final test output. The different evaluation protocols are labelled explicitly; metrics from different protocols are not compared directly.

## Current decision

The selected offline prototype is:

- Model 1: linear-output V3 two-tower retrieval.
- Candidate source 2: static author affinity, unweighted.
- Model 2: rebaselined V3-catalogue `ReelsRankerV2` checkpoint.
- Ranking formula: `p_is_like`.
- Per-source candidate budget in the evaluated hybrid: K=3,000 (deduplicated union up to 6,000).
- Prepared dataset: `data/processed/ranking/1k_v3_catalog/`.

This is ready for integration into the application API. It is not a production serving system and is not continuously learning yet.

## 1. Data and schema issues

### 1.1 V2 versus V3 catalogue mismatch

**Problem.** The earlier `1k_v2` prepared arrays did not contain the same metadata-backed validation/test catalogue behaviour as the final data. This mattered especially for unseen/OOV video IDs.

**Resolution.** Prepared `1k_v3_catalog` and made it the canonical data directory for final experiments. It has 10,381,632 train rows, 665,706 validation rows and 665,707 test rows; 1,001 users and 3,929,860 videos. The split is time-ordered. V3 includes catalogue metadata that can describe videos whose ID embeddings were not trained.

**Reproducibility.** Schema fingerprint: `7bbaadb7410036832014bdf7f3a225671f84bf60c8226c2f98ae78ca5ebb67e7`; prepared fingerprint: `a9e0107ff2818fba39e378746664cf1a1e70402af11ca0e62d85097c2da1195f`.

**Known limitation.** Static user features are provided without exact as-of timestamps. Aggregate video-statistic tables and random-exposure logs were excluded to avoid possible temporal leakage.

### 1.2 Timestamp / creator-history leakage

**Problem.** Creator-affinity experiments needed a consistent event-time definition and must not allow an event at the same timestamp or in the future to contribute to a query.

**Resolution.** `creator_history_v1` enforces `event_time_ms < query_time_ms`; all events at exactly the query timestamp are excluded together. The artifact records 4,347,085 positive creator events across train/validation/test. It was retained as an experimental feature source, not selected as the main author candidate source.

**Remaining caveat.** The underlying KuaiRand `time_ms` semantics relative to playback start/result availability are not fully established. The final 24-hour candidate pools also have within-window lookahead, so strict causal serving replay remains necessary.

## 2. Retrieval architecture and experiment history

### 2.1 Earlier pairwise / sampled retrieval did not solve catalogue retrieval (2026-10-30 to 2026-11-01)

**Problem.** Training loss and sampled-negative recall gave an incomplete picture. Full-catalogue metrics were low, and cold-start/OOV items were difficult because many item IDs had little or no dedicated embedding information.

**Resolution.** Changed the retrieval objective and validation process to V3 pool-softmax with time-window negatives, log-Q correction and gated video-ID embeddings. A 5,000-query active-pool evaluator was added alongside shuffled-user, previous-window popularity and author-affinity baselines.

### 2.2 Pool-softmax V3 baseline (2026-11-01)

**Run.** `artifacts/retrieval/v3_full_seed42_12ep/` (directory name is stale; metadata records 14 epochs), seed 42, batch 512, 1,024 shared negatives, 24-hour pools, temperature 0.07, log-Q 1.0, ID min count 20, ID dropout 0.2.

**Training validation.** Best epoch 14, pool hit@1%=0.1388 on 5,000 capped-pool validation queries.

**Indexed 5,000-query active-pool evaluation:**

| Metric | Original V3 |
|---|---:|
| Hit@1% | 14.10% |
| Hit@5% | 37.50% |
| Hit@20% | 70.90% |
| Mean percentile | 0.16391 |
| Recall@100 | 1.80% |
| Recall@1,000 | 10.16% |

### 2.3 Linear-output experiment improved every displayed retrieval metric (2026-11-02)

**Problem/hypothesis.** The legacy final ReLU constrains all embedding coordinates to be nonnegative before normalization. Removing the final ReLU and dropout may allow a more expressive angular representation.

**Controlled run.** Same V3 data and principal settings as the original pool-softmax run; only `linear_output=true` was added. Seed 42, 14 epochs, batch 512, 1,024 shared negatives, 24-hour pools, temperature 0.07, log-Q 1.0, ID min count 20, ID dropout 0.2.

**Training validation:** best epoch 14, pool hit@1%=0.1480 versus 0.1388 for original V3 (+0.92 percentage points).

**Indexed active-pool validation (same 5,000 queries and pool windows):**

| Metric | Original V3 | Linear output | Change |
|---|---:|---:|---:|
| Hit@1% | 14.10% | **15.02%** | +0.92 pp |
| Hit@5% | 37.50% | **38.86%** | +1.36 pp |
| Hit@20% | 70.90% | **72.36%** | +1.46 pp |
| Mean percentile | 0.16391 | **0.15443** | -0.00948 (better) |
| Recall@100 | 1.80% | **2.08%** | +0.28 pp |
| Recall@1,000 | 10.16% | **10.86%** | +0.70 pp |

**Resolution.** Promote `artifacts/retrieval/v3_linear_output_seed42_14ep/best_model.pt` and its matching index as the selected Model 1 artifacts. Keep original pool-softmax V3 as a rollback/reference checkpoint while useful.

### 2.4 Candidate union and static author affinity (2026-11-02)

**Finding.** Model 1 and author affinity contribute different candidate sets. In the 5,000-query validation pool evaluation, at K=3,000 the original V3 model recovered 20.42%, author affinity 23.60%, and union 35.74%. Linear output recovered 21.64%, author affinity 23.60%, and union 36.02%. At K=1,000, union recovery changed from 19.64% to 20.04%.

**Cohort finding.** On the same 5,000-query validation split, linear-output Model 1 hit@5% was 35.7% on unseen training video IDs and 42.7% on seen IDs. Static author affinity was 36.2% on unseen IDs and 28.6% on seen IDs.

**Resolution.** Keep static author affinity as a complementary candidate source; do not replace it with Model 1.

### 2.5 In-batch/hybrid-loss experiments not promoted (2026-10-31 to 2026-11-01)

**Problem.** Attempts to improve retrieval with separate seed-43 in-batch/hybrid-loss trials did not establish better performance on the final V3 active-pool protocol. Some early full-catalogue sampled metrics were similar to or worse than the previous run; the one-epoch hybrid-loss diagnostic showed Recall@1,000 0.96% in its full-catalogue report compared with 1.04% in the preceding catalogue report, but these were not a valid final active-pool promotion comparison.

**Resolution.** Do not retain the experimental checkpoints as active models. Keep the scripts/tests if they remain useful for future controlled work; the selected checkpoint is linear-output V3.

### 2.6 Time-decayed and chronological creator history did not replace static author affinity (2026-11-01)

**Time-decay candidate-union experiment.** At K=3,000, paired union deltas versus static author affinity were small and uncertain: 1-day +0.0002 (95% CI [-0.0056, +0.0069]); 3-day -0.0010 ([-0.0075, +0.0050]); 7-day -0.0020 ([-0.0077, +0.0038]); 14-day -0.0038 ([-0.0090, +0.0018]).

**As-of creator-history experiment.** Static author union recovery at K=3,000 was 35.54%. Lifetime-as-of history had a 35.94% point estimate, but paired delta +0.0040 had CI [-0.0016, +0.0099], so it was not conclusive. Short history windows regressed: 1-day 25.52%, 3-day 29.78%, 7-day 33.28%, 14-day 35.26%.

**Resolution.** Keep unweighted static author affinity. Keep creator-history code/report as future research material only; do not use it as the active retrieval source.

## 3. Model 2 ranking history

### 3.1 Early V2 ranker and fine-tuning (2026-10-30 to 2026-10-31)

The original V2 ranker run on the earlier dataset recorded 7 epochs, best epoch 3, validation score 0.82378 and validation loss 0.53813. A subsequent fine-tune was not consistently better: the V2 fine-tune's best checkpoint was epoch 1 and validation loss increased from 0.54388 at epoch 1 to 0.55626 at epoch 10.

### 3.2 V3 catalogue baseline versus V3 warm-start fine-tune (2026-11-01)

The V3 catalogue audit for the rebaselined original ranker reported mean AUC 0.862416 and mean GAUC 0.689180. The controlled V3 warm-start fine-tune recorded a best validation score 0.8625345 (best epoch 2), only about +0.000119 over the rebaselined baseline. This difference was too small to justify promotion.

**Resolution.** Keep the rebaselined original checkpoint at `.../warmstart_v3_catalog_baseline.pt`. The fine-tuned `best_model.pt` and `latest_checkpoint.pt` are not selected and can be deleted by the cleanup script.

### 3.3 Hand-weighted multi-objective score diluted useful ranking signal (2026-11-02)

The earlier composite assigned weights across click, like, follow, comment, forward, watch, profile, complete/short view and hate. Offline score ablations showed the `is_like` head gave a much better target percentile than the legacy composite. On the 20,000-query validation confirmation, `is_like` achieved mean target percentile 0.429 and Hit@100 1.57%; the legacy composite had percentile 0.484 and Hit@100 1.00%.

The paired current-weights-minus-is_like percentile delta was +0.05510 (95% CI [+0.04524, +0.06538]); the paired Hit@100 delta was -0.00570 (95% CI [-0.00760, -0.00375]).

**Resolution.** Select `head:is_like` as the frozen inference-time score for the current top-100 objective. Like+Follow and Like+Follow+Comment improved mean percentile slightly in paired validation, but neither showed a statistically significant Hit@100 gain over `is_like`. No Model 2 retraining was justified.

## 4. Final 20,000-query validation and held-out test (2026-11-02)

### 4.1 Validation score confirmation

20,000 validation queries; 885 users represented across evaluated queries; 7,373 queries had the held-out target in the hybrid candidate slate (36.9%). `is_like` achieved Hit@100 1.57% across all 20,000 queries. Its mean target percentile was 0.429, calculated conditionally over the 7,373 covered queries.

Paired score deltas versus `is_like` (variant minus baseline):

| Formula | Percentile delta and 95% CI | Hit@100 delta and 95% CI |
|---|---|---|
| Like + Follow | -0.00192 [-0.00350, -0.00040] | -0.00065 [-0.00145, +0.00010] |
| Like + Comment | -0.00106 [-0.00410, +0.00192] | -0.00080 [-0.00190, +0.00025] |
| Like + Follow + Comment | -0.00305 [-0.00601, -0.00004] | -0.00075 [-0.00184, +0.00024] |
| Legacy current weights | +0.05510 [+0.04524, +0.06538] | -0.00570 [-0.00760, -0.00375] |

Negative percentile delta means improved average position. Positive Hit@100 delta means more targets reach the top 100. This is why `is_like` remained the selected top-100 formula.

### 4.2 Frozen test evaluation

The final test used 20,000 held-out queries, 24-hour active exposure pools and K=3,000 per source.

**Candidate recovery:** Model 1 20.73%, author affinity 21.24%, hybrid union **33.50%**. The target was retrieved for 6,700 queries (868 users).

**Selected `is_like` score:** mean target percentile 0.42685; 95% CI [0.41863, 0.43512]; delta versus random -0.07315; 95% CI [-0.08137, -0.06488]. Hit@100 over all queries was 1.28%; 95% CI [1.118%, 1.464%].

**Cohort recovery:**

| Cohort | Queries | Model 1 coverage | Author coverage | Hybrid coverage |
|---|---:|---:|---:|---:|
| Cold-start video IDs | 13,071 | 19.26% | 23.50% | 33.40% |
| Seen video IDs | 6,929 | 23.51% | 16.97% | 33.68% |

Cohort Hit@K/MRR values in the final report use the legacy evaluator `current_weights` score, not the selected `is_like` variant. Do not attribute those cohort ranking values to `is_like`.

The report's top-level `ranker_reranking` and `same_candidate_random_baseline` also use `current_weights`; for selected-score metrics, read `score_variants.variants.hybrid_union.head:is_like`.

## 5. Operational / engineering issues resolved

- **Python import errors for scripts run by file path:** use `PYTHONPATH="$PWD"` from the repository root for `scripts/evaluate_*.py`.
- **Checkpoint/index mismatch risk:** store and use the matching checkpoint, V3 prepared data and index together; metadata fingerprints are retained.
- **Misleading training loss:** promotion is based on full active-pool and hybrid validation metrics, not training loss alone.
- **Run-directory naming drift:** the `v3_full_seed42_12ep` folder actually records 14 epochs; trust `run_metadata.json`, not the folder label.
- **Fine-tuning overfit / marginal gain:** retain the original V3-catalogue ranker checkpoint; delete the unpromoted fine-tune checkpoints.
- **Temporary source backups:** `.pre_score_ablation.bak` files are patch leftovers and are removed from the cleaned source snapshot.

## 6. Known limitations and open work

1. **No live API:** the repository is model/evaluation code, not a backend feed service.
2. **No session-history tower input:** Model 1 does not encode ordered recent watches.
3. **Candidate coverage ceiling:** the final hybrid recovered 33.5% of held-out targets. Unretrieved targets cannot be rescued by Model 2.
4. **24-hour pool lookahead:** the evaluation pool may include exposures after an individual query timestamp within its 24-hour window. Add strict point-in-time catalogue replay before production deployment.
5. **Static user feature timing:** source timestamps are not available for all static user features.
6. **No real-time or scheduled learning:** interactions are not yet being fed to an online feature/history service or periodic retraining pipeline.
7. **No production controls:** safety/visibility, already-seen filtering, pagination, deduplication, freshness, rate limiting, latency SLOs, monitoring and A/B experiments are not implemented.
8. **Score calibration:** `p_is_like` is a useful ordering score on this benchmark; it is not proven calibrated or equivalent to overall satisfaction in the app.

## 7. Post-cleanup repository state

The one-time cleanup completed on 2026-11-02 after verifying that the selected Model 1 checkpoint, Model 2 checkpoint, retrieval index and V3 prepared data were present.

The cleanup utility was used for this one-time operation and is intentionally not retained in the project. The selected artifacts and V3 dataset remain in place. External backups of the merged files and previous README files were created before cleanup.

The original pool-softmax V3 retrieval checkpoint remains temporarily as a reference/rollback model. The selected linear-output V3 checkpoint and selected V3-catalogue Model 2 checkpoint remain the active pair. The retained experiment reports document the rejected and promoted experiments.
