# Reels Recommendation Project — Problems, Resolutions, and Open Issues

**Document purpose:** Persistent engineering handover for the `feed-model-mac` / `reels-recommender-m4` project. Keep this file in the project root and include it in every future source-code patch ZIP.

**Last updated:** 2026-11-01 (section 16)  
**Current development machine:** Apple Silicon Mac mini M4, 16 GB unified memory  
**Current environment (reported):** macOS, Python 3.12.16, PyTorch 2.15.0, MPS available  
**Project focus:** Two-stage short-video/Reels recommendation using KuaiRand-1K, with KuaiRand-Pure experiments retained as historical baselines.

> Status labels: **RESOLVED** = a fix or workaround was applied and evidenced in the conversation; **KNOWN LIMITATION** = understood behavior or dataset limitation; **OPEN** = needs further investigation or implementation; **VERIFY** = a fix was produced but its behavior still needs confirmation on the user's machine.


## 2026-10-31 update — cold-start and in-batch retrieval patch

A source patch was built against the latest uploaded `feed-model-mac-main (5).zip`.

**Code changes supplied:**

- Separate `catalog_maps.npz` stores raw video IDs from the entire `video_features_basic_1k.csv` catalogue, training-ID embedding indices (0 for cold-start), train-fitted video metadata categories and numeric item features.
- `train.npz`, `validation.npz`, and `test.npz` gain `catalog_video_indices`. Existing `id_maps.npz` train vocabularies and model categorical embedding sizes are intentionally retained.
- Retrieval data loading/training/sampled evaluation/full-catalogue evaluation/index construction/query and ranker candidate scoring now use catalogue indices where appropriate.
- Full-catalogue report distinguishes unseen training IDs with metadata from positive rows missing from the basic-video catalogue.
- `--negative-mode in-batch` implements deduplicated candidate items and a multi-positive mask built from all known training positive user-item pairs. Known positives for the same user are included in the numerator rather than mislabeled as negatives. Existing sampled-negative mode remains the default.
- Ranking example inference uses current UTC time for video-age features by default, with `--as-of-ms` for reproducibility.
- Index metadata records a SHA-256 of `catalog_maps.npz`; evaluation and retrieval reject indexes built from another catalogue map.

**Verification performed in the patch workspace:**

- Python compilation passed for all source, script and test modules.
- Unit suite: 25 tests passed.
- Synthetic data-preparation fixture: passed. Train-seen videos retained their video ID indices; validation/test-only videos had ID embedding 0 but a valid catalogue index and retained metadata.
- Synthetic Model 1 in-batch training: one epoch completed and checkpoint saved.
- Synthetic full-catalogue index/evaluation: all five item rows indexed; unseen video positives were included because their metadata rows existed.
- Synthetic ranking inference and retrieval → ranking candidate-file handoff: passed and scored unseen catalogue items with OOV video-ID embeddings.

**Not yet verified:** The actual KuaiRand-1K raw data has not been re-prepared with this patch. The true percentage of unseen validation/test video IDs covered by `video_features_basic_1k.csv` is unknown. The in-batch retrieval variant has not yet been trained/evaluated on the full 1K dataset or on the user's M4. All new changes are therefore **VERIFY on user machine**, not yet production validated.

**Important user instruction:** Every future patch ZIP must include this file at project root and update it when problems/fixes/results change.


---

## 1. Current architecture and data context

### Model 1 — Retrieval

- Two-tower model: user tower and video tower trained together.
- Each tower uses a 512 → 256 → 128 MLP with normalized 128-dimensional output vectors.
- The retrieval item embedding uses a 16-dimensional video-ID embedding, with side-feature embeddings and static video metadata.
- Current training supports the original two-sampled-negative pairwise softplus objective (`--negative-mode sampled`) and a new optional masked multi-positive in-batch contrastive objective (`--negative-mode in-batch`). The sampled mode remains the default until controlled 1K experiments compare them.
- Training-time validation is sampled (historically 512 positive queries × 500 sampled negatives); this is a quick monitoring metric, not a full-catalogue metric.
- `model1_retrieval/build_index.py` builds a float16 vector index; `scripts/evaluate_retrieval.py` can exact-scan the full indexed catalogue.

### Model 2 — Ranking

- `ReelsRankerV2`, one shared feature-aware MLP with 36 categorical fields and 10 numerical fields.
- Shared hidden layers: 512 → 256 → 128; dropout 0.20 / 0.15 / 0.10.
- Embedding dimension 32 in the recorded 1K ranking experiments.
- Outputs: 10 behaviour predictions plus watch-time regression.
- Current training code historically used constant learning rate 0.001, weight decay 1e-5 and positive-class weights capped at 20.
- `scripts/evaluate_ranking.py` measures target-level AUC/PR-AUC/GAUC/calibration, watch-time metrics and NDCG for alternative scoring rules.

### Dataset preparation

- KuaiRand-1K standard-feed logs are split chronologically: older log + first 80% of newer log for training, next 10% for validation, final 10% for test.
- Prepared data lives under `data/processed/ranking/1k_v2`.
- Prepared categorical ID vocabularies are built from training-period IDs. Unseen validation/test user or video IDs map to OOV index 0.
- Legacy `1k_v2` arrays and indexes only catalogued training-vocabulary videos; when a video ID mapped to 0, its basic-video side metadata was also lost from the packed interaction row.
- The cold-start patch adds a separate `catalog_maps.npz` based on every raw video ID present in the basic-video feature table. An unseen video keeps ID embedding 0 but can retain known metadata encoded using training-only category vocabularies. Videos absent from the basic-video table remain unindexable.
- The patch also adds `catalog_video_indices` to splits so train-ID embeddings and full-catalogue row IDs are no longer conflated.

---

## 2. Environment and migration history

### Earlier Windows / NVIDIA environment

**Reported setup:** Windows 11, Python 3.14.6, NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB VRAM, CUDA-enabled PyTorch build.

**Historical experiments:**

- Older Pure V1-style training run, five epochs, mean validation AUC 0.6379.
- Pure V2 feature-aware run, embedding dimension 32, ten epochs: held-out test mean AUC 0.7784; test loss 0.5802.
- A later Pure V2 embedding-dimension-64 run reached validation mean AUC 0.7646 at epoch 2 in the visible log; the full later history was not available in the conversation.

**Status:** Historical only. Do not compare these numbers directly with the later KuaiRand-1K M4 experiments because dataset, feature preparation and model configurations differ.

### Migration to Apple Silicon M4

**Reported setup:** Mac mini M4, 16 GB unified memory, Python 3.12.16, PyTorch 2.15.0, MPS available.

The project was reorganized into `common/`, `model1_retrieval/`, `model2_ranking/`, `scripts/`, `tests/`, `artifacts/`, `reports/` and `docs/`.

**Status:** RESOLVED. Training was successfully run on MPS. Both ranking and retrieval later ran concurrently at batch size 4096 without an MPS out-of-memory failure in the reported run.

---

## 3. Problems encountered and how they were addressed

### 3.1 Ranking runner crashed when test evaluation was skipped

**Problem:** `train_v2.py` allowed `--skip-test-eval`, set `test = None`, but later attempted to read `test.n` when writing metadata, raising `AttributeError: 'NoneType' object has no attribute 'n'`.

**Resolution:** A patch was supplied to guard the test row-count access when test evaluation is skipped.

**Status:** RESOLVED in the parallel-training patch; preserve the guard in future edits.

### 3.2 The first two-stage runner was sequential, not concurrent

**Problem:** `run_two_stage_training.py` trained ranking first and retrieval second. This did not meet the user's request to train side by side.

**Resolution:** Added `scripts/run_parallel_training.py`, launching two independent Python processes with separate output/checkpoint directories, seeds and logs.

**Status:** RESOLVED. The user's terminal output showed both processes started concurrently (ranking PID 43152, retrieval PID 43153).

### 3.3 Parallel runner flooded terminal with repeated progress output

**Problem:** Child process output was captured through pipes, so `tqdm` did not see a TTY. The parent runner also repeatedly redrew dashboard text as additional terminal output.

**Resolution:** A terminal-dashboard patch was supplied. It uses an alternate terminal screen when an interactive TTY is available and avoids repeatedly printing the dashboard when output is non-interactive.

**Status:** VERIFY. Patch was supplied and tested with simulated workers in the development environment, but the user has not yet supplied a confirmation that the patched launcher updates exactly two fixed bars on their terminal.

### 3.4 Concurrent training was interrupted by Ctrl+C

**Problem:** The user pressed Ctrl+C during the first parallel run. The launcher interrupted both jobs. The log showed only partial progress, not completed epochs.

**Resolution:** No data recovery was assumed. A later parallel run completed successfully and produced the reported checkpoints.

**Status:** RESOLVED operationally. Continue to use the run logs and checkpoint files to determine completion; clearing the terminal does not delete logs or checkpoints.

### 3.5 Fine-tuning must not overwrite the previous best checkpoint

**Problem:** User wanted ten more epochs from existing best weights, without early stopping after four non-improving epochs, and without replacing a better checkpoint with a worse one.

**Resolution:** Supplied a warm-start fine-tuning runner that initializes from the previous best checkpoint, resets AdamW optimizer state, carries over the source validation score as the baseline, trains the requested epochs, and only replaces `best_model.pt` on strict validation improvement.

**Status:** RESOLVED for the reported fine-tuning run. Note: this is warm-starting from model weights, not resuming the exact optimizer trajectory, because the imported best checkpoints did not contain optimizer state.

### 3.6 Checkpoint paths and finding results after terminal cleanup

**Problem:** User cleared the terminal and could not see the completed output.

**Resolution:** Training outputs were stored in timestamped subdirectories under `artifacts/two_stage_runs_parallel/`, including per-model logs, `best_model.pt`, `latest_checkpoint.pt`, training history and metadata.

**Status:** RESOLVED. Always inspect the latest run directory and the two log files; terminal scrollback is not the result store.

### 3.7 Misleading training-time retrieval metrics

**Problem:** Initial sampled retrieval metrics evaluated each positive against only 500 sampled negatives (501 candidates total). Those metrics looked much stronger than full-catalogue results.

**Resolution:** Added/used `scripts/evaluate_retrieval.py`, which ranks held-out positives against the indexed full catalogue and compares the retrieval model with shuffled users and popularity.

**Status:** RESOLVED as an evaluation capability. The full-catalogue result revealed a genuine quality/coverage issue (see current issues below).

### 3.8 Model performance selection cannot rely on training loss alone

**Problem:** During ranking and retrieval fine-tuning, training loss continued to decrease while validation ranking/retrieval metrics did not improve consistently.

**Resolution:** Keep the checkpoint selected by validation metric and compare AUC, per-user GAUC, NDCG, recall, MRR and calibration rather than selecting the final epoch by training loss.

**Status:** RESOLVED as an experiment-selection principle; use it in all new experiment plans.

---

## 4. Recorded experiment history

### 4.1 Early Pure V2 run on Windows

Reported command:

```powershell
python train_v2.py --data-dir data\processed\pure_v2 --output-dir artifacts\pure_v2_10ep --epochs 10 --batch-size 1024 --embedding-dim 32
```

Reported held-out test metrics:

- Test loss: 0.5802
- Mean AUC: 0.7784
- Click: 0.7505
- Like: 0.8757
- Follow: 0.8128
- Comment: 0.7359
- Forward: 0.7279
- Hate: 0.7653
- Long view: 0.7473
- Profile enter: 0.7781
- Complete view: 0.8182
- Short view: 0.7721

### 4.2 Ranking batch-size / architecture experiments on M4

The source ZIP contains multiple ranking experiment folders including batch sizes 256, 1024, 2048, 4096, 8192, 16384 and 32768, plus a quality-oriented 20-epoch configuration. The user's summary of these sweeps is that validation performance was broadly similar across batch sizes 1024–32768, so batch size is not currently the main quality lever.

The README describes a quality run with batch size 16384, max 20 epochs and patience 4. It reports best validation mean AUC 0.8210 and held-out test mean AUC 0.8107 (test loss 0.5507).

**Historical discrepancy:** The archived JSON artifact under `artifacts/ranking/1k_v2_m4_b16384_quality20/` contains a separate short history/test record with different values (including a test mean AUC of 0.7941). These should not be treated as the same run without checking original local run IDs/logs. Keep both records and resolve provenance before using either as the canonical reference.

### 4.3 Ranking smoke run

Reported one-epoch smoke training on MPS using batch size 4096:

- 2,535 training batches.
- Training loss: 0.5121.
- Validation loss: 0.5444.
- Mean validation AUC: 0.8029.
- Checkpoint saved under an `artifacts/ranking/1k_v2_smoke/` run folder.

This smoke run confirmed end-to-end operation, not final model quality.

### 4.4 Retrieval smoke run

Reported smoke training using 100,000 eligible positive rows, batch size 4096, one epoch:

- Model parameters: 92,732,511.
- Sampled Recall@20: 0.1719.
- Sampled Recall@100: 0.7031.
- MRR: 0.0638.

This was only a smoke test and not full-catalogue evaluation.

### 4.5 First successful parallel M4 full-data run

Run folder:

```text
artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/
```

Configuration:

- Model 2 ranking seed 42.
- Model 1 retrieval seed 43.
- Batch size 4096 for both.
- Maximum 20 epochs, with each model's configured early stopping.
- Both jobs ran concurrently on MPS.

**Ranking result:**

- Best validation mean AUC in the later epoch-7 record: 0.7971 (epoch 7 output shown in the pasted terminal log).
- Training loss at epoch 7: 0.4358.
- Validation loss at epoch 7: 0.5791.
- Stopped after four completed epochs without validation AUC improvement.
- Held-out test evaluation was explicitly skipped.

**Retrieval result:**

- Completed 14 epochs.
- Training loss at epoch 14: 0.21997.
- Sampled Recall@20: 0.4082.
- Sampled Recall@100: 0.7871.
- MRR: 0.1154.
- Stopped after four epochs without sampled Recall@100 improvement.

Both best checkpoints were saved. The run completed without a reported MPS out-of-memory error.

### 4.6 Warm-start fine-tuning run

Run folder:

```text
artifacts/two_stage_runs_parallel/parallel_finetune10_r42_m1s43_20261031_203323/
```

Ten additional epochs were attempted from the prior best model weights. Optimizer states were reset for fine-tuning.

**Ranking:**

- Best fine-tuned validation mean AUC: 0.8251948 at fine-tuning epoch 1.
- Original checkpoint baseline in the evaluation output: 0.8238 mean AUC.
- Fine-tune epoch 1 training loss: 0.46826.
- Fine-tune epoch 1 validation loss: 0.54388.
- Epoch 10 training loss: 0.45556; validation loss: 0.55626; mean AUC: 0.81614.
- Fine-tuned checkpoint improved mean AUC slightly, but did not improve NDCG compared with the original checkpoint.

**Retrieval:**

- Fine-tune training loss decreased from 0.24698 at epoch 1 to 0.21070 at epoch 10.
- Best fine-tuned sampled Recall@20: 0.40039.
- Best fine-tuned sampled Recall@100: 0.78125.
- Best fine-tuned sampled MRR: 0.11520.
- Original checkpoint had sampled Recall@100 about 0.78906, so the original checkpoint remained best according to that metric.

### 4.7 Full-catalogue retrieval evaluation

The user built a float16 index containing 3,929,859 videos (128-dimensional vectors; approximately 0.94 GiB) and evaluated 5,000 validation queries.

Reported full-catalogue Recall:

| Method | R@20 | R@100 | R@500 | R@1000 |
|---|---:|---:|---:|---:|
| Retrieval model | 0.0000 | 0.0006 | 0.0032 | 0.0054 |
| Shuffled users | 0.0002 | 0.0004 | 0.0012 | 0.0028 |
| Popularity | 0.0004 | 0.0016 | 0.0086 | 0.0148 |

The retrieval model beats shuffled users at larger K, but popularity beats the model at every listed K. This is a major reason to prioritize retrieval improvements before integration.

The evaluator only uses validation positives with a non-zero training-known video index. It reports that 53.14% of eligible positive interactions are on video IDs absent from the training vocabulary and excludes those positives from the retrieval metric. Thus the metric covers only the retrievable, training-known video subset.

The evaluator description also notes that known training positives are not removed from the competing catalogue, making absolute numbers slightly pessimistic; method comparisons are on the same query set.

### 4.8 Ranking validation evaluation, original versus fine-tuned

**Original checkpoint:**

- Mean AUC: 0.8238.
- Mean per-user AUC (GAUC): 0.6232.
- NDCG@10: 0.3977.
- NDCG@20: 0.5191.
- Watch-time log-seconds MAE: 0.7534; RMSE: 1.0826; R²: 0.3630.

**Fine-tuned checkpoint:**

- Mean AUC: 0.8252.
- Mean per-user AUC: 0.6251.
- NDCG@10: 0.3952.
- NDCG@20: 0.5164.
- Watch-time log-seconds MAE: 0.7523; RMSE: 1.0759; R²: 0.3709.

The original checkpoint had better NDCG despite slightly lower AUC. The fine-tuned model had a small AUC/GAUC and watch-time improvement. Selection should be based on the intended feed utility, not AUC alone.

The original current hand-weighted score had NDCG@10 0.3977 / NDCG@20 0.5191. The complete-view probability alone achieved 0.4055 / 0.5246 in the original evaluation and 0.4048 / 0.5239 for the fine-tuned checkpoint. This indicates that the current score weights need a validation-based utility search, not that complete-view-only scoring is automatically the correct product objective.

---

## 5. Current data findings and open problems

### 5.1 High new-video cold-start rate — OPEN / HIGH PRIORITY

Latest diagnostics:

- Validation rows with video ID OOV: 57.269%.
- Test rows with video ID OOV: 67.74%.
- Positive validation rows with video ID OOV: 53.139%.
- Positive test rows with video ID OOV: 64.807%.
- The diagnostic reports the same rate for “no video metadata”, because `pack_split()` currently uses `vi = ids_to_index(video_id, video_ids_from_train)` and then indexes `video_cat_lookup[vi]`. When `vi == 0`, all video categorical side features are also read from the OOV row.

**Clarification:** This is not currently evidence of a corrupt index or a mismatched model/index ID map. It follows from the deliberate chronological, training-only ID vocabulary. It is a real cold-start/catalogue-coverage limitation in the current preparation and retrieval design.

**Needed investigation/change:** Determine whether `video_features_basic_1k.csv` contains valid metadata for validation/test-only video IDs. If it does, extend preprocessing to preserve metadata for unseen video IDs while keeping their learned video-ID embedding at OOV index 0. Build a time-aware candidate catalogue from eligible videos with known metadata, respecting upload time so future items are not made available before they existed. If that metadata is absent, cold-start content retrieval needs another source of item content/features.

Do not simply map unseen raw IDs into the existing learned video-ID embedding table: those embedding rows were not trained and would be random/untrained.

### 5.2 Retrieval underperforms popularity on full catalogue — OPEN / HIGH PRIORITY

The current model's full-catalogue Recall@20/100/500/1000 is below the popularity baseline. The model is above the shuffled-user baseline for larger K, but still does not add value beyond popularity on these metrics.

Potential actions after the cold-start audit:

- Add correctly masked in-batch negatives.
- Preserve the existing sampled-negative baseline for direct comparison.
- Mask other known positive user-video pairs to prevent false negatives.
- Support multiple positives per user in a batch.
- Add content-feature handling for new videos.
- Use full-catalogue evaluation with 5,000+ queries, popularity and a robust different-user baseline.

### 5.3 Shuffled-user baseline may need stronger control — OPEN

The evaluator rolls query user vectors by a random shift. Since multiple queries can belong to the same user (5,000 queries but only around 1,000 users), a shifted query can sometimes receive the same user's vector. The report records `pct_shuffled_queries_with_same_user`; inspect that field. A stricter baseline should permute user IDs or pair queries with a different user, ensuring no same-user matches where possible.

### 5.4 Ranking score weights need tuning — OPEN

The hand-weighted score is currently hand-set. Ranking evaluation shows complete-view-only scoring slightly outperforms the current weighted score on logged-impression NDCG@10/@20, but this is not a full online-feed evaluation.

Needed:

- Search weights on validation only, using NDCG@10/@20 or a documented utility objective.
- Avoid optimizing directly on test.
- Calibrate probabilities before combining them.
- Include watch-time contribution only after validating its utility.
- Include negative feedback, but avoid allowing rare or poorly calibrated probabilities to dominate.

### 5.5 Ranking generalization and calibration — OPEN

Latest ranking validation:

- Mean AUC about 0.8252.
- Mean per-user AUC about 0.6251.
- Fine-tuned NDCG@10 0.3952, below original 0.3977.
- Like, follow and other rare targets show predicted/base-rate ratios above 10 in some cases.

Proposed controlled experiments:

1. Separate user-ID embedding dimension from video-ID embedding dimension.
2. Compare video-ID embedding dimension 32 versus 16.
3. Compare video-ID dropout 0%, 10%, 20% (start at 10%).
4. Compare positive class-weight cap 20 versus 3 and square-root weighting capped at 3.
5. Try weight decay 1e-4 before 1e-3.
6. Add a learning-rate decay schedule; do not combine multiple changes in one experiment.
7. Use per-user NDCG, GAUC, PR-AUC and calibration alongside mean AUC.

Changing video-ID embedding size changes the model shape and requires a new model trained from scratch; the 32-dimension checkpoint cannot be loaded directly into a 16-dimension video-ID table without an explicit migration strategy.

### 5.6 Test split must remain untouched during tuning — OPEN / PROCESS RULE

Use validation to select architecture, score weights and hyperparameters. Run the held-out test split once the candidate model/configuration is frozen. The full-catalog retrieval evaluator requires `--allow-test` before reading test data.

### 5.7 End-to-end two-stage evaluation not yet completed — OPEN / HIGH PRIORITY

Need a reproducible evaluator that:

1. Creates user query representations.
2. Retrieves top 500 or 1,000 eligible candidates from Model 1.
3. Constructs ranker inputs using correct impression/request-time features.
4. Scores candidates with Model 2.
5. Measures NDCG@10/@20, recall, diversity, coverage and latency.
6. Compares against popularity-only and ranking-only baselines.

The current `recommend_v2.py` and retrieval `retrieve.py` are offline tools, not yet a production request-serving system.

### 5.8 Website-specific event data not integrated — OPEN / PRODUCT REQUIREMENT

The eventual Instagram-style website should log impressions, watch duration, completion, skips, likes, saves, shares, follows, comments and explicit negative feedback. Saves are not currently a separate training target. Online history features, creator/topic affinity, freshness, filtering, and feedback-driven retraining are not complete.

### 5.9 Temporal/user metadata limitations — KNOWN LIMITATION

The preparation code excludes aggregate video statistics to reduce leakage, but uses static user features from the dataset without a precise as-of timestamp. This is documented in `metadata.json`. Assess this carefully before treating offline test metrics as fully production-realistic.

---

## 6. Decisions already agreed with the user

### Ranking proposals to implement as controlled experiments

- Keep batch size 4096 initially; prior batch-size sweeps did not show a clear quality advantage from larger batches.
- Use short training budgets when validation peaks early; a decaying learning rate is preferred to constant 0.001.
- Keep hidden layers 512 → 256 → 128 initially.
- Keep dropout 0.20 / 0.15 / 0.10 initially; raise first-layer dropout only if justified by experiments.
- Test video-ID embedding dimension 16 (separate from user-ID dimension).
- Add video-ID dropout beginning at 10%, then test 20%.
- Test class-weight cap around 3 versus current cap 20; do not remove weighting without comparison.
- Test weight decay 1e-4 before 1e-3.
- Keep watch-time loss weight 0.20 as the first comparison baseline.
- Replace hand-picked score weights with validation-based utility search and probability calibration.

### Retrieval proposals to implement as controlled experiments

- Add in-batch negatives with correct multi-positive and false-negative masking.
- Keep existing two sampled negatives as a baseline and compare hybrid vs in-batch-only approaches.
- Keep temperature 0.10 for the first in-batch experiment, then test 0.05–0.10.
- Keep batch size 4096 initially; test larger batches only if memory and quality justify them.
- Use 5,000+ full-catalogue validation queries for serious evaluation. Retain 512-query sampled validation for quick per-epoch monitoring.

---

## 7. Recommended next execution order

1. **Re-prepare 1K into a new versioned output directory** with the cold-start catalogue patch; do not overwrite `1k_v2`.
2. Run data diagnostics and verify actual basic-video metadata coverage for unseen validation/test IDs.
3. Rebuild the full-catalogue index from the existing retrieval checkpoint and rerun 5,000-query validation against model, popularity and shuffled users.
4. Train the optional in-batch retrieval variant from the original best checkpoint; compare with the sampled-negative baseline on the identical evaluation queries.
5. Run ranking controlled experiments (ID dimension/dropout first; class weighting and weight decay separately).
6. Search ranking score weights on validation using NDCG and a documented product utility.
7. Evaluate the complete retrieval → ranking pipeline.
8. Freeze one candidate configuration and perform a final held-out test evaluation.
9. Only then begin website integration and online event collection.

---

## 8. Reproduction / location notes

From the user's Mac:

```bash
cd ~/Projects/reels-recommender-m4
source .venv/bin/activate
```

Important folders:

- Source code: `common/`, `model1_retrieval/`, `model2_ranking/`, `scripts/`
- Processed data: `data/processed/ranking/1k_v2/`
- Retrieval index: `data/indexes/retrieval/`
- Training runs: `artifacts/two_stage_runs_parallel/`
- Evaluation reports: `reports/experiments/`

Keep the following outside source-only ZIPs unless deliberately requested: raw datasets, processed NPZ arrays, large vector indexes and `.pt` checkpoints. Back these up separately because they are not guaranteed to be present in a code archive.

---

## 9. Maintenance rule for future patches

Every future code patch ZIP must include this file at the project root:

```text
PROJECT_ISSUES_AND_RESOLUTIONS.md
```

Update it whenever a problem is encountered, a fix is applied, a result changes the project assessment, or an old issue is superseded. Record exact commands, paths, outcomes and whether the fix has been verified on the user's machine. Do not mark a problem RESOLVED just because a patch was generated; mark it VERIFY until the user confirms it works.


---

## 10. Latest full-catalogue and ranking evaluation results (2026-10-31)

The user ran the evaluator commands on the Mac mini M4. These results are from the original retrieval and ranking checkpoints on the validation split; they are not held-out test results.

### Retrieval full-catalogue evaluation

- Catalogue size: 3,929,859 videos.
- Query count: 5,000 held-out positive interactions.
- 243,623 eligible positive rows were considered by the evaluator.
- 53.14% of those positives were for a video ID not present in the model's training vocabulary.
- Model Recall@20 / @100 / @500 / @1000: 0.0000 / 0.0006 / 0.0032 / 0.0054.
- Shuffled-user Recall@20 / @100 / @500 / @1000: 0.0002 / 0.0004 / 0.0012 / 0.0028.
- Popularity Recall@20 / @100 / @500 / @1000: 0.0004 / 0.0016 / 0.0086 / 0.0148.
- Conclusion: the original retrieval model beats shuffled users at larger cutoffs but falls below popularity at all listed cutoffs. This is a retrieval-quality blocker; investigate catalog coverage and representation of cold-start videos before claiming useful full-catalogue personalization. The evaluator does not filter already-seen training positives from competing catalogue items, so absolute recall is conservative; comparisons on identical queries are fair.

### Ranking full validation evaluation

Original checkpoint:
- Mean AUC 0.8238; mean per-user AUC 0.6232.
- NDCG@10 0.3977; NDCG@20 0.5191 using current hand-weighted score.
- Complete-view-only score NDCG@10 0.4055; NDCG@20 0.5246.
- Watch-time log-seconds MAE 0.7534, RMSE 1.0826, R2 0.3630.

Fine-tuned checkpoint:
- Mean AUC 0.8252; mean per-user AUC 0.6251.
- NDCG@10 0.3952; NDCG@20 0.5164 using current hand-weighted score.
- Complete-view-only score NDCG@10 0.4048; NDCG@20 0.5239.
- Watch-time log-seconds MAE 0.7523, RMSE 1.0759, R2 0.3709.

Conclusion: fine-tuning very slightly improved AUC and watch-time regression but slightly worsened the current hand-weighted NDCG. Keep both checkpoints and choose based on the intended validation ranking objective; do not assume mean AUC alone determines the best feed ranking. Current class weighting can inflate predicted positive rates and needs calibration/utility assessment.

### Data diagnostics

- Validation interactions with unseen training video IDs: 57.269%; test: 67.74%.
- Positive validation interactions on unseen IDs: 53.139%; test: 64.807%.
- Top 1% of train videos account for 17.22% of training rows; top 10% account for 50.596%.
- Train rows per user: 1,000 users, minimum 85, median 7,389, maximum 106,642.
- Event rate for `is_hate` changes from 0.101% in train to 0.028% validation and 0.033% test; monitor this distribution shift.

## 11. Cold-start catalogue and in-batch retrieval patch status (2026-10-31)

A source patch has been assembled with the following changes:

- `prepare_data_v2.py` produces a separate `catalog_maps.npz` from the basic video feature catalogue and retains metadata for catalogued video IDs that are OOV in the training video-ID embedding table. Train-fitted category vocabularies remain train-only; unseen categories map to OOV.
- Prepared splits carry a catalogue item index separately from the model's video-ID embedding index. This distinguishes a cold-start video with metadata from a video with no catalogue row.
- Retrieval training supports `--negative-mode in-batch` with deduplicated batch candidates and a multi-positive mask based on known positive user-video pairs from the training set. The sampled-negative mode remains available as the comparison baseline.
- The retrieval index/evaluator records and checks the catalogue-map fingerprint. New catalogue mappings require a new index.
- Ranking candidate scoring can use preserved metadata for catalogued OOV videos; freshness features can use current UTC time or an explicit `--as-of-ms`.
- Data diagnostics now separately report unseen video ID, unseen ID with catalogue metadata, and no catalogue metadata.

### Verification completed in the patch workspace

- Python compilation completed for `common/`, `model1_retrieval/`, `model2_ranking/`, `scripts/`, and `tests/`.
- All 25 unit tests passed.
- Synthetic end-to-end checks covered preparation with OOV video IDs and retained metadata, in-batch retrieval training, catalogue-index construction, full-catalogue evaluation, and retrieval-to-ranking candidate scoring.
- These checks used synthetic fixtures in the patch workspace. They are not full KuaiRand-1K training/evaluation and have not yet been verified on the user's Mac.

### Required user-side verification

1. Install the patch.
2. Re-prepare KuaiRand-1K into a new directory such as `data/processed/ranking/1k_v3_catalog`; do not overwrite `1k_v2`.
3. Run `scripts.diagnose_data` and inspect how many unseen positive IDs have basic-video catalogue metadata and how many have no catalogue row.
4. Rebuild the retrieval index using the existing retrieval checkpoint and the new prepared dataset. If a fingerprint/schema compatibility check fails, stop and train a new compatible checkpoint rather than bypassing the check.
5. Rerun 5,000-query full-catalogue validation. Compare model vs popularity and shuffled users.
6. Only after saving that expanded-catalogue baseline, train the in-batch variant from the original retrieval weights with a fresh optimizer and compare it on the same query set.

### Remaining limitations

- The patch cannot invent metadata for video IDs absent from the basic-video feature file. Those remain true metadata cold-start items and need a fallback strategy (e.g., content ingestion or popularity/trending fallback).
- In-batch negatives are only as correct as the known-positive table. Unobserved videos are treated as sampled negatives, not asserted dislikes.
- In-batch training uses a larger similarity matrix and must be smoke-tested for MPS memory on the 16 GB Mac.
- The 5,000 retrieval queries are held-out positive interactions, not necessarily 5,000 unique users.
- The current offline dataset has about 1,000 users and is not sufficient by itself to prove generalization to a production Instagram-scale population.
- The source patch is not yet confirmed on the user's Mac; do not mark the new cold-start and in-batch changes RESOLVED until the user confirms real-data preparation, index build, evaluation and training.

## 12. Expanded-catalogue retrieval baseline confirmed on the Mac (2026-10-31)

The user successfully installed the cold-start/catalogue patch, re-prepared KuaiRand-1K into `data/processed/ranking/1k_v3_catalog`, built the original retrieval checkpoint's expanded index, and completed 5,000-query validation evaluation.

### Actual data-preparation result

- `train.npz`: 10,381,632 rows; all 10,381,632 have a basic-video catalogue row.
- `validation.npz`: 665,706 rows; all 665,706 have a basic-video catalogue row.
- `test.npz`: 665,707 rows; all 665,707 have a basic-video catalogue row.
- Full basic-video catalogue: 4,371,868 videos.
- Validation: 57.269% of interaction rows use a video ID outside the training embedding vocabulary; 53.139% of positive rows use such IDs; all have basic-video metadata.
- Test: 67.74% of interaction rows use an unseen training video ID; 64.807% of positive rows use such IDs; the diagnostics reported no missing basic-video catalogue metadata in the interaction rows.
- Important: metadata coverage is not equivalent to every metadata field being populated or having a non-OOV category token.

### Expanded-catalogue full validation retrieval result

Index: `data/indexes/retrieval/1k_v3_catalog_original`, 4,371,868 item vectors, 128 dimensions, float16 (~1.04 GiB).
Evaluation: 5,000 held-out positive-interaction queries against the full catalogue, fixed default seed 314159.

| Method | Recall@20 | Recall@100 | Recall@500 | Recall@1,000 | MRR | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| Retrieval model | 0.0002 [0.000, 0.001] | 0.0012 [0.000, 0.002] | 0.0054 [0.003, 0.007] | 0.0104 [0.008, 0.013] | 0.0001 | 235,466 |
| Shuffled users | 0.0000 [0.000, 0.000] | 0.0000 [0.000, 0.000] | 0.0004 [0.000, 0.001] | 0.0014 [0.000, 0.003] | 0.0000 | 644,178 |
| Popularity | 0.0002 [0.000, 0.001] | 0.0008 [0.000, 0.002] | 0.0034 [0.002, 0.005] | 0.0062 [0.004, 0.008] | 0.0002 | 3,004,880 |

Interpretation:

- The retrieval model is now above the popularity baseline by point estimate at Recall@100, @500 and @1,000; the largest relative point-estimate difference is about 1.68x at @1,000.
- At @500 and @1,000, confidence intervals overlap or touch near their boundaries, so this is promising but not a definitive statistically separated win.
- The model is much stronger than shuffled-user retrieval at @500 and @1,000, which supports that user representations are adding personalization signal.
- Recall@20 remains extremely small and approximately tied with popularity at the displayed precision. Top-of-feed retrieval still needs improvement.
- MRR's point estimate is lower than popularity despite a much better median rank. Do not summarize quality with one metric; inspect the full rank distribution and top-K recall together.
- Do not compare this result naively with the legacy 3,929,859-item evaluation: that evaluator excluded the 53.14% cold-start positive rows. The v3 evaluation now includes metadata-backed unseen-ID positives, so the query population changed.

### Next retrieval experiment

The next experiment is a controlled warm-start from the original retrieval checkpoint, using `--negative-mode in-batch`, with a fresh AdamW optimizer. Keep the original checkpoint and v3 original index unchanged. Use the same 5,000-query full-catalogue validation seed (314159) after training for a paired comparison.

A code adjustment was added to the next retrieval-training patch: on `--init-from`, the trainer re-evaluates the imported checkpoint on the current prepared data's exact sampled validation queries and uses that result as the starting checkpoint-selection baseline. This avoids comparing new v3 epochs to a sampled validation metric saved on the older v2 catalogue.

The next trainer patch also adds an optional cosine learning-rate schedule. The planned M4 experiment uses 10 maximum epochs, batch size 4096, in-batch negatives, temperature 0.10, a conservative warm-start learning rate of 0.0003 decaying to 0.00005, and no early stopping (`--early-stopping-patience 0`). Checkpoint saving still only replaces the best model when sampled Recall@100 strictly improves by `min_delta`.

Verification for this trainer adjustment: Python compilation and all 25 unit tests passed; a 2-epoch synthetic warm-start in-batch run confirmed that current-data baseline re-evaluation, the cosine scheduler, checkpoint saving and resume-artifact writing work. Full KuaiRand-1K/MPS smoke test and training remain to be performed by the user.


## 13. Full in-batch retrieval experiment regressed — 2026-11-01

The user completed a full 10-epoch warm-start run on `1k_v3_catalog` with:

- Source checkpoint: original sampled-negative retrieval `best_model.pt` (source epoch 10).
- `--negative-mode in-batch`, batch size 4096.
- 3,907,757 eligible positive interactions; 955 batches per epoch.
- AdamW learning rate 0.0003 with cosine decay to 0.00005.
- Temperature 0.10, hard-negative candidate sample rate 0.25.
- Fixed sampled validation: 512 queries x 500 random negatives.

Observed sampled validation Recall@100:

| State | Recall@100 |
|---|---:|
| Imported checkpoint, re-evaluated on v3 | 0.7891 |
| Epoch 1 | 0.4844 |
| Epoch 2 | 0.3809 |
| Epoch 3 | 0.3496 |
| Epoch 4 | 0.3125 |
| Epoch 5 | 0.3008 |
| Epoch 6 | 0.2891 |
| Epoch 7 | 0.2773 |
| Epoch 8 | 0.2773 |
| Epoch 9 | 0.2812 |
| Epoch 10 | 0.2773 |

Training loss fell from 1.75313 to 1.26562. It is not directly comparable with the original pairwise softplus loss because the objective changed. MPS memory remained low (about 1.39 GiB current, 3.03 GiB driver after validation); this was not a memory failure.

The trainer retained the imported epoch-0 checkpoint as `best_model.pt`, because no epoch exceeded its starting sampled Recall@100. The final in-batch weights are in `latest_checkpoint.pt`. Therefore the subsequent full-catalogue evaluation using `best_model.pt` exactly matched the original model; it did not evaluate the final in-batch weights.

### Code audit of uploaded debug source

Files inspected: `model1_retrieval/train.py`, `model1_retrieval/data.py`, `model1_retrieval/model.py`, `common/data_version.py`, and this issue log.

Findings:

- `build_inbatch_candidates()` deduplicates batch item IDs and constructs a per-user multi-positive mask using the full sorted known-positive user-item key set. No obvious false-negative masking bug was found in this function.
- The in-batch loss uses a multi-positive log-sum-exp numerator over known positive batch candidates and a log-sum-exp denominator over all candidates. This is a valid multi-positive contrastive formulation in principle.
- The warm-start resets AdamW optimizer state, which is intentional, but the objective changes from sampled pairwise softplus to in-batch multi-positive contrastive loss.
- The 32,768-row smoke run made only 8 optimizer updates. The full run made 955 updates in its first epoch (about 119 times more updates than the entire smoke run) at the same 0.0003 learning rate. The smoke metric delta 0.7891 to 0.7988 on only 512 validation queries is small and not sufficient evidence of improvement.
- The full run showed severe degradation after the first epoch. The evidence is most consistent with objective/optimization mismatch and warm-start forgetting, not MPS memory exhaustion. This is a working diagnosis, not a proven singular root cause.
- Training-time sampled Recall@100 uses 512 queries and 500 uniform sampled negatives. It is not a proxy for exact full-catalogue quality. The full-catalogue evaluator must remain the final comparison.

### Current status and next action

- Original sampled-negative checkpoint remains the strongest validated retrieval checkpoint.
- Pure in-batch warm-start training is marked **FAILED TO IMPROVE / DO NOT REPEAT WITH SAME CONFIGURATION**.
- A conservative hybrid experiment is proposed: retain the original sampled pairwise objective as the main loss and add a small multi-positive in-batch auxiliary loss. Start with a much smaller warm-start learning rate (3e-5), a small in-batch loss coefficient (0.05), and a short 1-epoch medium-size diagnostic run before any full-data run.
- Keep checkpoints in new experiment directories. Never overwrite original or smoke-test artifacts.
- Compare full-catalogue validation Recall@20/100/500/1000 and MRR against the original and popularity baselines. Do not promote using only the 512-query sampled metric.


### Hybrid-loss corrective patch prepared

A corrective training option was added to `model1_retrieval/train.py`:

- New `--negative-mode hybrid` keeps the existing sampled pairwise softplus objective and adds the multi-positive in-batch contrastive loss as an auxiliary term.
- Default auxiliary coefficient is 0.05 (95% pairwise + 5% in-batch blend); configurable with `--inbatch-loss-weight`.
- Hybrid history records pairwise and in-batch loss components separately.
- Hybrid resumes validate the auxiliary-loss coefficient; old sampled/in-batch resume configs remain compatible.
- Existing `sampled` and `in-batch` modes are unchanged.

Verification performed in the patch-building environment:

- `py_compile` passed for updated trainer and new test.
- 3 new hybrid-loss unit tests passed.
- A tiny synthetic one-epoch hybrid run completed on CPU, including warm training steps, validation, checkpoint writing and metadata writing. This synthetic test validates the code path only, not model quality or MPS performance.

Suggested next real-data test on the Mac: warm-start from the original sampled-negative checkpoint with learning rate 3e-5, 5% in-batch auxiliary weight, one epoch and 131,072 eligible positive rows (32 updates). Use a larger fixed validation query set (2,048 queries) to reduce metric quantization. Do not run full 10-epoch in-batch-only training again with the prior configuration.

## 14. Retrieval diagnosis patch — raw/filtered and cohort breakdowns (2026-11-01)

The latest full-catalogue results confirm that the original retrieval model remains the best validated checkpoint. Before another architecture or loss change, the next step is to isolate the failure modes.

### Findings from the latest source audit

- The full-catalogue evaluator ranks each held-out positive against all catalogue items, including other positive items previously observed for the same user during training. This is explicitly documented in `scripts/evaluate_retrieval.py`.
- The retrieval training sampler rejects known-positive user-item pairs from sampled negatives. Thus training and the unfiltered evaluation candidate sets differ in this respect.
- The dataset has 53.14% of positive validation queries on video IDs unseen during training, although their catalogue metadata is available. These videos have OOV video-ID embeddings and rely on the video side features.
- The user tower uses user ID/profile and request context features; it does not currently encode an ordered sequence or aggregate of recent watched videos. This is a structural limitation for a personalized short-video feed, especially for changing interests/session context.
- The current evaluator samples multiple query rows per user. Its existing bootstrap intervals are per query, not clustered by user. The new diagnostic reports user-cluster bootstrap intervals.
- The current popularity and model metrics are not sufficient to distinguish cold-start, long-tail and seen-item competition effects. Cohort diagnostics are needed before selecting another training objective.

### New diagnostic capability

Added `scripts/diagnose_retrieval.py` and `tests/test_retrieval_diagnostics.py`.

The script:

- Uses the same validation query sampling seed and full-catalogue index as the existing evaluator.
- Reports raw full-catalogue ranks and filtered ranks after removing the user's other known-positive training items. The held-out target itself is retained, even if it appeared in training for that user.
- Applies the same raw/filtered candidate definitions to the popularity baseline.
- Breaks results down by metadata-backed unseen video IDs, training interaction exposure cohorts, and positive event types.
- Reports query-weighted means with user-cluster bootstrap confidence intervals for Recall@K.
- Does not modify model checkpoints, indexes, or the existing benchmark evaluator.

### Verification

- All project Python modules compile in the patch workspace.
- All 32 unit tests pass, including four new tests for filtered rank accounting and user-cluster bootstrap behavior.
- The new script has not yet been run against the user's local prepared KuaiRand arrays/indexes. That real-data run is the next required validation.

### Next action

Run the diagnostic against both the original checkpoint/index and the hybrid epoch-1 checkpoint/index using the same 5,000-query validation sample and seed 314159. Compare cold-start versus training-known items, head/tail exposure cohorts, and raw versus filtered Recall@K. Do not start another full training run until these reports identify which failure mode is dominant.


## 15. Source archive audit and evaluation correctness patch (2026-11-01)

### Scope and reproducibility

Reviewed the complete source archive supplied from the Mac (65 files before removing the already-verified duplicate diagnostics patch directory from this delivery). The archive intentionally excludes local datasets, checkpoints, vector indexes, `.venv`, and Git internals. Consequently, source-level tests can be run here, but data-dependent training and full-catalogue evaluation must be repeated on the Mac.

Before this patch, the repository test suite passed 32 tests.

### Fixes implemented

1. **Exact-score tie handling in full-catalogue retrieval.** `common/retrieval_eval.py` previously counted only scores strictly greater than the target. Equal-score catalogue items were ignored, yielding an optimistic rank when several items shared the same vector/score (especially relevant to OOV/cold-start representations). It now returns expected rank: `1 + greater + 0.5 * other_equal`.
2. **Consistent filtered diagnostic accounting.** `scripts/diagnose_retrieval.py` now removes one full rank position for each filtered strictly-higher known-positive item and half a position for each filtered tied item. It retains the held-out target.
3. **User-cluster confidence intervals in the main evaluator.** `scripts/evaluate_retrieval.py` previously bootstrapped individual query rows, although many rows belong to the same user. It now uses a query-weighted user-cluster bootstrap for Recall@K and MRR and records the method/unique-user count in the report.
4. **Prepared-array input validation.** `model1_retrieval.data.load_split` now rejects negative categorical/catalogue indices, non-finite numeric features, and non-finite/out-of-range labels before training/evaluation.
5. **Regression tests.** Added tests for tie-aware ranks, filtered tie accounting, reproducible user-cluster bootstrap, and invalid prepared arrays.

### Verification

- `python -m pytest -q`: **39 passed** in the extracted source workspace.
- Updated Python modules compile successfully.
- No actual KuaiRand training, checkpoint load, MPS run, or full-catalogue evaluation was performed in this environment because the archive did not include the corresponding data/checkpoints/indexes.
- The existing 32 tests passed before changes; the patch adds seven test cases, bringing the suite to 39.

### Findings deliberately not changed without data-backed validation

- The user tower still has no ordered/recent watch-history representation. The prepared NPZ schema currently omits per-row timestamps/history sequences. Implementing history safely requires changing preparation, split schema, training, validation/inference feature construction, fingerprints and tests together; simply adding a history embedding would risk temporal leakage and train/serve mismatch.
- User/video feature tables are static snapshots with unknown as-of times. This remains a documented temporal-leakage uncertainty, not a proven leak.
- Training checkpoint selection still uses sampled Recall@100 over a small fixed query/candidate set. Full-catalogue validation remains the required promotion gate; replacing the training selector with a costly multi-million-item scan every epoch needs a deliberate compute/quality design.
- The current full-catalogue recall is low, particularly in the tail. The previously tested pure in-batch and hybrid loss variants did not improve the overall full-catalogue baseline and are not promoted.
- Model 2 score weights remain heuristic and are not calibrated.
- No change to model architecture, loss weights, optimizer defaults, data split policy, or checkpoint contents was made in this patch. No model-quality improvement is claimed.

### Next required actions on the Mac

1. Install this source patch without replacing `best_model.pt`, prepared data, or the original retrieval index.
2. Activate the existing project `.venv` and run `python -m pytest -q`.
3. Run the tie-aware baseline evaluation and diagnosis commands in the README against the preserved original checkpoint/index. The metrics will differ slightly from older reports because exact ties are now split evenly and confidence intervals are user-clustered.
4. Only after those reports are recorded should we design the chronological watch-history schema and a new retrieval experiment. Keep the original checkpoint/index as the rollback baseline.


## 16. Retrieval v3 — root-cause analysis and new trainer (2026-11-01)

**Status labels:** all code below is **VERIFY on the user's Mac**. It was written in an environment with no PyTorch and no data, so torch code was not executed there.

### Evidence from the existing reports (not new experiments)

1. **The in-batch full-catalogue result was never a test of the in-batch weights.** `retrieval_validation_1k_v3_inbatch_seed43.json` has `checkpoint_best_epoch: 0` and numbers identical to the original model. The trainer kept the imported epoch-0 weights as `best_model.pt` because its selection metric (sampled Recall@100, 512 queries x 500 uniform videos) fell for the new objective. `latest_checkpoint.pt` (epoch 10) was never indexed or evaluated on the full catalogue.
2. **The selection metric rewards the failure mode.** Uniform negatives are mostly videos that the model has pushed down as training negatives. A model that memorises "rare IDs are negatives" scores well on it, so it blocked every alternative objective (in-batch, hybrid).
3. **Cohorts (retrieval_diagnosis_corrected.json, 5,000 queries):** 54.4% of queries are on videos unseen in training. Recall@1000 by training exposure of the positive video: unseen-ID 1.65%, 1-4 exposures 0.11%, 5-19 exposures 0.12%, 20-99 exposures 0.82%, 100+ exposures 1.8% (n=56). Rarely seen IDs are the worst cohort even though they are close to half of the positives.
4. **Mechanism (code reading, not proven by ablation):** `sample_negative_video_indices` draws negatives uniformly from the 4.37M catalogue (about 1.8 negative draws per video per epoch), while a video with 1-4 exposures has under one positive per epoch. Its ID embedding therefore learns to be a negative. In training all positives have a known video ID, while about 10% of sampled negatives (catalogue videos unseen in training) have the all-zero ID block, so the OOV path is trained only as a negative. Neither tower sees time, and the catalogue includes videos not yet uploaded at query time.
5. **The benchmark itself:** ranking against 4.37M videos includes items that did not exist or were inactive at the query time. A time-windowed pool is the realistic comparison and keeps the old protocol available.

### Changes (this patch)

* `model1_retrieval/model.py`: video tower gets `id_keep` buffer (IDs outside it behave as OOV) and training-time `id_dropout`. Legacy checkpoints load with every ID kept (verified by a test, torch-run pending).
* `model1_retrieval/train_v3.py`, `losses.py`, `common/pools.py`: time-local batches, same-day pool negatives, logQ-corrected sampled softmax with the user's other known positives excluded, ID gating by training frequency (default >= 20 exposures), pool-based checkpoint selection (hit rate in the top 1% of the day's pool).
* `scripts/evaluate_retrieval_pools.py`: pool protocol with shuffled-user, previous-window popularity and user-author-affinity baselines, user-cluster bootstrap, cold/seen split.
* `scripts/add_row_times.py`: writes `row_times.npz` (new file; fingerprints unchanged) after verifying row-for-row alignment with the prepared arrays.
* Tests: `test_pools.py`, `test_add_row_times.py` (run without torch), `test_retrieval_v3.py` (torch vs NumPy reference loss, gating, legacy loading, gradients).

### Verified in the patch workspace

* 34 torch-free tests pass (42 including skipped torch tests). Pool helpers, tie-aware pool ranks, the known-pair bit matrix (brute-force comparison), batch builder, loss reference, row-time alignment (including rejection of scrambled/short data) and the real `read_log` path on tiny CSVs.
* `evaluate_retrieval_pools.py` and the trainer's `PoolValidator` ran end to end on synthetic data with a stand-in for torch: planted signal found, shuffled users near chance, reversed user vectors score about 0.97 mean percentile.
* Static undefined-name check over all modules.

### NOT verified

* Any torch execution: `FeatureTower` changes, `pool_softmax_loss`, the training loop, MPS speed/memory, real KuaiRand numbers. Run `tests/test_retrieval_v3.py` first, then a smoke run.
* Whether v3 beats the original, popularity or author affinity. No improvement is claimed.
* Videos absent from the basic-video table (the v3 diagnostics found none among interaction rows) would remain unindexable.

### Next actions

1. Install patch, run the tests (README_RETRIEVAL_V3.md step 0), create `row_times.npz`.
2. Record the pool-protocol baseline for the original checkpoint (also shows how strong author affinity is).
3. Smoke run, then the full 8-epoch run; compare on the pool report and the full-catalogue report.
4. If author affinity still wins: add a leak-free recent-positive-history input to the user tower. If a v3 model wins: restrict `retrieve.py` to the active pool and re-run the end-to-end two-stage evaluation.
5. Optional free check: index `latest_checkpoint.pt` of the in-batch run and evaluate it on the full catalogue (finding 1).
