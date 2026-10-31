# Cold-start catalogue and in-batch retrieval patch

This patch addresses the retrieval evaluation result where the model fell below the training-popularity baseline and over half of positive validation rows used a video ID outside the training vocabulary.

## What changed

1. `prepare_data_v2.py` still fits video-ID and metadata category vocabularies from training-period IDs only.
2. It writes a new `catalog_maps.npz` using all raw video IDs present in `video_features_basic_*.csv`.
3. Every interaction split gains `catalog_video_indices`. The model video-ID embedding remains 0 for an unseen ID, but known author/type/music/tag/duration/width/height metadata is preserved and encoded using the training-only category vocabularies.
4. Retrieval training, sampled validation, full-catalogue evaluation, index construction, and recommendation candidate scoring use catalogue indices rather than assuming that every catalogue row has a train video-ID embedding.
5. Full-catalogue evaluation reports separately: positives on cold-start video IDs that have metadata, and positives with no basic-video catalogue row.
6. `--negative-mode in-batch` adds deduplicated in-batch candidate videos with multi-positive masking. The mask is built against all known positive (user, video) pairs in the training set, so a user's other known positive videos in the batch are not incorrectly treated as negatives. Existing `--negative-mode sampled` remains the default and is unchanged.
7. The ranking example recommender can score metadata-backed cold-start candidates and accepts `--as-of-ms` for reproducible freshness/video-age features; by default it uses current UTC time.
8. `PROJECT_ISSUES_AND_RESOLUTIONS.md` is included and updated.

## Important compatibility details

- The original `id_maps.npz` train-ID maps and model categorical embedding sizes are kept unchanged. The separate `catalog_maps.npz` is the expanded item catalogue.
- Existing ranking checkpoints can be evaluated against the newly prepared data if the schema fingerprint and categorical sizes still match; their predictions for unseen videos can now use retained metadata.
- Existing retrieval checkpoints can be warm-started with `--init-from` only when the dataset, schema fingerprint and categorical embedding sizes still match. The warm start resets the optimizer. If the new schema check fails, do not bypass it; train a new retrieval checkpoint for the revised feature mapping.
- Existing retrieval indexes must be rebuilt after using `catalog_maps.npz`. The catalogue-map SHA-256 is checked by `evaluate_retrieval` and `retrieve.py`.
- Keep the old `data/processed/ranking/1k_v2` directory, existing indexes, checkpoints and reports. Prepare the revised dataset in a new directory.

## 1. Apply the patch

From the project root with the virtual environment active:

```bash
unzip -o ~/Downloads/cold-start-inbatch-retrieval-patch.zip -d .
source .venv/bin/activate
python -m py_compile common/*.py model1_retrieval/*.py model2_ranking/*.py scripts/*.py tests/*.py
python -m unittest discover -s tests -v
```

## 2. Re-prepare KuaiRand-1K into a NEW directory

Do not overwrite the old prepared data:

```bash
python -m scripts.prepare_data_v2 \
  --dataset 1k \
  --data-dir data/raw/KuaiRand-1K/data \
  --output-dir data/processed/ranking/1k_v3_catalog
```

The script reports the percentage of each split whose video has a row in the basic-video feature table.

## 3. Re-run diagnostics

```bash
python -m scripts.diagnose_data \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --output reports/experiments/data_diagnostics_1k_v3_catalog.json
```

Check the difference between `unseen video ID`, `unseen but metadata available`, and `no video metadata`. The latter is the true catalogue coverage limitation in this patch.

## 4. Build a metadata-backed index with the existing retrieval checkpoint

Use this only if the data-preparation output preserves the old train ID/schema mapping. The script checks the checkpoint fingerprint and fails safely if it does not.

```bash
python -m model1_retrieval.build_index \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --model-path artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
  --output-dir data/indexes/retrieval/1k_v3_catalog_original
```

Then evaluate the original checkpoint against the expanded catalogue:

```bash
python -m scripts.evaluate_retrieval \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --model-path artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
  --index-dir data/indexes/retrieval/1k_v3_catalog_original \
  --split validation --queries 5000 \
  --output reports/experiments/retrieval_1k_v3_catalog_original_validation.json
```

Compare model vs popularity and shuffled-user Recall@20/100/500/1000. The report also says how many positive validation rows have metadata and can be retrieved.

## 5. Train the in-batch retrieval variant

Only after the original-checkpoint expanded-catalogue evaluation is saved:

```bash
python -m model1_retrieval.train \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --output-dir artifacts/retrieval/1k_v3_inbatch_seed42 \
  --init-from artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model1_retrieval/best_model.pt \
  --epochs 10 --batch-size 4096 \
  --negative-mode in-batch \
  --temperature 0.10 \
  --hard-negative-sample-rate 0.25 \
  --eval-queries 512 --eval-negatives 500 \
  --seed 42 --eval-seed 314159 \
  --early-stopping-patience 4
```

This is a warm start of model weights with a fresh optimizer, not a bitwise resume. `--negative-mode sampled` remains the baseline for a controlled comparison. On the M4, monitor memory for the first epoch; the in-batch similarity matrix uses substantially more temporary memory than two sampled negatives.

After training, build a new index from the new checkpoint and run the same 5,000-query full-catalogue evaluation. Do not overwrite the old index or checkpoint.

## 6. Ranking model compatibility check

The ranking network architecture and weights are unchanged in this patch. The new prepared validation rows contain metadata for eligible unseen IDs where present, so the existing ranker can be evaluated against the revised feature arrays:

```bash
python -m scripts.evaluate_ranking \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --model-path artifacts/two_stage_runs_parallel/parallel_r42_m1s43_20261030_225836/model2_ranking/best_model.pt \
  --split validation \
  --output reports/experiments/ranking_original_1k_v3_catalog_validation.json
```

The patch does **not** yet implement the agreed ranking experiments (16-d video ID embedding, video-ID dropout, positive-weight cap comparison, weight-decay sweep or learned ranking-score search). Those remain the next isolated ranking experiment after retrieval catalogue evaluation.
