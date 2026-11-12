# Two-stage Reels recommender

An offline PyTorch prototype for personalized short-video recommendations using KuaiRand-1K. The project has two learned stages plus a lightweight creator-affinity candidate source:

1. **Model 1 — retrieval:** a two-tower model retrieves likely videos from a large catalogue.
2. **Static author affinity:** a user's prior positive interactions with creators provide complementary candidates, especially for videos whose own ID embedding is unavailable.
3. **Model 2 — ranking:** a multi-task ranker scores the union. The selected offline scoring formula for the current top-100 objective is the `is_like` sigmoid output.

This repository is **ready for API integration, not production deployment**. It does not yet contain a live feed API, real-time event pipeline, feature store, online training loop, ANN serving layer, policy/ranking filters, monitoring, or online experiment framework.

## Current selected artifacts

Use these paths together. Do not mix checkpoints, feature maps, prepared-data versions, or indexes from different runs.

| Component | Selected path |
|---|---|
| Prepared data | `data/processed/ranking/1k_v3_catalog/` |
| Model 1 checkpoint | `artifacts/retrieval/v3_linear_output_seed42_14ep/best_model.pt` |
| Model 1 vector index | `data/indexes/retrieval/1k_v3_v3_linear_output_seed42_14ep/` |
| Model 2 checkpoint | `artifacts/two_stage_runs_parallel/parallel_v3catalog_warmstart_r42_20261101_203431/model2_ranking/warmstart_v3_catalog_baseline.pt` |
| Primary ranking formula | `head:is_like` (the `is_like` sigmoid probability) |

The previous pool-softmax V3 checkpoint `artifacts/retrieval/v3_full_seed42_12ep/best_model.pt` is a reference/rollback model, not the selected retrieval model. The folder name says `12ep`, but its metadata records 14 training epochs. Keep it only while a rollback/reference checkpoint is useful.

The Model 2 `best_model.pt` in the V3 warm-start run directory is the fine-tuned checkpoint, **not** the selected checkpoint. The selected file is the separately rebaselined `warmstart_v3_catalog_baseline.pt`.

## Architecture and request flow

```text
Reels request (user, tab/context, as-of time)
        |
        +--> Model 1 user tower --> top-K video candidates
        |
        +--> static user-author affinity --> top-K creator candidates
                         |
                    deduplicate union
                         |
              build Model 2 input features
                         |
              ReelsRankerV2 inference
                         |
                sort by p(is_like)
                         |
              apply serving filters / return page
```

The last three serving steps are not yet implemented as one API service. In particular, `model2_ranking/recommend_v2.py` still contains the older hand-weighted formula. An API integration must explicitly apply `p_is_like` (or call a shared scorer using that formula); do not assume the CLI's default score already matches the selected offline formula.

### Model 1 — retrieval

- Two towers independently encode user/context and video/metadata features.
- Each tower outputs a normalized 128-dimensional vector.
- The selected V3 trainer uses 24-hour exposure pools, pool-softmax training, in-batch positives/negatives, log-Q correction, video-ID gating (`id_min_count=20`) and `id_dropout=0.2`.
- The selected `linear_output` option removes the final ReLU and dropout before vector normalization, allowing negative as well as positive coordinates.
- 60,183 video IDs receive dedicated embeddings; 27.4% of training positive events are on those kept IDs. Other IDs use metadata/OOV behavior.
- The recorded run used seed 42, batch size 512, 1,024 shared negatives, temperature 0.07, and 14 epochs on Apple MPS.

### Static author-affinity candidates

This is a separate non-neural retrieval source, not a third neural network. It uses positive user-author history and is not decay-weighted in the selected configuration. It complements Model 1 rather than replacing it.

### Model 2 — ranking

- `ReelsRankerV2` consumes 36 categorical and 10 numeric features.
- The network has a 512 → 256 → 128 shared representation, 10 binary behavior heads, and one watch-time regression head.
- The selected V3-catalogue checkpoint is the rebaselined original ranker. Fine-tuning changed validation AUC only marginally and was not promoted.
- The selected ranking formula is `p(is_like)`, not the old multi-objective hand-weighted sum. The score is a ranking signal, not a calibrated probability of overall user satisfaction.

See [Model 1 details](docs/MODEL1_RETRIEVAL.md), [Model 2 details](docs/MODEL2_RANKING.md), and [evaluation definitions](docs/EVALUATION.md).

## Measured results

Results below are from the recorded experiments. Do not mix the different evaluation protocols.

### Model 1: same 5,000-query validation pool evaluation

Both checkpoints were evaluated on the same 5,000 validation queries and the same 24-hour active exposure pools (66,267–255,916 videos per window).

| Metric | Original pool-softmax V3 | Linear-output V3 | Change |
|---|---:|---:|---:|
| Pool hit@1% | 14.10% | **15.02%** | +0.92 pp |
| Pool hit@5% | 37.50% | **38.86%** | +1.36 pp |
| Pool hit@20% | 70.90% | **72.36%** | +1.46 pp |
| Mean target percentile (lower is better) | 0.1639 | **0.1544** | -0.0095 |
| Recall@100 | 1.80% | **2.08%** | +0.28 pp |
| Recall@1,000 | 10.16% | **10.86%** | +0.70 pp |

The union of per-source top-K candidates also improved:

| Per-source K | Original union recovery | Linear-output union recovery |
|---|---:|---:|
| 1,000 | 19.64% | **20.04%** |
| 3,000 | 35.74% | **36.02%** |

### Model 2: validation score selection

On the 20,000-query validation run, the selected `is_like` formula had all-query Hit@100 **1.57%**; mean target percentile was **0.429** conditional on the target being retrieved (7,373 covered queries). `is_like` was selected for the top-100 objective. Adding follow/comment heads sometimes improved mean percentile, but did not show a reliable Hit@100 lift over `is_like` alone.

### Final held-out test

The final evaluation used 20,000 test queries and was run after the scoring decision was frozen.

| Metric | Test result |
|---|---:|
| Model 1 candidate recovery | 20.73% |
| Static author-affinity recovery | 21.24% |
| Hybrid union candidate recovery | **33.50%** |
| Queries where target was in hybrid slate | 6,700 / 20,000 |
| Users with covered queries | 868 |
| `is_like` mean target percentile | **0.42685** |
| `is_like` percentile 95% CI | [0.41863, 0.43512] |
| `is_like` Hit@100 (all queries) | **1.28%** |
| `is_like` Hit@100 95% CI | [1.118%, 1.464%] |

The `is_like` percentile is conditional on the target being present in the candidate slate. Hit@100 is reported over all evaluated queries; a target that was not retrieved cannot be ranked into the top 100. See [Evaluation](docs/EVALUATION.md) for the full definitions and caveats.

### Cold-start split

On test, 13,071 of 20,000 queries (65.36%) had a target video ID not seen in training. Candidate coverage was:

| Cohort | Model 1 | Author affinity | Hybrid union |
|---|---:|---:|---:|
| Cold-start video ID | 19.26% | **23.50%** | 33.40% |
| Seen video ID | **23.51%** | 16.97% | 33.68% |

This is why both candidate sources are retained. The cohort Hit@K/MRR values stored in the final JSON are produced by the evaluator's default composite ranker score, not the selected `is_like` score; do not label those cohort ranking metrics as `is_like` metrics.

## Data and evaluation caveats

- Canonical prepared dataset: `1k_v3_catalog`; 10,381,632 train rows, 665,706 validation rows, and 665,707 test rows.
- The V3 data has 1,001 users and 3,929,860 catalogue videos. It uses 36 categorical fields, 10 numeric fields, and 10 binary behavior targets.
- The split is time-ordered: older standard logs plus the first 80% of the newer log for training, then 10% validation, then 10% test.
- Aggregate video-statistic features and random-exposure logs are excluded. Static user features are used as supplied; their exact as-of timestamps are not available.
- The retrieval-pool and hybrid evaluators use 24-hour exposure windows. Their pool can contain exposures that occur later than an individual query within that same 24-hour window. This is optimistic versus strict point-in-time serving.
- This is offline KuaiRand evaluation, not a live Instagram/Reels A/B test. KuaiRand video IDs are not media URLs from the target app.
- The evaluation is single-target-per-query over a held-out positive. Production ranking must account for pagination, already-seen videos, availability, policy/safety filtering, freshness, deduplication, and repeated requests.

## Setup

The recorded development environment was Apple Silicon Mac mini M4 (16 GB unified memory), Python 3.12.16, PyTorch 2.15.0, and MPS. The scripts also select CUDA when available, then MPS, then CPU.

```bash
cd ~/Projects/reels-recommender-m4
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
PYTHONPATH="$PWD" python scripts/check_environment.py
PYTHONPATH="$PWD" pytest -q
```

The last user-side test run on 2026-11-02 passed **97 tests**. Re-run tests after any source change.

## Rebuild / evaluate selected Model 1 index

Only rebuild the index if its binary files are missing or the selected checkpoint changes. Use a new output directory for experiments; never overwrite the selected index casually.

```bash
cd ~/Projects/reels-recommender-m4
source .venv/bin/activate
python -m model1_retrieval.build_index \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --model-path artifacts/retrieval/v3_linear_output_seed42_14ep/best_model.pt \
  --output-dir data/indexes/retrieval/1k_v3_v3_linear_output_seed42_14ep \
  --batch-size 2048
```

The selected index contains 4,371,868 video vectors with 128 dimensions, stored as float16 (about 1.04 GiB for the embedding matrix).

## Important commands

```bash
cd ~/Projects/reels-recommender-m4
source .venv/bin/activate

python -m model1_retrieval.train_v3 --help
python -m model1_retrieval.build_index --help
PYTHONPATH="$PWD" python scripts/evaluate_retrieval_pools.py --help
PYTHONPATH="$PWD" python scripts/evaluate_hybrid_ranking.py --help
python -m model2_ranking.train_v2 --help
```

The scripts in `scripts/` are invoked with `PYTHONPATH="$PWD"` when run by file path, because they import the top-level `common` package.

## Project map

```text
common/                 Shared metrics, pools, feature fingerprints, score variants
model1_retrieval/       Two-tower model, losses, training, retrieval index and querying
model2_ranking/         Multi-task ranker, training and standalone recommendation CLI
scripts/                Data prep, evaluation, diagnostics and training orchestration
reports/experiments/    Retained decision evidence (JSON)
artifacts/              Model metadata/history and local checkpoints
data/                  Raw and prepared KuaiRand data plus retrieval index
 docs/                  Model, evaluation, training and project-history documents
 tests/                 Unit and integration tests
```

## Current next step

Integrate the frozen artifacts into the existing application API. The integration should load the models once, reuse the video index, build feature tensors with the exact V3 mappings, merge Model 1 and author-affinity candidates, score with `p_is_like`, and apply feed-serving constraints. The model-training phase is paused; online adaptation and scheduled retraining are intentionally deferred.

## Gather two-stage integration

This repository now contains the private serving contract in `serving/`, a Gather schema status manifest in `configs/gather_recommendation_schema_v1.json`, and a conservative impression-based preparation/readiness script in `scripts/prepare_gather_dataset.py`. Install only the separate API dependencies from `requirements-serving.txt` when you intend to run the service. It currently stays `COLD_START`; no Gather model or index is ready, and KuaiRand references must not serve Gather IDs. See `docs/GATHER_SERVING.md`, `docs/GATHER_TRAINING_READINESS.md`, and `docs/GATHER_MODEL_COMPATIBILITY.md`.
