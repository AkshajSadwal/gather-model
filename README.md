# Reels Recommendation Model — Apple Silicon / macOS

A PyTorch-based offline recommendation project for the KuaiRand-Pure and KuaiRand-1K datasets. The project contains two offline recommendation stages: **Model 1 (two-tower retrieval)** generates a broad candidate set, and **Model 2 (multi-task ranking)** reorders those candidates using richer features.

1. **Model 1 — Retrieval (implemented as an offline prototype):** separate user and video towers produce normalized 128-dimensional vectors.
2. **Model 2 — Ranking (implemented):** score candidate rows using user, video, metadata, and context features, then rank them.

The retrieval implementation is an offline KuaiRand prototype built against the repository’s V2 prepared-data schema. It needs a full run on the Mac with the actual prepared 1K arrays before its training quality or MPS memory behavior can be considered validated. The repository is not yet a production serving system: it still needs live event logging, online feature freshness, operational filtering, latency testing, monitoring, and online experimentation.

The training code automatically selects CUDA when available, then Apple Silicon MPS, then CPU.

## Project structure

Run commands from the project root: `reels-recommender-m4/`.

```text
reels-recommender-m4/
├── common/
│   ├── model_common.py              # Shared target definitions
│   └── data_version.py              # Prepared-data fingerprints
├── model1_retrieval/
│   ├── model.py                    # Two-tower 512 -> 256 -> 128 retrieval model
│   ├── data.py                     # Feature, positive and negative sampling
│   ├── train.py                    # Pairwise training and sampled Recall@K evaluation
│   ├── build_index.py              # Build float16 item-vector index
│   └── retrieve.py                 # Query top-K candidates for a user
├── model2_ranking/
│   ├── model_v2.py                  # ReelsRankerV2 architecture
│   ├── train_v2.py                  # Training, validation, test, checkpoints
│   ├── recommend_v2.py              # Score/rank videos for a user
│   └── model_summary.py              # Parameter and memory estimates
├── scripts/
│   ├── prepare_data_v2.py           # KuaiRand feature preparation
│   ├── download_datasets_macos.sh   # Download and verify datasets
│   ├── check_environment.py         # Python/PyTorch/MPS checks
│   ├── run_ranking_batch_sweep.py   # Three seeded ranking repeats at batch 4096
│   └── run_retrieval_repeats.py    # Three seeded retrieval repeats at batch 4096
├── configs/
│   ├── retrieval/                   # Reserved for retrieval configuration
│   └── ranking/                     # Reserved for ranking configuration
├── data/
│   ├── raw/                         # Original downloaded/extracted data
│   ├── processed/
│   │   ├── ranking/
│   │   │   ├── pure_v2/
│   │   │   └── 1k_v2/
│   │   └── retrieval/               # Reserved for retrieval data
│   └── indexes/                     # Reserved for retrieval indexes
├── artifacts/
│   ├── ranking/                     # Ranking checkpoints and experiment outputs
│   └── retrieval/                   # Reserved for retrieval artifacts
├── reports/
│   ├── experiments/
│   └── legacy/
├── docs/
│   ├── MODEL1_RETRIEVAL.md          # Retrieval objective, evaluation and limitations
│   └── archive/                     # Archived project notes
├── tests/
│   └── test_retrieval_core.py       # Retrieval model/negative-sampling unit tests
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/                            # Local Python virtual environment (not committed)
```

The `configs/` and `reports/experiments/` directories are reserved for future configuration and evaluation files. The current Model 1 retrieval index is a flat chunk-scanned baseline; ANN indexing and live serving are future production work.

## Current ranking model

`ReelsRankerV2` uses 36 categorical fields and 10 numeric fields for the prepared KuaiRand data. The categorical features use embeddings. The shared MLP has three fully connected hidden layers:

**512 → 256 → 128**

Each hidden layer uses ReLU and dropout (20%, 15%, and 10%, respectively). The final representation feeds 11 prediction outputs: 10 behaviour targets and one watch-time regression target. The project default training batch size is **4,096** on the 16 GB M4; it is a starting configuration, not a claim that it is globally optimal.

For the current KuaiRand-1K feature mappings and embedding dimension 32, the model has **184,836,669 parameters**. Float32 weights alone require approximately 0.69 GiB. The approximate FP32 AdamW model state (weights, gradients, and two optimizer states) is 2.75 GiB; this excludes dataset arrays, activations, temporary buffers, and macOS/MPS overhead.

Checkpoints from the previous 256 → 128 architecture are not shape-compatible with this 512 → 256 → 128 architecture. Train a new checkpoint for the current architecture.

## 1. Mac and environment setup

The working development environment used for this project is:

- Apple Silicon Mac mini M4, 16 GB unified memory
- macOS 27.0.1, arm64
- Python 3.12.16
- PyTorch 2.15.0
- MPS available

Check your machine:

```bash
uname -m
system_profiler SPHardwareDataType | grep -E 'Chip|Memory'
sw_vers
```

On Apple Silicon, `uname -m` should report `arm64`.

Install Xcode Command Line Tools if they are not already installed:

```bash
xcode-select --install
```

Install Python 3.12 if needed (for example, with Homebrew):

```bash
brew install python@3.12
```

Create and activate the project environment from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an existing setup, activate the existing environment instead of creating another one:

```bash
cd ~/Projects/reels-recommender-m4
source .venv/bin/activate
```

Check the environment:

```bash
python -m scripts.check_environment
```

On the M4 setup, the expected accelerator lines include:

```text
MPS built: True
MPS available: True
Selected device: Apple Silicon GPU (MPS)
```

Do not assume GPU acceleration is active just because PyTorch is installed; check the output. If MPS is unavailable, investigate the Python/PyTorch installation before a long training run.

## 2. Download KuaiRand datasets

The current preparation pipeline supports the official **KuaiRand-Pure** and **KuaiRand-1K** datasets. It does not support KuaiRand-27K.

From the project root:

```bash
chmod +x scripts/download_datasets_macos.sh
bash scripts/download_datasets_macos.sh all
```

To download only one dataset:

```bash
bash scripts/download_datasets_macos.sh pure
bash scripts/download_datasets_macos.sh 1k
```

The downloader stores the datasets under `data/raw/` and verifies the published MD5 checksums before extraction. Expected extracted directories:

```text
data/raw/KuaiRand-Pure/data/
data/raw/KuaiRand-1K/data/
```

Keep the downloaded archives until extraction has been confirmed. Raw archives, extracted CSVs, and prepared arrays can coexist, so ensure sufficient free storage before downloading and preparing both datasets.

## 3. Prepare datasets

Prepare Pure first as a smaller end-to-end check:

```bash
python -u -m scripts.prepare_data_v2 \
  --dataset pure \
  --data-dir data/raw/KuaiRand-Pure/data \
  --output-dir data/processed/ranking/pure_v2
```

Prepare KuaiRand-1K:

```bash
python -u -m scripts.prepare_data_v2 \
  --dataset 1k \
  --data-dir data/raw/KuaiRand-1K/data \
  --output-dir data/processed/ranking/1k_v2
```

Pure and 1K are prepared separately. **Do not combine their user/video IDs, feature mappings, or processed NPZ files.**

The preparation script uses the standard-feed logs and makes chronological train/validation/test splits. It combines the earlier standard-feed log with the first 80% of the later log for training, then uses the final chronological portions for validation and test. It deliberately excludes aggregate video-statistics files to reduce the risk of temporal leakage and does not use random-exposure logs.

## 4. Inspect the prepared model size

For the 1K dataset:

```bash
python -m model2_ranking.model_summary \
  --data-dir data/processed/ranking/1k_v2
```

For Pure:

```bash
python -m model2_ranking.model_summary \
  --data-dir data/processed/ranking/pure_v2
```

The summary reports the parameter count and an approximate model-state memory estimate. The estimate is not the full runtime memory requirement; it excludes data arrays, activations, temporary buffers, and operating-system overhead.

## 5. Run a small smoke test

Before a full run, verify MPS and train a small subset on Pure:

```bash
python -m scripts.check_environment

python -u -m model2_ranking.train_v2 \
  --data-dir data/processed/ranking/pure_v2 \
  --output-dir artifacts/ranking/pure_v2_smoke \
  --epochs 1 \
  --batch-size 4096 \
  --eval-batch-size 4096 \
  --embedding-dim 32 \
  --max-train-rows 100000 \
  --cpu-threads 4 \
  --cooldown-every-batches 100 \
  --cooldown-seconds 1
```

`--max-train-rows 100000` limits the training subset. Validation and test still use their complete prepared splits unless test evaluation is explicitly skipped. Confirm that the training output selects `mps` on the M4.

## 6. Train the ranking model

Use this quality-oriented starting configuration for KuaiRand-1K on the M4:

```bash
python -u -m model2_ranking.train_v2 \
  --data-dir data/processed/ranking/1k_v2 \
  --output-dir artifacts/ranking/1k_v2_m4_b4096 \
  --epochs 20 \
  --batch-size 4096 \
  --eval-batch-size 4096 \
  --embedding-dim 32 \
  --learning-rate 0.001 \
  --cpu-threads 4 \
  --cooldown-every-batches 0 \
  --early-stopping-patience 4
```

This run configuration is a tested starting point for the described M4 machine, not a guarantee that it is optimal for every Mac or dataset state. Monitor system memory pressure during training. Do not run multiple large MPS training jobs concurrently on a 16 GB unified-memory machine.

The uploaded historical artifact snapshot contains a two-epoch run for the older 16,384-batch configuration (best recorded validation mean AUC 0.7992; test mean AUC 0.7941). Other notes from earlier runs report different values. Because the repository snapshots contain inconsistent historical metrics, treat them as unverified until the original checkpoints and logs are reconciled. Use the fresh seeded 4,096-batch runs below as the controlled baseline.

The training scripts write a fingerprint of the complete prepared train/validation/test arrays in run metadata and checkpoints. Resume is rejected if those arrays or feature mappings have changed; a fresh output directory is required for a new prepared-data version. The ranking recommender checks the feature/ID-map fingerprint before scoring a dataset.

The training script supports:

- `best_model.pt`: best checkpoint according to validation performance; use for inference.
- `latest_checkpoint.pt`: resumable training state, including optimizer state.
- `training_history.json`: per-epoch training/validation history.
- `test_metrics.json`: held-out test metrics when test evaluation is run.

To resume an interrupted experiment, use the **same output directory**, the same dataset and model dimensions, and add `--resume`:

```bash
python -u -m model2_ranking.train_v2 \
  --data-dir data/processed/ranking/1k_v2 \
  --output-dir artifacts/ranking/1k_v2_m4_b4096 \
  --epochs 20 \
  --batch-size 4096 \
  --eval-batch-size 4096 \
  --embedding-dim 32 \
  --learning-rate 0.001 \
  --cpu-threads 4 \
  --cooldown-every-batches 0 \
  --early-stopping-patience 4 \
  --resume
```

`--epochs` is the maximum total epoch number, including epochs completed before resuming. Do not change batch size, embedding dimension, dataset mappings, or other model-defining settings when resuming a checkpoint unless the training script explicitly supports that change.

For experimentation, `--skip-test-eval` skips the held-out test evaluation. Use validation metrics for model tuning, and reserve test results for final evaluation rather than repeatedly tuning against the test set.

## 7. Generate recommendations with the ranking model

Use a raw user ID from the selected dataset's standard-feed logs and the best checkpoint for that same dataset:

```bash
python -u -m model2_ranking.recommend_v2 \
  --data-dir data/processed/ranking/1k_v2 \
  --model-path artifacts/ranking/1k_v2_m4_b4096/best_model.pt \
  --user-id USER_ID \
  --top-k 20 \
  --batch-size 4096 \
  --output artifacts/ranking/1k_v2_m4_b4096/sample_recommendations.csv
```

Replace `USER_ID` with an actual raw user ID from the 1K dataset. For Pure, use the Pure processed directory and a Pure checkpoint instead. KuaiRand user/video IDs are dataset identifiers; they are not IDs or video URLs from a separate app.

The ranker scores candidate rows using the ranking feature schema. Its current final utility score is a manually weighted combination of predicted action probabilities; those weights are a baseline heuristic, not a learned or online-calibrated business objective. The output also includes predicted watch time for analysis, but the default score does not yet include it. Calibrate the utility function with reliable logged outcomes and then validate it online. A production feed still needs live user/item feature serving, authoritative seen-video and policy filters, index refresh, feedback/impression logging, monitoring, latency testing, and online A/B experiments.

## 8. Model 1 — two-tower retrieval

Model 1 is an offline two-tower candidate retriever. Its user and video towers each use three descending hidden layers:

**512 → 256 → 128**

The user tower uses user ID, feed tab, static user profile categories, time-of-day context, and user numeric features. The video tower uses video ID, author/video/music/tag metadata, duration, width, and height. The retrieval item tower deliberately excludes video age because age changes over time and would make a cached item vector stale; the ranker retains per-impression video age. The two towers produce L2-normalized 128-dimensional vectors. Video ID uses a 16-dimensional embedding because the KuaiRand-1K catalogue contains nearly four million videos; the tower output remains 128-dimensional.

The retrieval training objective uses pairwise ranking against sampled catalogue items. Positive events are weighted with stronger emphasis on long views, complete views, follows, forwards, likes, and comments. Clicks contribute a weaker positive signal. A `short_view` label subtracts 0.5 from that row's positive strength; a click-only short skip is not selected as a positive, while stronger engagement can still outweigh the short-view penalty. Explicit `is_hate` and `short_view` examples can be sampled as hard negatives. `short_view` is a weak negative signal (weight 0.5); stronger actions such as a like or follow can outweigh it on the same row. Random unobserved catalogue items are *sampled implicit negatives*, not confirmed dislikes. This limitation must be considered when interpreting results.

The default retrieval batch size is **4,096 positive interactions**, with two sampled negative videos per positive. This means the video tower processes 12,288 items in each full training step. The current KuaiRand-1K model has **92,732,511 parameters** (about 0.35 GiB of FP32 weights, or roughly 1.38 GiB for weights, gradients, and two AdamW state tensors). Both retrieval and ranking default to batch size 4,096 on this M4, but their memory requirements are not identical.

First run a one-epoch smoke test using 100,000 sampled positive rows. This checks data loading, feature alignment, negative sampling, training, and validation without committing to a full run:

```bash
python -u -m model1_retrieval.train \
  --data-dir data/processed/ranking/1k_v2 \
  --output-dir artifacts/retrieval/1k_v2_smoke \
  --epochs 1 \
  --batch-size 4096 \
  --max-positive-rows 100000 \
  --eval-queries 128 \
  --eval-negatives 200 \
  --seed 42 \
  --cpu-threads 4
```

For the full repeated experiment, run three sequential trials at the agreed batch size:

```bash
python -u -m scripts.run_retrieval_repeats \
  --data-dir data/processed/ranking/1k_v2 \
  --output-root artifacts/retrieval/repeated_runs \
  --batch-size 4096 \
  --seeds 17 42 2026 \
  --eval-seed 314159 \
  --epochs 20 \
  --negatives-per-positive 2 \
  --eval-queries 512 \
  --eval-negatives 500 \
  --early-stopping-patience 4 \
  --cpu-threads 4
```

Each seed receives its own checkpoint folder and log. The runner uses one fixed evaluation seed (`314159`) so all training seeds share identical validation queries and sampled negatives. It writes `trial_results.csv` and `repeat_summary.csv`. Compare mean and standard deviation of sampled Recall@100, sampled Recall@20, MRR and runtime. This is a three-seed stability check at batch 4,096, not a batch-size comparison. Do not run the retrieval and ranking repeat runners simultaneously on the 16 GB Mac.

The trainer selects the best checkpoint using **sampled Recall@100** on a fixed validation query/candidate sample. It reports sampled Recall@20, Recall@100, MRR, and median sampled rank. These are not full-catalogue recall values. Tuning should use validation only; `--evaluate-test` runs a sampled test evaluation after training and should be reserved for final evaluation.

Build a local flat video-vector index after training. Retrieval currently scans the float16 vectors in chunks; this is an exact offline baseline, not an approximate-nearest-neighbour (ANN) service. A production catalogue at this scale should move to an ANN index (for example, FAISS or another evaluated vector index) after offline quality and recall are established:

```bash
python -m model1_retrieval.build_index \
  --data-dir data/processed/ranking/1k_v2 \
  --model-path artifacts/retrieval/repeated_runs/YOUR_RUN_GROUP/seed_YOUR_BEST_SEED/best_model.pt \
  --output-dir data/indexes/retrieval/1k_v2 \
  --batch-size 4096
```

Replace `YOUR_RUN_GROUP` with the actual timestamped run-group directory and `YOUR_BEST_SEED` with the seed that has the strongest validation sampled Recall@100 in `trial_results.csv` (or the per-seed metrics). Do not assume seed 42 is best. If the index already exists, the builder stops rather than silently overwriting it; pass `--overwrite` to preserve the old index as a timestamped backup and publish a replacement.

The index stores one float16, 128-dimensional vector per training-catalogue video. For 3.93 million videos this is about 1.0 GB for the vector matrix, plus IDs and metadata. It is generated locally and should not be committed to Git.

Retrieve candidates for a raw dataset user ID:

```bash
python -m model1_retrieval.retrieve \
  --data-dir data/processed/ranking/1k_v2 \
  --model-path artifacts/retrieval/repeated_runs/YOUR_RUN_GROUP/seed_YOUR_BEST_SEED/best_model.pt \
  --index-dir data/indexes/retrieval/1k_v2 \
  --user-id USER_ID \
  --tab 0 \
  --top-k 500 \
  --output artifacts/retrieval/1k_v2_m4_b4096/candidates.csv
```

Pass Model 1's candidate CSV into Model 2 to complete the offline retrieval-to-ranking path:

```bash
python -m model2_ranking.recommend_v2 \
  --data-dir data/processed/ranking/1k_v2 \
  --model-path artifacts/ranking/1k_v2_m4_b4096/best_model.pt \
  --user-id USER_ID \
  --candidate-file artifacts/retrieval/1k_v2_m4_b4096/candidates.csv \
  --top-k 20 \
  --batch-size 4096 \
  --output artifacts/ranking/1k_v2_m4_b4096/ranked_recommendations.csv
```

`recommend_v2` can still score the whole prepared catalogue when `--candidate-file` is omitted, but the two-stage workflow should pass the Model 1 candidate file.

An optional `--exclude-video-ids path.txt` accepts one raw video ID per line to filter already-seen or unavailable videos. The current offline example does not connect to a live user-history store. The `retrieve.py` search is exact chunked dot-product scanning over the full catalogue; no ANN data structure is built yet.

## 9. Reproducible ranking trials at the agreed batch size

The ranking trainer accepts `--seed` and `--run-name`. Run three sequential repeats at batch size 4,096:

```bash
python -u -m scripts.run_ranking_batch_sweep \
  --data-dir data/processed/ranking/1k_v2 \
  --output-root artifacts/ranking/repeated_runs \
  --batch-sizes 4096 \
  --seeds 17 42 2026 \
  --epochs 20 \
  --eval-batch-size 4096 \
  --embedding-dim 32 \
  --learning-rate 0.001 \
  --cpu-threads 4 \
  --early-stopping-patience 4
```

This is now a repeatability/stability check at one selected batch size, not a batch-size comparison. The reports include per-run results and mean/standard deviation across the seeds. Held-out test evaluation is skipped during these trials.

## Useful maintenance commands

Compile the reorganized Python modules:

```bash
python -m py_compile model1_retrieval/*.py model2_ranking/*.py scripts/*.py
```

Check the environment:

```bash
python -m scripts.check_environment
```

Run the lightweight unit tests:

```bash
python -m unittest discover -s tests
```

Show retrieval repeat-runner options without starting training:

```bash
python -m scripts.run_retrieval_repeats --help
```

Show retrieval training options without starting training:

```bash
python -m model1_retrieval.train --help
```

Show ranking training options without starting training:

```bash
python -m model2_ranking.train_v2 --help
```

Show data-preparation options without modifying data:

```bash
python -m scripts.prepare_data_v2 --help
```


## Run both training stages with one command

See [`docs/TWO_STAGE_TRAINING.md`](docs/TWO_STAGE_TRAINING.md). The orchestrator runs ranking and retrieval **sequentially**, not concurrently, to keep unified-memory usage safer on a 16 GB M4. The default is batch size 4,096, maximum 20 epochs, seed 42, and early-stopping patience 4.

```bash
python -u -m scripts.run_two_stage_training \
  --data-dir data/processed/ranking/1k_v2 \
  --output-root artifacts/two_stage_runs \
  --epochs 20 \
  --batch-size 4096 \
  --eval-batch-size 4096 \
  --seed 42 \
  --eval-seed 314159 \
  --early-stopping-patience 4 \
  --cpu-threads 4
```
