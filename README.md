# Reels Recommendation Model — Apple Silicon / macOS

A PyTorch-based offline recommendation project for the KuaiRand-Pure and KuaiRand-1K datasets. The current implemented model is **Model 2: a feature-aware, multi-task ranking model**. The project is being organized toward a two-stage recommendation pipeline:

1. **Model 1 — Retrieval (planned):** retrieve a manageable set of candidate videos from the catalogue.
2. **Model 2 — Ranking (implemented):** score candidates using user, video, metadata, and context features, then rank them.

The retrieval stage is not installed or implemented in the current working project yet. Do not treat this repository as a complete production retrieval/serving system.

The training code automatically selects CUDA when available, then Apple Silicon MPS, then CPU.

## Project structure

Run commands from the project root: `reels-recommender-m4/`.

```text
reels-recommender-m4/
├── common/
│   └── model_common.py              # Shared target definitions
├── model1_retrieval/                # Reserved for the planned retrieval model
├── model2_ranking/
│   ├── model_v2.py                  # ReelsRankerV2 architecture
│   ├── train_v2.py                  # Training, validation, test, checkpoints
│   ├── recommend_v2.py              # Score/rank videos for a user
│   └── model_summary.py              # Parameter and memory estimates
├── scripts/
│   ├── prepare_data_v2.py           # KuaiRand feature preparation
│   ├── download_datasets_macos.sh   # Download and verify datasets
│   └── check_environment.py         # Python/PyTorch/MPS checks
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
│   └── archive/                     # Archived project notes
├── tests/
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/                            # Local Python virtual environment (not committed)
```

Some reserved directories may be empty until the retrieval stage is added.

## Current ranking model

`ReelsRankerV2` uses 36 categorical fields and 10 numeric fields for the prepared KuaiRand data. The categorical features use embeddings. The shared MLP has three fully connected hidden layers:

**512 → 256 → 128**

Each hidden layer uses ReLU and dropout (20%, 15%, and 10%, respectively). The final representation feeds 11 prediction outputs: 10 behaviour targets and one watch-time regression target.

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
  --batch-size 256 \
  --eval-batch-size 1024 \
  --embedding-dim 32 \
  --max-train-rows 100000 \
  --cpu-threads 4 \
  --cooldown-every-batches 100 \
  --cooldown-seconds 1
```

`--max-train-rows 100000` limits the training subset. Validation and test still use their complete prepared splits unless test evaluation is explicitly skipped. Confirm that the training output selects `mps` on the M4.

## 6. Train the ranking model

A quality-oriented KuaiRand-1K command used successfully on the M4 is:

```bash
python -u -m model2_ranking.train_v2 \
  --data-dir data/processed/ranking/1k_v2 \
  --output-dir artifacts/ranking/1k_v2_m4_b16384_quality20 \
  --epochs 20 \
  --batch-size 16384 \
  --eval-batch-size 4096 \
  --embedding-dim 32 \
  --learning-rate 0.001 \
  --cpu-threads 4 \
  --cooldown-every-batches 0 \
  --early-stopping-patience 4
```

This run configuration is a tested starting point for the described M4 machine, not a guarantee that it is optimal for every Mac or dataset state. Monitor system memory pressure during training. Do not run multiple large MPS training jobs concurrently on a 16 GB unified-memory machine.

The latest recorded run with this configuration stopped early after validation stopped improving. Its reported best validation mean AUC was **0.8210**, and its held-out test mean AUC was **0.8107** (test loss 0.5507). These are results from this project's experiment and split; they should not be compared directly with scores from unrelated datasets, splits, or tasks.

The training script supports:

- `best_model.pt`: best checkpoint according to validation performance; use for inference.
- `latest_checkpoint.pt`: resumable training state, including optimizer state.
- `training_history.json`: per-epoch training/validation history.
- `test_metrics.json`: held-out test metrics when test evaluation is run.

To resume an interrupted experiment, use the **same output directory**, the same dataset and model dimensions, and add `--resume`:

```bash
python -u -m model2_ranking.train_v2 \
  --data-dir data/processed/ranking/1k_v2 \
  --output-dir artifacts/ranking/1k_v2_m4_b16384_quality20 \
  --epochs 20 \
  --batch-size 16384 \
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
  --model-path artifacts/ranking/1k_v2_m4_b16384_quality20/best_model.pt \
  --user-id USER_ID \
  --top-k 20 \
  --batch-size 1024 \
  --output artifacts/ranking/1k_v2_m4_b16384_quality20/sample_recommendations.csv
```

Replace `USER_ID` with an actual raw user ID from the 1K dataset. For Pure, use the Pure processed directory and a Pure checkpoint instead. KuaiRand user/video IDs are dataset identifiers; they are not IDs or video URLs from a separate app.

The current ranker scores the candidate rows supplied by its recommendation workflow. A large-scale production feed still needs a retrieval stage, catalogue/index management, online feature serving, filtering, and serving infrastructure.

## 8. Two-stage roadmap

The planned recommendation flow is:

```text
User + context
      │
      ▼
Model 1: Retrieval (planned)
      │  Retrieve hundreds/thousands of candidate videos
      ▼
Candidate set
      │
      ▼
Model 2: Ranking (implemented)
      │  Predict behaviour scores and watch time
      ▼
Ranked recommendations
```

Keep the two stages separate:

- `model1_retrieval/` will contain the user/video retrieval model, its training code, and index-building/query utilities once implemented and tested.
- `model2_ranking/` contains the existing feature-aware multi-task ranker.

Do not move the ranking model or its checkpoints into the retrieval stage. Retrieval and ranking are different tasks and will have separate datasets/artifacts when the retrieval implementation is added.

## Useful maintenance commands

Compile the reorganized Python modules:

```bash
python -m py_compile model2_ranking/*.py scripts/*.py
```

Check the environment:

```bash
python -m scripts.check_environment
```

Show training options without starting training:

```bash
python -m model2_ranking.train_v2 --help
```

Show data-preparation options without modifying data:

```bash
python -m scripts.prepare_data_v2 --help
```
