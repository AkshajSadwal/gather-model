# Reels Recommendation Model — V2 (Apple Silicon / macOS)

This is the V2 feature-aware, multi-task Reels ranker. It is configured for Apple Silicon Macs (M-series) using PyTorch's Metal Performance Shaders (MPS) backend. CUDA is still selected automatically on a supported NVIDIA system; otherwise it selects MPS, then CPU.

## Included files

- `model_common.py`: shared target names used by V2.
- `model_v2.py`: embedding-based, three-hidden-layer ranking model (512 → 256 → 128).
- `model_summary.py`: reports exact parameter counts and approximate model-state memory for a prepared dataset.
- `prepare_data_v2.py`: chronological dataset preparation for KuaiRand-Pure and KuaiRand-1K.
- `train_v2.py`: V2 training, validation, test, early stopping, and resumable checkpoints.
- `recommend_v2.py`: candidate scoring using a saved V2 model.
- `download_datasets_macos.sh`: downloads and verifies the official Pure and 1K archives.
- `check_environment.py`: checks Python packages and accelerator availability.

The old V1 training pipeline is intentionally omitted from this clean V2 project. The target definitions are in `model_common.py` so V2 no longer imports the V1 model file.

## Current model architecture

The shared MLP has three fully connected hidden layers: **512 → 256 → 128**, each followed by ReLU and dropout (20%, 15%, and 10% respectively). The final 128-dimensional representation feeds two heads: 10 behaviour predictions and one watch-time prediction. Categorical embeddings and numeric input features are unchanged. The model is initialized from scratch for this architecture; checkpoints from the previous 256 → 128 architecture are not shape-compatible.

## 1. Check your Mac

Open Terminal and run:

```bash
uname -m
system_profiler SPHardwareDataType | grep -E 'Chip|Memory'
sw_vers
```

`uname -m` should say `arm64`. Note your unified-memory size. Keep at least 15 GB of free storage before downloading and preparing both datasets; more is helpful because raw archives, extracted CSVs, and processed arrays coexist.

## 2. Install prerequisites

Apple's current PyTorch MPS instructions require Apple silicon, macOS 14 or later, Python 3.10 or later, and Xcode Command Line Tools. Install the tools if they are missing:

```bash
xcode-select --install
```

Install Python 3.12 with Homebrew (install Homebrew first if `brew` is not found):

```bash
brew install python@3.12
```

From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_environment.py
```

On an M-series Mac, the environment check should report `MPS available: True` and `Selected device: Apple Silicon GPU (MPS)`. Do not start full training if it reports CPU instead; inspect the Python/PyTorch install first.

## 3. Download datasets

The project currently supports the official KuaiRand-Pure and KuaiRand-1K datasets. It does not use KuaiRand-27K, so the 27K archive is not downloaded by this project script.

From the project root:

```bash
chmod +x download_datasets_macos.sh
./download_datasets_macos.sh all
```

The script downloads the official Zenodo archives, verifies their published MD5 checksums, and extracts them under `data/raw/`. It can be rerun safely; already extracted datasets are skipped. To download only one dataset, use `./download_datasets_macos.sh pure` or `./download_datasets_macos.sh 1k`.

Expected folders:

```text
data/raw/KuaiRand-Pure/data/
data/raw/KuaiRand-1K/data/
```

Keep the archives until extraction is confirmed. Pure is about 47 MB compressed; 1K is about 1.1 GB compressed and roughly 4.3 GB extracted. The 27K dataset is much larger and is not supported by the current preparation script.

## 4. Prepare the datasets

Run Pure first as a small end-to-end check:

```bash
python -u prepare_data_v2.py \
  --dataset pure \
  --data-dir data/raw/KuaiRand-Pure/data \
  --output-dir data/processed/pure_v2
```

Then prepare 1K (this is a much larger pandas workload; close memory-heavy apps first):

```bash
python -u prepare_data_v2.py \
  --dataset 1k \
  --data-dir data/raw/KuaiRand-1K/data \
  --output-dir data/processed/1k_v2
```

Pure and 1K are prepared separately. Do not combine their user/video IDs or their NPZ files. The script uses the two standard-feed logs, assigns the earlier log plus the first 80% of the later log to training, and the final chronological 10%/10% to validation/test. It intentionally excludes aggregate video statistics and random-exposure logs.

## Model parameter summary

After preparing a dataset, print the exact parameter count for the selected architecture:

```bash
python model_summary.py --data-dir data/processed/1k_v2
```

For the uploaded KuaiRand-1K metadata and 32-dimensional user/video embeddings, the 512 → 256 → 128 model has **184,836,669 parameters**. The model-state estimate assumes float32 parameters, gradients, and two AdamW state tensors; it excludes the prepared dataset arrays, activation memory, temporary buffers, and macOS/MPS overhead.

## 5. Verify MPS, then run a small smoke test

```bash
python check_environment.py
python -u train_v2.py \
  --data-dir data/processed/pure_v2 \
  --output-dir artifacts/pure_v2_smoke \
  --epochs 1 \
  --batch-size 256 \
  --embedding-dim 32 \
  --max-train-rows 100000 \
  --cpu-threads 4 \
  --cooldown-every-batches 100 \
  --cooldown-seconds 1
```

This limits the training subset to 100,000 rows, but validation and test still use their complete prepared splits. Confirm the output says `Device: mps` before moving on.

## 6. Train from scratch

For the larger KuaiRand-1K experiment, start conservatively on the Mac mini. Use batch size 256 initially; if memory pressure remains low and the smoke test succeeds, 512 can be tested later.

```bash
python -u train_v2.py \
  --data-dir data/processed/1k_v2 \
  --output-dir artifacts/1k_v2_m4_512_256_128 \
  --epochs 10 \
  --batch-size 256 \
  --embedding-dim 32 \
  --cpu-threads 4 \
  --cooldown-every-batches 100 \
  --cooldown-seconds 1 \
  --early-stopping-patience 3
```

For Pure full training instead, use `data/processed/pure_v2` and `artifacts/pure_v2_m4` as the two directory values.

The script automatically selects CUDA, then MPS, then CPU. It saves:

- `best_model.pt`: best validation model for inference.
- `latest_checkpoint.pt`: model + optimizer + epoch state, plus available CPU/CUDA/MPS RNG state, for resuming.
- `training_history.json`: per-epoch metrics.
- `test_metrics.json`: held-out test metrics after training ends.

To resume an interrupted run, use the **same output directory** and add `--resume` while keeping the same data and model dimensions:

```bash
python -u train_v2.py \
  --data-dir data/processed/1k_v2 \
  --output-dir artifacts/1k_v2_m4_512_256_128 \
  --epochs 10 \
  --batch-size 256 \
  --embedding-dim 32 \
  --cpu-threads 4 \
  --cooldown-every-batches 100 \
  --cooldown-seconds 1 \
  --early-stopping-patience 3 \
  --resume
```

## 7. Generate recommendations

Replace `USER_ID` with a raw user ID from the selected dataset's standard-feed log:

```bash
python -u recommend_v2.py \
  --data-dir data/processed/1k_v2 \
  --model-path artifacts/1k_v2_m4_512_256_128/best_model.pt \
  --user-id USER_ID \
  --top-k 20 \
  --batch-size 1024 \
  --output artifacts/1k_v2_m4/sample_recommendations.csv
```

The IDs are KuaiRand dataset IDs, not IDs or video URLs from the user's app. This remains an offline candidate-ranking model, not a production retrieval/serving system.
# feed-model-mac
