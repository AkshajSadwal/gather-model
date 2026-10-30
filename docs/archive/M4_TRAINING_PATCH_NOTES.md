# M4 training patch notes

This patch updates only `train_v2.py`.

## Changes

1. `--eval-batch-size` (default 4096) separates validation/test evaluation memory from the training batch size. This matters when benchmarking very large training batches.
2. `--skip-test-eval` lets you tune batch sizes and architecture using validation AUC without repeatedly inspecting the held-out test split.
3. MPS memory reports show current allocator use, driver allocation, and recommended MPS working-set limit when PyTorch exposes those APIs.
4. Ctrl+C during an epoch no longer writes partially updated weights as though they were a completed epoch. The last fully completed `latest_checkpoint.pt` is preserved; resuming replays the interrupted epoch cleanly.
5. When resuming, the `--learning-rate` value supplied on the command line is now applied after loading the optimizer state. Previously, loading the optimizer state silently restored the previous LR.

## Suggested full-data 32,768 batch benchmark

```bash
python -u train_v2.py \
  --data-dir data/processed/1k_v2 \
  --output-dir artifacts/1k_v2_m4_b32768_full \
  --epochs 1 \
  --batch-size 32768 \
  --eval-batch-size 4096 \
  --embedding-dim 32 \
  --learning-rate 0.001 \
  --cpu-threads 4 \
  --cooldown-every-batches 0 \
  --skip-test-eval
```

This is a throughput/memory benchmark for one complete pass over the 10.38M training rows. One epoch at 32,768 gives about 317 optimizer steps, so its AUC should not be treated as a converged model result. Use validation metrics to compare configurations; reserve the test split for final model selection/evaluation.

Do not use this 32,768 run's output directory to resume from a different batch size or architecture unless that is deliberate. The command uses a new output directory and starts from scratch.
