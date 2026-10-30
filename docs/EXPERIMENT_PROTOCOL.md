# Ranking batch-size experiment protocol

This protocol compares ranking batch sizes on the same prepared KuaiRand-1K data.
It is intended for offline tuning on one Apple Silicon Mac; it is not a production
benchmark by itself.

## Planned comparison

- Batch sizes: 4,096; 8,192; 16,384
- Seeds: 17; 42; 2026
- Runs: 9 total, sequential (never concurrent on a 16 GB unified-memory Mac)
- Architecture: 512 -> 256 -> 128, embedding dimension 32
- Learning rate: 0.001
- Maximum epochs: 20; early-stopping patience: 4; minimum delta: 0.0001
- Evaluation batch: 4,096
- Test evaluation: skipped during selection

Use the same three seeds for each batch size. This pairs initializations and data
shuffle streams as far as the backend permits. PyTorch MPS operations are not
guaranteed to be bitwise deterministic, so interpret this as seeded repeatability,
not a promise of byte-identical outcomes.

## Selection rule

1. Compare mean best validation mean AUC across the three seeds.
2. Compare standard deviation across seeds; do not select on the single highest run.
3. Review each behaviour target's validation AUC, best epoch, and elapsed time.
4. Prefer a materially better and stable validation score. If results are within
   noise, prefer the faster batch size that leaves comfortable memory headroom.
5. After selecting the configuration, train/confirm the final candidate and evaluate
   the held-out test split once. Do not tune against the test results.

The sweep writes one directory per run, a `run_metadata.json` file, per-run logs,
`trial_results.csv/json`, and `batch_comparison.csv/json`. The trainer refuses to
silently reuse an output directory containing checkpoints or result files. Give
new experiments a new sweep name; use `--resume` only to continue an interrupted
run with its original seed and settings.

## Limitations

- A fixed seed does not make every MPS kernel bitwise deterministic.
- Batch size changes optimizer update count per epoch as well as throughput. Keep
  all other settings fixed during this first comparison; interpret the results as
  a practical batch-size comparison under the same epoch-based training budget.
- Offline validation AUC does not prove online user satisfaction or causal lift.
  Later production evaluation needs impression logging, temporal validation,
  retrieval metrics, ranking metrics, latency and online A/B testing.
