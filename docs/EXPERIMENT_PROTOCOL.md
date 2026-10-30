# Ranking experiment protocol — fixed 4,096 batch size

The agreed ranking batch size for the current Apple Silicon M4 development machine is **4,096**. The purpose of the immediate repeatability experiment is to measure variability across three seeds, not compare different batch sizes.

## Trial design

- Batch size: 4,096 for every run
- Seeds: 17, 42, 2026
- Runs: 3 total, sequential (never concurrent on a 16 GB unified-memory Mac)
- Architecture: 512 -> 256 -> 128, embedding dimension 32
- Learning rate: 0.001
- Maximum epochs: 20; early-stopping patience: 4; minimum delta: 0.0001
- Evaluation batch: 4,096
- Test evaluation: skipped during tuning

PyTorch MPS operations are not guaranteed to be bitwise deterministic. Fixed seeds improve comparability but do not promise byte-identical outcomes across devices or software versions.

## Selection and interpretation

1. Review mean validation mean AUC and per-target validation AUC for all three seeds.
2. Review standard deviation across seeds; do not rely on the single highest run.
3. Compare best epoch and elapsed time to detect unstable or prematurely stopped runs.
4. If the variation is large, investigate sampling, target imbalance, and training stability before changing architecture.
5. Evaluate the selected checkpoint on the held-out test split only after the validation review. Avoid repeated test-set tuning.

Batch size affects optimizer updates per epoch as well as throughput. For the 10.38 million-row KuaiRand-1K training split, batch 4,096 gives approximately 2,535 training updates per epoch.

## Limitations

- The fixed seed does not make all MPS kernels deterministic.
- Three seeds provide a useful initial stability check, not a definitive statistical guarantee.
- Offline AUC does not prove online user satisfaction or causal lift.
- A production evaluation requires trustworthy impression logging, temporal validation, candidate retrieval metrics, ranking metrics, latency and online A/B testing.


## Dataset identity

Each trial records a SHA-256 fingerprint covering `metadata.json`, `id_maps.npz`, `train.npz`, `validation.npz`, and `test.npz`. A checkpoint cannot be resumed on a different prepared data version. The feature-schema fingerprint is also stored in the ranking checkpoint and checked by the offline recommender before scoring a dataset.
