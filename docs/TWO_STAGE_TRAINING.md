# Run both model training jobs on the M4

The project uses a 16 GB unified-memory Apple Silicon Mac. **Do not run the
ranking and retrieval trainers as concurrent MPS processes.** The separate
smoke tests reported about 5.02 GiB MPS driver allocation for ranking and
3.03 GiB for retrieval after validation. Those are not peak measurements and
must not be added as exact forecasts, but they show that concurrent runs could
put substantial pressure on unified memory. Each process also loads its own
prepared training arrays and maintains its own model and optimizer state.

Use the orchestrator below to start both jobs with one command. It runs Model 2
ranking first, then Model 1 retrieval, releasing the first process before
starting the second. Both jobs use batch size 4,096 and up to 20 epochs. The
default early-stopping patience is four, so a run may correctly finish before
epoch 20 if validation stops improving.

```bash
python -u -m scripts.run_two_stage_training \\
  --data-dir data/processed/ranking/1k_v2 \\
  --output-root artifacts/two_stage_runs \\
  --epochs 20 \\
  --batch-size 4096 \\
  --eval-batch-size 4096 \\
  --seed 42 \\
  --eval-seed 314159 \\
  --early-stopping-patience 4 \\
  --cpu-threads 4
```

Each execution creates a new timestamped run directory containing:
- `ranking/`: Model 2 best checkpoint, resumable checkpoint, history and metadata.
- `retrieval/`: Model 1 best checkpoint, resumable checkpoint, history and metadata.
- `logs/`: separate console logs for each stage.
- `pipeline_metadata.json`: data fingerprints, settings, stage status and elapsed time.

The pipeline skips held-out test evaluation. Use validation metrics for model
selection, then evaluate the selected checkpoints on the test split once the
training run has been reviewed. The 20-epoch setting is a maximum, not a
requirement to continue after early stopping.

The current pipeline runs one seed per model (default 42). For a repeatability
study, use the dedicated ranking and retrieval repeat runners, which run seeds
17, 42 and 2026 sequentially. Do not run those runners concurrently either.

This is still an offline research pipeline, not a production recommendation
service. The retrieval index is currently exact chunked search; ANN indexing,
impression logging, exposure-aware evaluation, serving, monitoring and online
A/B tests remain future work.
