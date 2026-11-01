# Model 1 — two-tower retrieval

## Selected model

- Checkpoint: `artifacts/retrieval/v3_linear_output_seed42_14ep/best_model.pt`
- Index: `data/indexes/retrieval/1k_v3_v3_linear_output_seed42_14ep/`
- Dataset: `data/processed/ranking/1k_v3_catalog/`
- Architecture: independent user and video towers, each 512 → 256 → 128, with L2-normalized 128-dimensional output.
- Output layer: `linear_output=true`; the last ReLU and dropout are omitted before L2 normalization.
- Device used: Apple MPS.

The earlier pool-softmax V3 checkpoint, `artifacts/retrieval/v3_full_seed42_12ep/best_model.pt`, is retained only as a comparison/rollback artifact. Its metadata says 14 epochs even though the directory name says `12ep`.

## Training configuration

The selected run used seed 42, 14 epochs, batch size 512, 1,024 shared negatives, 24-hour pools, temperature 0.07, log-Q strength 1.0, ID minimum count 20, ID dropout 0.2, learning rate 0.001, minimum learning rate 0.00005, and weight decay 0.00001. Validation used 5,000 queries in three 24-hour windows, with a 50,000-video pool cap for checkpoint selection. Best epoch: 14. Best training-validation pool hit@1%: 0.1480.

The model records 92,732,511 parameters in the run log. It retains a dedicated video-ID embedding for 60,183 video IDs; 27.4% of training positive events occur on those IDs. Other videos are represented using available metadata and OOV ID handling.

## Same-protocol validation comparison

The indexed evaluation used the same 5,000 queries, seed 314159, 24-hour exposure windows and un-capped active pools (66,267–255,916 videos per window).

| Metric | Original V3 | Linear output |
|---|---:|---:|
| Hit@1% | 0.1410 | **0.1502** |
| Hit@5% | 0.3750 | **0.3886** |
| Hit@20% | 0.7090 | **0.7236** |
| Mean percentile | 0.16391 | **0.15443** |
| Recall@100 | 0.0180 | **0.0208** |
| Recall@1,000 | 0.1016 | **0.1086** |

The new checkpoint improved every displayed metric. These are active-pool metrics and must not be compared directly with full-catalogue Recall@K from older evaluation scripts.

## Hybrid candidate union

With a per-source top-K of 3,000, original V3 + static author affinity recovered 35.74% of validation targets; linear output + static author affinity recovered 36.02%. At K=1,000, union recovery was 19.64% versus 20.04%. The union is deduplicated and can contain up to 2K candidates.

## Build the selected video index

```bash
python -m model1_retrieval.build_index \
  --data-dir data/processed/ranking/1k_v3_catalog \
  --model-path artifacts/retrieval/v3_linear_output_seed42_14ep/best_model.pt \
  --output-dir data/indexes/retrieval/1k_v3_v3_linear_output_seed42_14ep \
  --batch-size 2048
```

The resulting index contains 4,371,868 vectors × 128 dimensions and uses approximately 1.04 GiB for float16 embeddings. The builder also writes an ID array and metadata; the full binary index is required for retrieval.

## Known limitations

- The current user tower uses user ID, static profile features and request context; it does not encode an ordered recent-watch session sequence.
- Static user feature timestamps are not available, so exact as-of freshness of those values cannot be verified.
- The 24-hour pool evaluation has within-window lookahead and is not strict point-in-time serving.
- The current flat chunk-scanned vector index is an offline baseline, not a production ANN service.
- Candidate coverage is still about 33.5% for the final hybrid on the held-out test set; most held-out targets are not in the candidate slate and cannot be fixed by the ranker.

## Experiments not promoted

The seed-43 in-batch and hybrid-loss trials, plus the one-epoch hybrid smoke trial, were not promoted. A hybrid sampled-negative experiment reported full-catalogue Recall@1,000 of 0.0096 versus 0.0104 in the earlier V3 catalogue baseline report; those experiments were not validated as an improvement on the final active-pool protocol. The final linear-output V3 run is the selected retrieval checkpoint.
