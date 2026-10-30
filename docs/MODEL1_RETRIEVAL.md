# Model 1 — two-tower retrieval implementation

## Scope

Model 1 is an offline retrieval baseline built against the project's prepared KuaiRand V2 schema. It is designed to retrieve a few hundred candidate video IDs for Model 2. It does not fetch media, serve a live feed, or build an approximate-nearest-neighbour (ANN) index.

## Architecture

The user and video towers are separate networks. Each uses three hidden layers (512 -> 256 -> 128), ReLU and dropout rates of 0.20, 0.15 and 0.10. Both emit L2-normalized 128-dimensional vectors; dot product is the retrieval score.

The user tower uses user ID, tab, 26 user profile categorical features, hour-of-day sin/cos and four user numeric profile values. The video tower uses video ID, seven video metadata categorical fields, log duration, log width and log height. The video ID embedding is 16-dimensional because the KuaiRand-1K catalogue contains nearly four million videos. Video age is intentionally excluded from the cacheable retrieval item vector; Model 2 keeps the per-impression video-age feature.

For the uploaded KuaiRand-1K metadata, the current model has 92,732,511 parameters. FP32 weights are about 0.35 GiB; a simple weights + gradients + two AdamW states estimate is about 1.38 GiB. This estimate excludes the prepared arrays, activations, temporary buffers, index, and operating-system/MPS memory.

## Training objective

Training uses a pairwise softplus ranking loss. Each positive interaction is compared with two sampled negative video IDs by default. Positive strengths are weighted as follows:

| Event | Weight |
|---|---:|
| Click | 0.25 |
| Like | 1.00 |
| Follow | 1.50 |
| Comment | 1.25 |
| Forward | 1.50 |
| Long view | 2.00 |
| Profile entry | 0.75 |
| Complete view | 2.00 |

A `short_view` row subtracts 0.5 from its positive strength. A click-only short skip is therefore not selected as a positive retrieval example, while stronger engagement can outweigh this penalty. `is_hate` and `short_view` pairs can also be sampled as explicit hard negatives, unless that user-video pair is already represented as a positive in the training split.

Random unobserved catalogue items are implicit sampled negatives, not verified dislikes. This creates exposure/label uncertainty: the dataset does not prove that a user saw and rejected every randomly sampled item. The model should be evaluated and ultimately retrained with trustworthy impression logging and exposure-aware negative sampling when those data are available.

The agreed training batch size is 4,096 positive interaction rows. With two negatives per positive, a full batch runs the video tower for 4,096 positives plus 8,192 negatives (12,288 item examples total). This is pairwise sampled-negative training, not a 4,096-by-4,096 in-batch similarity matrix.

## Evaluation

Validation samples up to 512 positive interactions; each positive is ranked against 500 sampled catalogue negatives by default. The trainer reports sampled Recall@20, sampled Recall@100, MRR and median sampled rank. These values are conditional on the sampled candidate protocol and are **not full-catalogue recall**. All training-seed repeats use one fixed evaluation seed (default `314159`), so they are compared on the exact same validation queries and sampled negatives. The held-out sampled test candidate set uses a separate fixed seed derived from the evaluation seed.

Three sequential training seeds (17, 42, 2026) are the default repeatability check; all share `--eval-seed 314159`. This is an initial stability check, not a statistical guarantee. MPS operations may not be bitwise deterministic.

## Index and query path

`build_index.py` encodes all mapped catalogue videos in chunks and stores float16 vectors plus raw video IDs. For 3.93 million videos, the vector matrix is about 1.0 GB decimal (~0.94 GiB). The builder verifies the dataset schema and checkpoint hash, refuses to silently overwrite a populated index, and preserves a timestamped backup when `--overwrite` is explicitly supplied.

`retrieve.py` computes the user vector and performs exact chunked dot-product scanning over the saved item vectors. It can exclude raw video IDs from a text file. This is a local exact baseline, not ANN: catalogue latency and throughput must be benchmarked, and a production-scale service should evaluate an ANN index such as FAISS or an equivalent vector-search system.

The retrieval output CSV contains raw `video_id` values and similarity scores. Pass that file to `model2_ranking.recommend_v2 --candidate-file` to produce the two-stage offline ranking result.

## Reproducibility and limitations

Prepared data are content-fingerprinted. A resume is rejected if the schema, ID maps, or train/validation/test arrays have changed. Each repeat has its own seed folder, checkpoint and log. The full KuaiRand-1K MPS training runs have not yet been executed from this implementation; first run the documented smoke test on the M4 and inspect memory pressure before running all three full repeats.

Before calling this production-ready, the project still needs online impression/action logging, exposure-aware evaluation, user/item freshness, authoritative seen/policy filters, ANN indexing, service APIs, latency/load tests, monitoring, privacy controls, and online A/B tests.
