
## Retrieval feature split details

Model 1 source input column positions are explicit. The user tower categorical columns are `[0, 2, 3..28]`: `user_id`, `tab`, then the 26 fields prefixed `user:` in prepared metadata order. User numeric columns are `[1..6]`: `hour_sin_utc`, `hour_cos_utc`, `log1p_user_follow_count`, `log1p_user_fan_count`, `log1p_user_friend_count`, `log1p_user_register_days`. The video tower categorical columns are `[1, 29..35]`: `video_id`, then the seven `video:` fields. Video numeric columns are `[0,8,9]`: log1p duration, width, height. Video age is intentionally excluded from static cached vectors. Both towers are independent `512 -> 256 -> 128` MLPs; normalized dot product is the retrieval similarity.

The index builder writes item vectors batchwise into a float16 NumPy memory-mapped matrix with width 128. Retrieval defaults to chunks of 131,072 item rows, scores each chunk against the user vector, and keeps partial top-K with `argpartition`. Runtime validation must compare checkpoint SHA-256, data fingerprint, catalogue-map SHA-256, dimension, dtype and normalization. Model 1 selected run metadata names dataset `1k`, seed 42, 14 epochs, pool-softmax training, and validation hit@1% selection across three time windows. These remain KuaiRand provenance.

Model 2's binary loss uses batchwise positive weights `clamp(negative_count / max(positive_count, 1), 1, 20)` and adds `0.20 * SmoothL1` for watch target. Its recommendation script applies sigmoid, and converts the watch head using `expm1(clamp(raw, min=0))`. The KuaiRand script's multi-head score weights are not carried into Gather; Gather's initial primary head is `is_like` only.

## Frozen zero-shot serving decision

The current Gather integration does not copy or fine-tune any KuaiRand tensor. It loads the selected original Model 1 and Model 2 checkpoints unchanged with their original KuaiRand schema, then explicitly adapts Gather feature snapshots at inference time. This is frozen zero-shot inference, not transfer learning and not a Gather-trained model. All Gather user, Reel, and creator IDs use OOV index 0. Model 1 scores the eligible Gather candidate pool directly; the KuaiRand vector index remains untouched and is never queried for Gather results. Model 2's primary score is sigmoid(is_like_logit).

## Per-weight transfer decision for future Gather-trained models

No tensor in the selected checkpoints is approved for direct reuse in a Gather model. Specifically:

| Tensor group | Why shape inspection is insufficient | Gather action |
|---|---|---|
| Model 1 user/reel ID embeddings, including the 3,929,860-row reel-ID table and `id_keep` mask | IDs and train-frequency gates encode the KuaiRand 1k catalog/user maps | Reinitialize from Gather-only vocabularies |
| Model 1 side-category embeddings | The 36 ordered fields have different category values and vocabularies; Gather schema fingerprint differs | Rebuild vocabularies; reinitialize |
| Model 1 first dense layers `[512,159]` / `[512,60]` | Their input coordinates are sums/concatenations of incompatible learned embeddings and different feature transforms | Reinitialize |
| Model 1 deeper dense layers | Same dimensions do not establish aligned hidden representations; no paired representation alignment set exists | Reinitialize unless a future controlled transfer experiment proves benefit |
| Model 2's 36 embedding tables | Includes massive KuaiRand user/video/author/music vocabularies; no Gather index mapping exists | Reinitialize all tables |
| Model 2 first dense layer `[512,257]` | Input includes KuaiRand embedding coordinates and KuaiRand standardized numeric distributions | Reinitialize |
| Model 2 shared hidden layers `[512,512]`, `[256,512]`, `[128,256]` | Shape-compatible only; prior latent space and behavioral sampling differ | No direct transfer absent a validated alignment experiment |
| Model 2 action/watch heads | Gather label availability/definitions differ (`complete_view` 90% and `short_view` 25% versus KuaiRand full-duration and `min(3s,duration)`); watch units transform align but latent and target distributions do not | Reinitialize all heads |
| Retrieval item index | Contains KuaiRand video IDs and is tied to the checkpoint/catalog hashes | Keep untouched; never query it for Gather. Encode the supplied eligible Gather candidates directly |

Even the shared `is_like` name and `log1p(seconds)` watch target do not prove weight compatibility. No arbitrary tensor copying or fingerprint substitution is allowed.

Serving note (2026-11-14): the frozen kuairand_zero_shot_v1 runtime is ready with adapter fingerprint 1fc0fea6c5e75d6948183afb9469dc34bfe62d6c67bd8a1e41ad05dbdcff0bf1 and selected checkpoints SHA-256 6f3114de877b4c2a3770b750704bb0cae6e0a7e2441b6ac7e17ee1162946888e and a4c91097fd409145253e9f43454b80852018286e9da5c26b870a75872d47ba8e. It validates the original KuaiRand schema (7bbaadb7410036832014bdf7f3a225671f84bf60c8226c2f98ae78ca5ebb67e7) and prepared-data fingerprint (a9e0107ff2818fba39e378746664cf1a1e70402af11ca0e62d85097c2da1195f). Its adapter preserves the exact 36/10 feature orders, uses reconstructed training-only range/tag maps checked against id_maps.npz, normalizes with the original KuaiRand means/stds, and keeps IDs/unsupported categories OOV. Service tests and local CUDA/CPU inference passed; recommendation quality remains unvalidated on Gather.
