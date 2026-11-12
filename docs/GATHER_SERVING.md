# Gather private inference service

The persistent FastAPI contract is implemented under `serving/`. It intentionally reports `COLD_START` until Gather-trained artifacts and a real inference runtime pass schema, checkpoint, vocabulary, catalog and index fingerprint checks. It never attempts to load the selected KuaiRand references.

PowerShell local setup (from `A:\reels-recommender-m4`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-serving.txt
$env:REELS_RECOMMENDER_TOKEN = '<set a local random service token>'
uvicorn serving.app:app --host localhost --port 8100 --workers 1
```

Configure Gather with `REELS_RECOMMENDER_URL=http://localhost:8100` and the same token. Restrict production service traffic to the private backend network, terminate TLS at the private ingress, and rotate service credentials through deployment secret management. Never expose the service publicly or put a token in source control. `/health/live` is process liveness; `/health/ready` distinguishes `COLD_START` from model readiness. Current implementation always returns `COLD_START`; model inference is intentionally blocked pending compatible Gather-trained models and an available PyTorch runtime.

The service reads `REELS_RECOMMENDER_ROOT` (default: the ML repository root), `GATHER_MODEL_MANIFEST`, `GATHER_RETRIEVAL_CHECKPOINT`, `GATHER_RANKING_CHECKPOINT`, and `GATHER_RETRIEVAL_INDEX` as environment-configurable artifact paths. Example PowerShell path setup (the Gather-specific files do not yet exist, so these are placeholders only):

```powershell
$env:REELS_RECOMMENDER_ROOT = 'A:\reels-recommender-m4'
$env:GATHER_MODEL_MANIFEST = 'A:\reels-recommender-m4\artifacts\gather\manifest.json'
$env:GATHER_RETRIEVAL_CHECKPOINT = 'A:\reels-recommender-m4\artifacts\gather\retrieval.pt'
$env:GATHER_RANKING_CHECKPOINT = 'A:\reels-recommender-m4\artifacts\gather\ranking.pt'
$env:GATHER_RETRIEVAL_INDEX = 'A:\reels-recommender-m4\artifacts\gather\index'
```

The current service does not load these artifact paths yet; it stays `COLD_START` until the separate validated inference runtime is implemented.

## Persistent two-stage inference

`serving/app.py` loads the Gather `TwoTowerRetrieval`, `ReelsRankerV2`, versioned categorical mappings and memory-mapped full-catalogue index once at startup. It validates the canonical feature fingerprint, prepared-data schema fingerprint, retrieval checkpoint SHA-256, catalogue-map SHA-256, ranking target/vocabulary metadata, and index dimensions before becoming `READY`. Device selection uses CUDA when available and CPU otherwise. `is_like` sigmoid probability is the ranking score; predicted watch time is returned as `expm1(clamp(log_seconds, 0, log1p(3600)))`.

The service retrieves against all indexed reels and intersects those IDs with the privacy-eligible features supplied by Gather before Model 2 ranking. This allows the backend to recheck current block, mute, private-account, archive and moderation rules. Candidates carry a real Gather reel ID and its opaque model key; the service returns the real ID only after matching a currently eligible candidate. The backend verifies every result against the eligible candidate set. Missing artifacts or any compatibility mismatch return readiness `COLD_START`; the API does not substitute KuaiRand recommendations.

Optional paths: `GATHER_DATA_DIR` (defaults to `data/processed/gather_v1`) and `GATHER_ARTIFACT_DIR` (defaults to `artifacts/gather`). Optional `GATHER_ACTIVE_MODEL_MANIFEST` selects the promoted pointer (defaults to `artifacts/gather/active_gather_model.json`). `GATHER_MODEL_MANIFEST`, `GATHER_RETRIEVAL_CHECKPOINT`, `GATHER_RANKING_CHECKPOINT`, and `GATHER_RETRIEVAL_INDEX` override individual paths. The serving dependency file is `requirements-serving.txt`; installing it is an operator action. The repository venv currently lacks FastAPI, so service startup was not run here.

Train a ranker only after `prepare_gather_dataset.py` reports READY and first-party minimums are met. Evaluate held-out data with `scripts/evaluate_gather_v1.py`; promotion is a separate explicit operation through `scripts/promote_gather_model.py --activate --approval PROMOTE` with operator-selected minimum `is_like` AUC and Recall@100 thresholds. No training, evaluation against real Gather data, or promotion was run during this change.
