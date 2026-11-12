# Gather first-party dataset readiness

Run only against an explicit export. The preparation script never invents examples and does not emit train/validation/test files unless every configured threshold and both point-in-time feature snapshots pass. Defaults are conservative and configurable on the command line: 100 users, 10,000 impressions, 1,000 positive rows, 100 reels, 20 creators, and 14 days of temporal coverage. These are initial operational gates, not statistical guarantees; evaluate uncertainty and temporal holdouts before promotion.

The script computes a SHA-256 over the source JSONL, checks `gather-labels-v1`, requires recommendation-time user and reel snapshots, then makes chronological 80/10/10 splits. Positive rows are rows with a like, follow, comment, or share event. Short views alone do not count as positive; unseen reels are absent from the exporter. Label availability is recorded rather than fabricating click or other missing labels.

PowerShell:

```powershell
node backend/scripts/export-reco-training-data.js exports/reco-training.jsonl
python A:\reels-recommender-m4\scripts\prepare_gather_dataset.py exports\reco-training.jsonl --output-dir data\gather\v1
```

Training and index generation must remain blocked until readiness is met. The exported pseudonymous keys are stable within this exporter, not a substitute for a feature vocabulary manifest.
