"""Prepare versioned, temporal Gather recommendation splits without fabricating examples."""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

LABEL_SCHEMA = "gather-labels-v1"
SCHEMA = "gather-reco-v1"
POSITIVE = {"LIKE", "CREATOR_FOLLOW", "COMMENT", "SHARE"}
DEFAULTS = {"min_users": 100, "min_impressions": 10000, "min_positive_rows": 1000, "min_reels": 100, "min_creators": 20, "min_coverage_days": 14}

def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def flatten_keys(value: Any) -> set[str]:
    if not isinstance(value, dict): return set()
    keys = set(value)
    for child in value.values(): keys.update(flatten_keys(child))
    return keys

def row_features(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    categorical = {**row["userFeatures"]["categorical"], **row["reelFeatures"]["categorical"]}
    numeric = {**row["userFeatures"]["numeric"], **row["reelFeatures"]["numeric"]}
    return categorical, numeric

def invalid_label_rows(rows: list[dict[str, Any]], target_names: list[str]) -> int:
    invalid = 0
    for row in rows:
        labels, available = row.get("labels"), row.get("labelsAvailable")
        if not isinstance(labels, dict) or not isinstance(available, dict) or any(name not in labels or name not in available for name in target_names):
            invalid += 1; continue
        for name in target_names:
            if available[name] and (labels[name] not in (0, 1, 0.0, 1.0)):
                invalid += 1; break
        watch = row.get("watchLogSeconds")
        if row.get("watchLabelAvailable") and (not isinstance(watch, (int, float)) or not np.isfinite(watch) or watch < 0): invalid += 1
    return invalid
def write_model_arrays(splits: dict[str, list[dict[str, Any]]], output_dir: Path, manifest: dict[str, Any], schema_fingerprint: str, source_sha256: str) -> dict[str, Any]:
    cat_names, numeric_names = manifest["categoricalFeatures"], manifest["numericFeatures"]
    target_names = manifest["ranking"]["binaryTargets"]
    train_cats = [row_features(row)[0] for row in splits["train"]]
    train_nums = [row_features(row)[1] for row in splits["train"]]
    vocabularies = {}
    for name in cat_names:
        values = sorted({str(values.get(name) if values.get(name) not in (None, "") else "__MISSING__") for values in train_cats})
        vocabularies[name] = {value: index + 1 for index, value in enumerate(values)}
    numeric_medians, numeric_mean, numeric_std = [], [], []
    for name in numeric_names:
        values = np.asarray([float(row[name]) if row.get(name) is not None and np.isfinite(float(row[name])) else np.nan for row in train_nums], dtype=np.float64)
        valid = values[np.isfinite(values)]
        median = float(np.median(valid)) if valid.size else 0.0
        values = np.where(np.isfinite(values), values, median)
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std()) if values.size else 1.0
        if not np.isfinite(std) or std <= 1e-6: std = 1.0
        numeric_medians.append(median); numeric_mean.append(mean); numeric_std.append(std)

    def encode_split(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        cats, nums, labels, masks, watches, watch_masks, catalog_indices, times = [], [], [], [], [], [], [], []
        for row in rows:
            categorical, numeric = row_features(row)
            cats.append([vocabularies[name].get(str(categorical.get(name) if categorical.get(name) not in (None, "") else "__MISSING__"), 0) for name in cat_names])
            raw = np.asarray([float(numeric[name]) if numeric.get(name) is not None and np.isfinite(float(numeric[name])) else numeric_medians[i] for i, name in enumerate(numeric_names)], dtype=np.float64)
            nums.append((raw - np.asarray(numeric_mean)) / np.asarray(numeric_std))
            available = row["labelsAvailable"]
            values = row["labels"]
            labels.append([float(values.get(name, 0)) if available.get(name, False) else 0.0 for name in target_names])
            masks.append([1.0 if available.get(name, False) else 0.0 for name in target_names])
            watches.append(float(row.get("watchLogSeconds") or 0.0))
            watch_masks.append(1.0 if row.get("watchLabelAvailable") else 0.0)
            catalog_indices.append(vocabularies["reel_id"].get(row["reelKey"], 0))
            times.append(int(parse_time(row["servedAt"]).timestamp() * 1000))
        return {"categorical": np.asarray(cats, dtype=np.int32), "numeric": np.asarray(nums, dtype=np.float32),
                "labels": np.asarray(labels, dtype=np.float32), "label_mask": np.asarray(masks, dtype=np.float32),
                "watch_log_seconds": np.asarray(watches, dtype=np.float32), "watch_mask": np.asarray(watch_masks, dtype=np.float32),
                "catalog_video_indices": np.asarray(catalog_indices, dtype=np.int32), "row_times_ms": np.asarray(times, dtype=np.int64)}

    encoded = {name: encode_split(rows) for name, rows in splits.items()}
    for name, data in encoded.items():
        np.savez_compressed(output_dir / f"{name}.npz", **{key: value for key, value in data.items() if key != "row_times_ms"})
    np.savez_compressed(output_dir / "row_times.npz", **{name: data["row_times_ms"] for name, data in encoded.items()})

    train_by_user, train_row_by_reel = {}, {}
    for row, cat, num in zip(splits["train"], encoded["train"]["categorical"], encoded["train"]["numeric"]):
        train_by_user.setdefault(row["viewerKey"], (cat, num))
        train_row_by_reel.setdefault(row["reelKey"], row)
    user_keys = sorted(train_by_user); reel_keys = sorted(train_row_by_reel)
    user_id_to_index = {key: index + 1 for index, key in enumerate(user_keys)}
    reel_id_to_index = {key: index + 1 for index, key in enumerate(reel_keys)}
    video_columns = [cat_names.index(name) for name in ("video:creator_id", "video:topic", "video:language", "video:duration_bucket", "video:hashtag_group", "video:creator_age_bucket", "video:creator_visibility")]
    user_columns = list(range(3, 29))
    catalog_cats = np.zeros((len(reel_keys) + 1, 7), dtype=np.int32)
    catalog_model_ids = np.zeros(len(reel_keys) + 1, dtype=np.int32)
    duration = np.zeros(len(reel_keys) + 1, dtype=np.float32)
    creator_followers = np.zeros(len(reel_keys) + 1, dtype=np.float32)
    media_bytes = np.zeros(len(reel_keys) + 1, dtype=np.float32)
    for key, index in reel_id_to_index.items():
        categorical, numeric = row_features(train_row_by_reel[key])
        catalog_cats[index] = [vocabularies[name].get(str(categorical.get(name) or "__MISSING__"), 0) for name in ("video:creator_id", "video:topic", "video:language", "video:duration_bucket", "video:hashtag_group", "video:creator_age_bucket", "video:creator_visibility")]
        catalog_model_ids[index] = index
        duration[index] = float(np.expm1(max(0.0, float(numeric["log1p_reel_duration_seconds"] or 0))))
        creator_followers[index] = float(np.expm1(max(0.0, float(numeric["log1p_creator_follower_count"] or 0))))
        media_bytes[index] = float(np.expm1(max(0.0, float(numeric["log1p_reel_media_bytes"] or 0))))
    np.savez_compressed(output_dir / "catalog_maps.npz", video_ids=np.arange(1, len(reel_keys) + 1, dtype=np.int64), video_model_indices_by_index=catalog_model_ids, video_cat_by_index=catalog_cats, video_duration_seconds_by_index=duration, video_width_by_index=creator_followers, video_height_by_index=media_bytes)
    user_cat = np.zeros((len(user_keys) + 1, 26), dtype=np.int32); user_numeric = np.zeros((len(user_keys) + 1, 6), dtype=np.float32)
    for key, index in user_id_to_index.items():
        categorical, numeric = train_by_user[key]
        user_cat[index] = categorical[user_columns]
        user_numeric[index] = numeric[[1, 2, 3, 4, 5, 6]]
    np.savez_compressed(output_dir / "id_maps.npz", user_ids=np.arange(1, len(user_keys) + 1, dtype=np.int64), video_ids=np.arange(1, len(reel_keys) + 1, dtype=np.int64), tab_ids=np.asarray([1], dtype=np.int64), user_cat_by_index=user_cat, video_cat_by_index=catalog_cats, user_numeric_by_index=user_numeric, video_duration_seconds_by_index=duration, video_width_by_index=creator_followers, video_height_by_index=media_bytes, video_upload_ms_by_index=np.zeros(len(reel_keys) + 1, dtype=np.int64))
    mappings = {"schemaVersion": SCHEMA, "schemaFingerprint": schema_fingerprint, "categoricalVocabularies": vocabularies, "userIdToIndex": user_id_to_index, "reelIdToIndex": reel_id_to_index, "creatorIdToIndex": vocabularies["video:creator_id"], "indexToReelId": {str(index): key for key, index in reel_id_to_index.items()}}
    (output_dir / "feature_mappings.json").write_text(json.dumps(mappings, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    metadata = {"model_version": 1, "dataset": "gather_v1", "schema_version": SCHEMA, "schema_fingerprint": schema_fingerprint, "prepared_data_fingerprint": source_sha256, "num_users": len(user_keys) + 1, "num_videos": len(reel_keys) + 1, "num_tabs": 2, "categorical_features": cat_names, "categorical_sizes": [len(vocabularies[name]) + 1 for name in cat_names], "numeric_features": numeric_names, "numeric_mean": numeric_mean, "numeric_std": numeric_std, "numeric_medians": numeric_medians, "binary_targets": target_names, "user_categorical_features": cat_names[3:29], "video_categorical_features": cat_names[29:36], "split_policy": "chronological 80/10/10; vocabularies and numeric statistics fit on train only", "train_rows": len(splits["train"]), "validation_rows": len(splits["validation"]), "test_rows": len(splits["test"]), "id_zero_policy": "OOV and missing index 0"}
    return metadata

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in DEFAULTS.items(): parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args()
    digest = hashlib.sha256()
    rows = []
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            digest.update(line.encode("utf-8"))
            if line.strip(): rows.append(json.loads(line))
    rows.sort(key=lambda row: (parse_time(row["servedAt"]), row.get("requestId", ""), row.get("reelKey", "")))
    users = {r.get("viewerKey") for r in rows if r.get("viewerKey")}
    reels = {r.get("reelKey") for r in rows if r.get("reelKey")}
    creators = {r.get("creatorKey") for r in rows if r.get("creatorKey")}
    positives = sum(bool(POSITIVE.intersection(r.get("actions") or [])) for r in rows)
    times = sorted(parse_time(r["servedAt"]) for r in rows if r.get("servedAt"))
    coverage = (times[-1] - times[0]).total_seconds() / 86400 if len(times) > 1 else 0
    manifest = json.loads((Path(__file__).resolve().parents[1] / "configs" / "gather_recommendation_schema_v1.json").read_text(encoding="utf-8"))
    manifest_payload = dict(manifest)
    expected_schema_fingerprint = manifest_payload.pop("fingerprint", None)
    actual_schema_fingerprint = hashlib.sha256(json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    required_features = set(manifest["categoricalFeatures"]) | set(manifest["numericFeatures"])
    missing_feature_fields = sum(len(required_features - (flatten_keys(r.get("userFeatures")) | flatten_keys(r.get("reelFeatures")))) for r in rows)
    missing_snapshots = sum(not isinstance(r.get("userFeatures"), dict) or not isinstance(r.get("reelFeatures"), dict) for r in rows)
    labels = Counter(label for row in rows for label, available in (row.get("labelsAvailable") or {}).items() if available)
    invalid_labels = invalid_label_rows(rows, manifest["ranking"]["binaryTargets"])
    checks = {"label_values": invalid_labels == 0, "users": len(users) >= args.min_users, "impressions": len(rows) >= args.min_impressions, "positive_rows": positives >= args.min_positive_rows, "reels": len(reels) >= args.min_reels, "creators": len(creators) >= args.min_creators, "time_coverage": coverage >= args.min_coverage_days, "snapshots": missing_snapshots == 0 and missing_feature_fields == 0, "schema": expected_schema_fingerprint == actual_schema_fingerprint and all(r.get("labelSchemaVersion") == LABEL_SCHEMA and r.get("schemaVersion") == SCHEMA for r in rows)}
    report = {"status": "READY" if all(checks.values()) else "NOT_READY", "checks": checks, "rows": len(rows), "users": len(users), "reels": len(reels), "creators": len(creators), "positive_rows": positives, "coverage_days": round(coverage, 2), "rows_missing_snapshots": missing_snapshots, "missing_feature_fields": missing_feature_fields, "invalid_label_rows": invalid_labels, "feature_schema_version": manifest["schemaVersion"], "feature_schema_fingerprint": actual_schema_fingerprint, "label_rows_available": dict(labels), "source_sha256": digest.hexdigest(), "schema_version": SCHEMA, "label_schema_version": LABEL_SCHEMA}
    if not all(checks.values()):
        print(json.dumps(report, sort_keys=True)); return 2
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {args.output_dir}")
    train_end, valid_end = int(len(rows) * .8), int(len(rows) * .9)
    splits = {"train": rows[:train_end], "validation": rows[train_end:valid_end], "test": rows[valid_end:]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as output:
            for row in split: output.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    model_metadata = write_model_arrays(splits, args.output_dir, manifest, actual_schema_fingerprint, digest.hexdigest())
    report.update(model_metadata)
    report.update({"status": "READY", "schema_fingerprint": actual_schema_fingerprint, "prepared_data_fingerprint": digest.hexdigest(), "split_rows": {name: len(split) for name, split in splits.items()}, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True)); return 0

if __name__ == "__main__": sys.exit(main())
