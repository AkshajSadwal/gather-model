"""Prepare KuaiRand logs with user/profile and basic-video metadata for Reels ranker V2.

Uses standard-feed logs only. Deliberately excludes video_features_statistic_* because
its aggregate time window may overlap validation/test events (potential leakage).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from common.model_common import BINARY_TARGETS

RAW_COLUMNS = ["user_id", "video_id", "time_ms", "tab", "duration_ms", "play_time_ms",
               "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
               "long_view", "is_profile_enter"]
LABEL_COLUMNS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
                 "long_view", "is_profile_enter"]
USER_CAT = ["user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
            "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range"] + [f"onehot_feat{i}" for i in range(18)]
USER_NUM = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
VIDEO_CAT = ["author_id", "video_type", "upload_type", "visible_status", "music_id", "music_type", "tag"]
VIDEO_NUM = ["video_duration", "server_width", "server_height"]
NUMERIC_FEATURES = ["log1p_video_duration_seconds", "hour_sin_utc", "hour_cos_utc",
                    "log1p_user_follow_count", "log1p_user_fan_count", "log1p_user_friend_count",
                    "log1p_user_register_days", "log1p_video_age_hours", "log1p_video_width", "log1p_video_height"]


def clean_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("__MISSING__").str.strip().replace("", "__MISSING__")


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in RAW_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    dtypes = {"user_id": "int64", "video_id": "int64", "time_ms": "int64", "tab": "int16",
              "duration_ms": "float32", "play_time_ms": "float32"}
    dtypes.update({c: "float32" for c in LABEL_COLUMNS})
    df = pd.read_csv(path, usecols=RAW_COLUMNS, dtype=dtypes, low_memory=False)
    df = df.dropna(subset=["user_id", "video_id", "time_ms"])
    for c in LABEL_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1).astype(np.float32)
    for c in ["duration_ms", "play_time_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0).astype(np.float32)
    df["tab"] = pd.to_numeric(df["tab"], errors="coerce").fillna(-1).astype(np.int16)
    return df.sort_values("time_ms", kind="mergesort").reset_index(drop=True)


def ids_to_index(values: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    """Map raw IDs to indices 1..N; unseen IDs use 0."""
    vals = np.asarray(values, dtype=np.int64)
    ix = np.searchsorted(sorted_ids, vals)
    out = np.zeros(len(vals), dtype=np.int32)
    valid = ix < len(sorted_ids)
    if valid.any():
        eq = np.zeros(len(vals), dtype=bool)
        eq[valid] = sorted_ids[ix[valid]] == vals[valid]
        out[eq] = ix[eq].astype(np.int32) + 1
    return out


def fit_cat_vocab(table: pd.DataFrame, id_col: str, train_ids: np.ndarray, fields: list[str]):
    """Fit category vocabularies from training-period entities only."""
    indexed = table.drop_duplicates(id_col, keep="first").set_index(id_col)
    aligned = indexed.reindex(train_ids)
    vocabs: list[dict[str, int]] = []
    sizes: list[int] = []
    for field in fields:
        values = clean_cat(aligned[field])
        vocabulary = {value: i + 1 for i, value in enumerate(sorted(values.unique().tolist()))}
        vocabs.append(vocabulary)
        sizes.append(len(vocabulary) + 1)
    return vocabs, sizes


def encode_cat_lookup(table: pd.DataFrame, id_col: str, ids: np.ndarray,
                      fields: list[str], vocabs: list[dict[str, int]]) -> np.ndarray:
    """Encode any entity IDs with a training-fitted vocabulary; unknowns use 0."""
    indexed = table.drop_duplicates(id_col, keep="first").set_index(id_col)
    aligned = indexed.reindex(ids)
    lookup = np.zeros((len(ids) + 1, len(fields)), dtype=np.int32)
    for j, field in enumerate(fields):
        lookup[1:, j] = clean_cat(aligned[field]).map(vocabs[j]).fillna(0).to_numpy(dtype=np.int32)
    return lookup


def build_cat_lookup(table: pd.DataFrame, id_col: str, ids: np.ndarray, fields: list[str]):
    vocabs, sizes = fit_cat_vocab(table, id_col, ids, fields)
    return encode_cat_lookup(table, id_col, ids, fields, vocabs), sizes


def median_nonnegative(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(lower=0)
    med = x.median()
    return float(med) if pd.notna(med) and np.isfinite(med) else 0.0


def build_user_numeric(user_table: pd.DataFrame, user_ids: np.ndarray):
    table = user_table.drop_duplicates("user_id", keep="first").set_index("user_id")
    aligned = table.reindex(user_ids)
    out = np.zeros((len(user_ids) + 1, len(USER_NUM)), dtype=np.float32)
    medians = {}
    for j, field in enumerate(USER_NUM):
        values = pd.to_numeric(aligned[field], errors="coerce").replace([np.inf, -np.inf], np.nan).clip(lower=0)
        med = values.median()
        med = float(med) if pd.notna(med) else 0.0
        medians[field] = med
        values = values.fillna(med).to_numpy(dtype=np.float32)
        out[1:, j] = np.log1p(np.maximum(values, 0.0))
    return out, medians


def build_video_numeric(video_table: pd.DataFrame, video_ids: np.ndarray,
                        duration_fallback_ms: float, medians: dict[str, float] | None = None):
    """Build item numeric maps; medians are fitted on train IDs and reused for catalogue IDs."""
    table = video_table.drop_duplicates("video_id", keep="first").set_index("video_id")
    aligned = table.reindex(video_ids)
    out_dur = np.zeros(len(video_ids) + 1, dtype=np.float32)  # seconds, not log transformed
    out_width = np.zeros(len(video_ids) + 1, dtype=np.float32)
    out_height = np.zeros(len(video_ids) + 1, dtype=np.float32)
    fitted_medians = dict(medians or {})
    for field in VIDEO_NUM:
        vals = pd.to_numeric(aligned[field], errors="coerce").replace([np.inf, -np.inf], np.nan).clip(lower=0)
        if field not in fitted_medians:
            med = vals.median()
            fitted_medians[field] = float(med) if pd.notna(med) else 0.0
        vals = vals.fillna(fitted_medians[field]).to_numpy(dtype=np.float32)
        if field == "video_duration":
            vals_ms = np.where(vals > 0, vals, duration_fallback_ms)
            out_dur[1:] = vals_ms / 1000.0
        elif field == "server_width":
            out_width[1:] = vals
        else:
            out_height[1:] = vals
    upload = pd.to_datetime(aligned["upload_dt"], errors="coerce", utc=True)
    upload_ns = upload.astype("int64", copy=False).to_numpy(dtype=np.int64)
    upload_ms = upload_ns.astype(np.float64) / 1_000_000.0
    upload_ms[pd.isna(upload).to_numpy()] = 0.0
    out_upload_ms = np.zeros(len(video_ids) + 1, dtype=np.float64)
    out_upload_ms[1:] = upload_ms
    return out_dur, out_width, out_height, out_upload_ms, fitted_medians


def pack_split(df: pd.DataFrame, user_ids: np.ndarray, video_ids: np.ndarray, tab_ids: np.ndarray,
               user_cat_lookup: np.ndarray, catalog_video_ids: np.ndarray,
               catalog_video_cat_lookup: np.ndarray, user_numeric_lookup: np.ndarray,
               catalog_video_duration: np.ndarray, catalog_video_width: np.ndarray,
               catalog_video_height: np.ndarray, catalog_video_upload_ms: np.ndarray):
    ui = ids_to_index(df["user_id"].to_numpy(), user_ids)
    vi = ids_to_index(df["video_id"].to_numpy(), video_ids)
    cvi = ids_to_index(df["video_id"].to_numpy(), catalog_video_ids)
    ti = ids_to_index(df["tab"].to_numpy(), tab_ids)
    # Keep the train-vocabulary video ID in column 1. For an unseen video this
    # is OOV=0, but known metadata from the basic-video table remains available.
    cat = np.column_stack([ui, vi, ti, user_cat_lookup[ui], catalog_video_cat_lookup[cvi]]).astype(np.int32, copy=False)

    durations = catalog_video_duration[cvi].copy()
    missing_dur = durations <= 0
    durations[missing_dur] = df["duration_ms"].to_numpy(dtype=np.float32)[missing_dur] / 1000.0
    durations = np.maximum(durations, 0.0)
    times = df["time_ms"].to_numpy(dtype=np.int64)
    hours = ((times // 3_600_000) % 24).astype(np.float32)
    angle = 2 * np.pi * hours / 24.0
    upload_ms = catalog_video_upload_ms[cvi]
    age_hours = np.where(upload_ms > 0, np.maximum((times - upload_ms) / 3_600_000.0, 0.0), 0.0).astype(np.float32)
    user_nums = user_numeric_lookup[ui]
    numeric_raw = np.column_stack([
        np.log1p(durations), np.sin(angle), np.cos(angle), user_nums,
        np.log1p(age_hours), np.log1p(np.maximum(catalog_video_width[cvi], 0.0)),
        np.log1p(np.maximum(catalog_video_height[cvi], 0.0)),
    ]).astype(np.float32)

    frame = df.copy()
    duration_log = frame["duration_ms"].to_numpy(dtype=np.float32)
    watched = frame["play_time_ms"].to_numpy(dtype=np.float32)
    frame["complete_view"] = ((duration_log > 0) & (watched >= duration_log)).astype(np.float32)
    frame["short_view"] = ((duration_log > 0) & (watched < np.minimum(3000.0, duration_log))).astype(np.float32)
    labels = frame[BINARY_TARGETS].to_numpy(dtype=np.float32, copy=True)
    watch = np.log1p(watched / 1000.0).astype(np.float32)
    return {"categorical": cat, "numeric_raw": numeric_raw, "labels": labels,
            "watch_log_seconds": watch, "catalog_video_indices": cvi.astype(np.int32, copy=False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["pure", "1k"], default="pure")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed/ranking/pure_v2")
    args = parser.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "pure" if args.dataset == "pure" else "1k"
    user_feature_path = data_dir / f"user_features_{suffix}.csv"
    video_feature_path = data_dir / f"video_features_basic_{suffix}.csv"
    old_path = data_dir / f"log_standard_4_08_to_4_21_{suffix}.csv"
    new_path = data_dir / f"log_standard_4_22_to_5_08_{suffix}.csv"

    print(f"Loading earlier standard-feed log: {old_path}")
    older = read_log(old_path); print(f"  rows: {len(older):,}")
    print(f"Loading newer standard-feed log: {new_path}")
    newer = read_log(new_path); print(f"  rows: {len(newer):,}")
    if len(newer) < 10: raise ValueError("Newer log has too few rows for chronological split.")
    latest_timestamp = int(max(older["time_ms"].max(), newer["time_ms"].max()))
    n = len(newer); train_cut, val_cut = int(n * .80), int(n * .90)
    recent_train = newer.iloc[:train_cut].copy()
    validation = newer.iloc[train_cut:val_cut].copy()
    test = newer.iloc[val_cut:].copy()
    train = pd.concat([older, recent_train], ignore_index=True)
    del older, newer, recent_train

    user_ids = np.sort(train["user_id"].unique().astype(np.int64))
    video_ids = np.sort(train["video_id"].unique().astype(np.int64))
    tab_ids = np.sort(train["tab"].unique().astype(np.int64))

    print(f"Loading user features: {user_feature_path}")
    user_table = pd.read_csv(user_feature_path, low_memory=False)
    print(f"Loading basic video features: {video_feature_path}")
    video_table = pd.read_csv(video_feature_path, low_memory=False)
    for col in ["user_id", *USER_CAT, *USER_NUM]:
        if col not in user_table.columns: raise ValueError(f"User feature table missing required column: {col}")
    for col in ["video_id", *VIDEO_CAT, *VIDEO_NUM, "upload_dt"]:
        if col not in video_table.columns: raise ValueError(f"Video feature table missing required column: {col}")

    print("Encoding user metadata categories...")
    user_cat_lookup, user_cat_sizes = build_cat_lookup(user_table, "user_id", user_ids, USER_CAT)
    print("Fitting video metadata vocabularies on training-period videos...")
    video_cat_vocabs, video_cat_sizes = fit_cat_vocab(video_table, "video_id", video_ids, VIDEO_CAT)
    # The full basic-video feature table is the cold-start catalogue. Category
    # vocabularies remain train-only; unseen categories map to OOV=0.
    catalog_video_ids = np.sort(video_table["video_id"].dropna().astype(np.int64).unique())
    print(f"Encoding metadata for {len(catalog_video_ids):,} catalogue videos...")
    catalog_video_cat_lookup = encode_cat_lookup(
        video_table, "video_id", catalog_video_ids, VIDEO_CAT, video_cat_vocabs)
    user_numeric_lookup, user_numeric_medians = build_user_numeric(user_table, user_ids)
    train_median_duration = median_nonnegative(train["duration_ms"])
    (video_duration, video_width, video_height, video_upload_ms,
     video_numeric_medians) = build_video_numeric(video_table, video_ids, train_median_duration)
    (catalog_video_duration, catalog_video_width, catalog_video_height,
     catalog_video_upload_ms, _) = build_video_numeric(
        video_table, catalog_video_ids, train_median_duration, medians=video_numeric_medians)

    print("Packing interaction arrays (retaining metadata for unseen video IDs)...")
    packed = {}
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        packed[name] = pack_split(
            frame, user_ids, video_ids, tab_ids, user_cat_lookup,
            catalog_video_ids, catalog_video_cat_lookup, user_numeric_lookup,
            catalog_video_duration, catalog_video_width, catalog_video_height, catalog_video_upload_ms)
        with_metadata = int(np.count_nonzero(packed[name]["catalog_video_indices"]))
        print(f"  {name}: {len(frame):,} rows; video metadata available: {with_metadata:,} ({with_metadata/max(len(frame), 1):.1%})")
    train_num = packed["train"].pop("numeric_raw")
    means = train_num.mean(axis=0).astype(np.float32)
    stds = train_num.std(axis=0).astype(np.float32)
    stds[~np.isfinite(stds) | (stds < 1e-6)] = 1.0
    packed["train"]["numeric"] = ((train_num - means) / stds).astype(np.float32)
    del train_num
    for name in ["validation", "test"]:
        raw = packed[name].pop("numeric_raw")
        packed[name]["numeric"] = ((raw - means) / stds).astype(np.float32)
        del raw

    for name, arrays in packed.items():
        path = out / f"{name}.npz"
        print(f"Writing {path} ...")
        np.savez_compressed(path, **arrays)
    feature_cat_names = ["user_id", "video_id", "tab"] + [f"user:{x}" for x in USER_CAT] + [f"video:{x}" for x in VIDEO_CAT]
    cat_sizes = [len(user_ids)+1, len(video_ids)+1, len(tab_ids)+1] + user_cat_sizes + video_cat_sizes
    # Keep id_maps.npz byte-for-byte schema-compatible with existing ranking
    # checkpoints. The separate catalogue map can grow with cold-start videos.
    np.savez_compressed(out / "id_maps.npz", user_ids=user_ids, video_ids=video_ids, tab_ids=tab_ids,
                        user_cat_by_index=user_cat_lookup,
                        video_cat_by_index=encode_cat_lookup(video_table, "video_id", video_ids, VIDEO_CAT, video_cat_vocabs),
                        user_numeric_by_index=user_numeric_lookup, video_duration_seconds_by_index=video_duration,
                        video_width_by_index=video_width, video_height_by_index=video_height,
                        video_upload_ms_by_index=video_upload_ms)
    catalog_video_model_indices = np.concatenate([
        np.zeros(1, dtype=np.int32), ids_to_index(catalog_video_ids, video_ids)
    ])
    np.savez_compressed(
        out / "catalog_maps.npz", video_ids=catalog_video_ids,
        video_model_indices_by_index=catalog_video_model_indices,
        video_cat_by_index=catalog_video_cat_lookup,
        video_duration_seconds_by_index=catalog_video_duration,
        video_width_by_index=catalog_video_width,
        video_height_by_index=catalog_video_height,
        video_upload_ms_by_index=catalog_video_upload_ms,
    )
    metadata = {
        "model_version": 2, "dataset": args.dataset, "num_users": int(len(user_ids)+1),
        "num_videos": int(len(video_ids)+1), "num_tabs": int(len(tab_ids)+1),
        "categorical_features": feature_cat_names, "categorical_sizes": [int(x) for x in cat_sizes],
        "numeric_features": NUMERIC_FEATURES, "numeric_mean": means.tolist(), "numeric_std": stds.tolist(),
        "binary_targets": BINARY_TARGETS, "user_categorical_features": USER_CAT,
        "video_categorical_features": VIDEO_CAT, "user_numeric_features": USER_NUM,
        "user_numeric_medians": user_numeric_medians, "video_numeric_medians": video_numeric_medians,
        "video_duration_fallback_ms": train_median_duration,
        "reference_time_ms": latest_timestamp,
        "training_rows": int(len(train)), "validation_rows": int(len(validation)), "test_rows": int(len(test)),
        "split_policy": "older standard logs + first 80% of newer log train; next 10% validation; final 10% test, ordered by time",
        "excluded_features": ["video_features_statistic_*", "random-exposure logs"],
        "caveat": "Static user features are used as provided in the dataset; their exact as-of timestamp is not supplied. Video aggregate statistics are excluded due to possible temporal leakage.",
        "id_zero_policy": "0 is reserved for OOV/missing ID or feature values",
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f: json.dump(metadata, f, indent=2)
    print("\nV2 preparation complete.")
    print(f"Categorical fields: {len(cat_sizes)}; numeric fields: {len(NUMERIC_FEATURES)}")
    print(f"Processed files: {out.resolve()}")
    print(f"Next: python -m model2_ranking.train_v2 --data-dir {out} --output-dir artifacts/ranking/{args.dataset}_v2 --epochs 5 --batch-size 4096")

if __name__ == "__main__": main()
