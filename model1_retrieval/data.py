"""Data adapters for the prepared KuaiRand V2 arrays and item catalogue."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.data_version import prepared_data_fingerprint, schema_fingerprint as data_fingerprint
from model1_retrieval.model import TwoTowerRetrieval


@dataclass
class PreparedSplit:
    categorical: np.ndarray
    numeric: np.ndarray
    labels: np.ndarray
    catalog_video_indices: np.ndarray | None = None

    @property
    def rows(self) -> int:
        return int(self.categorical.shape[0])



def load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared-data metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"categorical_sizes", "numeric_features", "numeric_mean", "numeric_std", "binary_targets"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"Prepared metadata is missing required keys: {sorted(missing)}")
    if len(metadata["categorical_sizes"]) != 36 or len(metadata["numeric_features"]) != 10:
        raise ValueError("Retrieval expects the V2 prepared schema: 36 categorical and 10 numeric features")
    return metadata


def load_split(data_dir: Path, name: str) -> PreparedSplit:
    path = data_dir / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared split: {path}. Run scripts.prepare_data_v2 first.")
    with np.load(path, allow_pickle=False) as payload:
        categorical = np.asarray(payload["categorical"], dtype=np.int32)
        numeric = np.asarray(payload["numeric"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.float32)
        catalog_video_indices = (np.asarray(payload["catalog_video_indices"], dtype=np.int32)
                                 if "catalog_video_indices" in payload.files else None)
    if categorical.ndim != 2 or categorical.shape[1] != 36:
        raise ValueError(f"{path}: expected categorical shape [N, 36], got {categorical.shape}")
    if numeric.ndim != 2 or numeric.shape[1] != 10:
        raise ValueError(f"{path}: expected numeric shape [N, 10], got {numeric.shape}")
    if labels.ndim != 2 or labels.shape != (categorical.shape[0], 10):
        raise ValueError(f"{path}: expected labels shape [N, 10], got {labels.shape}")
    if catalog_video_indices is not None and catalog_video_indices.shape != (categorical.shape[0],):
        raise ValueError(f"{path}: catalog_video_indices must have shape [N]")
    if np.any(categorical < 0):
        raise ValueError(f"{path}: categorical indices must be nonnegative (0 is the OOV index)")
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path}: numeric features contain NaN or infinite values")
    if not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1)):
        raise ValueError(f"{path}: labels must be finite values in [0, 1]")
    if catalog_video_indices is not None and np.any(catalog_video_indices < 0):
        raise ValueError(f"{path}: catalogue video indices must be nonnegative")
    return PreparedSplit(categorical, numeric, labels, catalog_video_indices)


def target_indices(metadata: dict[str, Any]) -> tuple[dict[str, int], dict[str, float]]:
    names = list(metadata["binary_targets"])
    weights = {
        "is_click": 0.25,
        "is_like": 1.0,
        "is_follow": 1.5,
        "is_comment": 1.25,
        "is_forward": 1.5,
        "long_view": 2.0,
        "is_profile_enter": 0.75,
        "complete_view": 2.0,
    }
    missing = [name for name in weights if name not in names]
    if missing:
        raise ValueError(f"Prepared labels lack retrieval positive targets: {missing}")
    if "is_hate" not in names:
        raise ValueError("Prepared labels lack is_hate; explicit negative sampling cannot be enabled")
    if "short_view" not in names:
        raise ValueError("Prepared labels lack short_view; retrieval needs the early-skip negative signal")
    return {name: names.index(name) for name in names}, weights


def item_catalog_indices(split: PreparedSplit) -> np.ndarray:
    """Return 1-based full-catalogue indices, falling back to train video IDs for legacy data."""
    if split.catalog_video_indices is not None:
        return split.catalog_video_indices.astype(np.int32, copy=False)
    return split.categorical[:, 1].astype(np.int32, copy=False)


def positive_rows(split: PreparedSplit, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    indices, weights = target_indices(metadata)
    strength = np.zeros(split.rows, dtype=np.float32)
    for name, weight in weights.items():
        strength += split.labels[:, indices[name]] * float(weight)
    strength -= split.labels[:, indices["short_view"]] * 0.5
    hate = split.labels[:, indices["is_hate"]] > 0
    item_indices = item_catalog_indices(split)
    mask = (strength > 0) & ~hate
    mask &= (split.categorical[:, 0] > 0) & (item_indices > 0)
    row_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    row_weights = np.clip(strength[row_indices], 0.25, 5.0)
    if row_weights.size:
        row_weights = (row_weights / max(float(row_weights.mean()), 1e-6)).astype(np.float32)
    return row_indices, row_weights


def explicit_negative_pairs(split: PreparedSplit, metadata: dict[str, Any], positive_pair_keys: np.ndarray,
                            num_videos: int) -> dict[int, np.ndarray]:
    """Return per-user explicit hate/short-view catalogue items, excluding known positives."""
    indices, _ = target_indices(metadata)
    negative_mask = (split.labels[:, indices["is_hate"]] > 0) | (split.labels[:, indices["short_view"]] > 0)
    item_indices = item_catalog_indices(split)
    negative_mask &= (split.categorical[:, 0] > 0) & (item_indices > 0)
    rows = np.flatnonzero(negative_mask)
    if rows.size == 0:
        return {}
    users = split.categorical[rows, 0].astype(np.int64)
    videos = item_indices[rows].astype(np.int64)
    keys = users * int(num_videos) + videos
    idx = np.searchsorted(positive_pair_keys, keys)
    if positive_pair_keys.size:
        safe = np.minimum(idx, positive_pair_keys.size - 1)
        is_positive = (idx < positive_pair_keys.size) & (positive_pair_keys[safe] == keys)
    else:
        is_positive = np.zeros(keys.shape, dtype=bool)
    users, videos = users[~is_positive], videos[~is_positive]
    result: dict[int, np.ndarray] = {}
    for user in np.unique(users):
        result[int(user)] = np.unique(videos[users == user]).astype(np.int32)
    return result


def pair_keys(split: PreparedSplit, rows: np.ndarray, num_videos: int) -> np.ndarray:
    items = item_catalog_indices(split)[rows].astype(np.int64, copy=False)
    users = split.categorical[rows, 0].astype(np.int64, copy=False)
    return np.unique(users * int(num_videos) + items)


def build_inbatch_candidates(user_indices: np.ndarray, positive_video_indices: np.ndarray,
                             num_videos: int, positive_keys: np.ndarray,
                             hard_negative_by_user: dict[int, np.ndarray], rng: np.random.Generator,
                             hard_negative_sample_rate: float = 0.25,
                             row_block_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Build a deduplicated in-batch item set and multi-positive mask.

    A batch item is positive for a user whenever its (user, item) key exists in
    the complete known-positive training set. This masks duplicate/multi-positive
    items instead of incorrectly treating them as negatives.
    """
    users = np.asarray(user_indices, dtype=np.int64)
    positives = np.asarray(positive_video_indices, dtype=np.int32)
    if users.ndim != 1 or positives.shape != users.shape:
        raise ValueError("user_indices and positive_video_indices must be matching 1-D arrays")
    if not len(users):
        raise ValueError("Cannot build in-batch candidates from an empty batch")
    candidates = positives.tolist()
    if hard_negative_sample_rate > 0:
        for user in np.unique(users):
            hard = hard_negative_by_user.get(int(user))
            if hard is not None and hard.size and rng.random() < hard_negative_sample_rate:
                candidates.append(int(rng.choice(hard)))
    candidate_ids = np.unique(np.asarray(candidates, dtype=np.int32))
    if np.any(candidate_ids <= 0) or np.any(candidate_ids >= int(num_videos)):
        raise ValueError("In-batch candidate indices must be within the catalogue's 1-based range")
    positive_mask = np.zeros((len(users), len(candidate_ids)), dtype=bool)
    # Avoid a B x C int64 temporary for the full 4096 batch: calculate known
    # positive membership in bounded row blocks.
    for start in range(0, len(users), max(1, row_block_size)):
        end = min(start + max(1, row_block_size), len(users))
        keys = users[start:end, None] * int(num_videos) + candidate_ids[None, :].astype(np.int64)
        locations = np.searchsorted(positive_keys, keys)
        if positive_keys.size:
            safe = np.minimum(locations, positive_keys.size - 1)
            positive_mask[start:end] = (locations < positive_keys.size) & (positive_keys[safe] == keys)
    if not positive_mask.any(axis=1).all():
        raise RuntimeError("An in-batch row has no known positive candidate; positive-pair masking is inconsistent")
    return candidate_ids, positive_mask


def sample_negative_video_indices(user_indices: np.ndarray, positive_video_indices: np.ndarray,
                                  num_videos: int, positive_keys: np.ndarray,
                                  hard_negative_by_user: dict[int, np.ndarray], rng: np.random.Generator,
                                  negatives_per_positive: int, hard_negative_sample_rate: float) -> np.ndarray:
    batch = len(user_indices)
    negatives = rng.integers(1, num_videos, size=(batch, negatives_per_positive), dtype=np.int32)

    # Reject known positive user-video pairs; other unobserved catalogue items are
    # sampled negatives, not asserted dislikes. Retry if a sampled item is known positive.
    for _ in range(20):
        keys = user_indices.astype(np.int64)[:, None] * int(num_videos) + negatives.astype(np.int64)
        locations = np.searchsorted(positive_keys, keys)
        safe_locations = np.minimum(locations, max(positive_keys.size - 1, 0))
        collision = (locations < positive_keys.size) & (positive_keys[safe_locations] == keys) if positive_keys.size else np.zeros(keys.shape, dtype=bool)
        collision |= negatives == positive_video_indices[:, None]
        if not collision.any():
            break
        negatives[collision] = rng.integers(1, num_videos, size=int(collision.sum()), dtype=np.int32)

    if hard_negative_sample_rate > 0:
        for row, user in enumerate(user_indices.tolist()):
            hard_negatives = hard_negative_by_user.get(int(user))
            if hard_negatives is not None and hard_negatives.size and rng.random() < hard_negative_sample_rate:
                candidate = int(rng.choice(hard_negatives))
                if candidate != int(positive_video_indices[row]):
                    negatives[row, int(rng.integers(0, negatives_per_positive))] = candidate
    return negatives


@dataclass
class VideoCatalog:
    categorical: np.ndarray  # row 0 is the reserved OOV row
    numeric: np.ndarray      # standardized using the ranking dataset's training statistics
    raw_ids: np.ndarray      # rows correspond to model indices 1..N


def build_video_catalog(data_dir: Path, metadata: dict[str, Any]) -> VideoCatalog:
    """Build a static item catalogue including metadata-only cold-start videos when available."""
    catalog_path = data_dir / "catalog_maps.npz"
    maps_path = catalog_path if catalog_path.exists() else data_dir / "id_maps.npz"
    if not maps_path.exists():
        raise FileNotFoundError(f"Missing {maps_path}; regenerate the processed dataset with prepare_data_v2.py")
    with np.load(maps_path, allow_pickle=False) as maps:
        raw_ids = np.asarray(maps["video_ids"], dtype=np.int64)
        video_cat_by_index = np.asarray(maps["video_cat_by_index"], dtype=np.int32)
        duration = np.asarray(maps["video_duration_seconds_by_index"], dtype=np.float32)
        width = np.asarray(maps["video_width_by_index"], dtype=np.float32)
        height = np.asarray(maps["video_height_by_index"], dtype=np.float32)
        if "video_model_indices_by_index" in maps.files:
            video_model_indices = np.asarray(maps["video_model_indices_by_index"], dtype=np.int32)
        else:
            video_model_indices = np.arange(len(raw_ids) + 1, dtype=np.int32)

    n = len(raw_ids)
    if video_cat_by_index.shape[0] != n + 1 or video_model_indices.shape != (n + 1,):
        raise ValueError("Video catalogue feature maps do not align with catalogue video IDs")
    if np.any(video_model_indices < 0) or np.any(video_model_indices >= int(metadata["categorical_sizes"][1])):
        raise ValueError("Catalogue contains a video-ID embedding index outside the training vocabulary")
    categorical = np.column_stack([video_model_indices, video_cat_by_index]).astype(np.int32, copy=False)

    duration = duration.copy()
    fallback_ms = float(metadata.get("video_duration_fallback_ms", 0.0))
    duration[duration <= 0] = fallback_ms / 1000.0
    numeric_raw = np.column_stack([
        np.log1p(np.maximum(duration, 0.0)),
        np.log1p(np.maximum(width, 0.0)),
        np.log1p(np.maximum(height, 0.0)),
    ]).astype(np.float32)
    means_all = np.asarray(metadata["numeric_mean"], dtype=np.float32)
    stds_all = np.asarray(metadata["numeric_std"], dtype=np.float32)
    cols = [0, 8, 9]
    means, stds = means_all[cols], stds_all[cols]
    stds = np.where(stds > 1e-6, stds, 1.0)
    numeric = ((numeric_raw - means) / stds).astype(np.float32)
    numeric[0] = 0.0
    return VideoCatalog(categorical, numeric, raw_ids)


def build_validation_queries(split: PreparedSplit, metadata: dict[str, Any], catalog: VideoCatalog,
                              train_positive_keys: np.ndarray, seed: int, query_count: int,
                              negative_count: int) -> tuple[np.ndarray, np.ndarray]:
    all_positive_rows, _ = positive_rows(split, metadata)
    catalog_indices = item_catalog_indices(split)
    all_positive_rows = all_positive_rows[(catalog_indices[all_positive_rows] > 0) &
                                          (catalog_indices[all_positive_rows] < catalog.categorical.shape[0])]
    if all_positive_rows.size == 0:
        raise ValueError("Validation split contains no eligible positive interactions for sampled retrieval evaluation")
    rng = np.random.default_rng(seed)
    rows = all_positive_rows
    if rows.size > query_count:
        rows = rng.choice(rows, size=query_count, replace=False)
    users = split.categorical[rows, 0].astype(np.int64)
    positives = catalog_indices[rows].astype(np.int32)
    n_videos = catalog.categorical.shape[0]
    all_validation_positive_keys = pair_keys(split, all_positive_rows, n_videos)
    all_positive = np.unique(np.concatenate([train_positive_keys, all_validation_positive_keys]))
    candidates = np.empty((len(rows), negative_count + 1), dtype=np.int32)
    candidates[:, 0] = positives
    for i, user in enumerate(users):
        chosen: set[int] = {int(positives[i])}
        attempts = 0
        while len(chosen) < negative_count + 1 and attempts < negative_count * 30 + 100:
            candidate = int(rng.integers(1, n_videos))
            key = int(user) * n_videos + candidate
            at = np.searchsorted(all_positive, key)
            known_positive = at < all_positive.size and int(all_positive[min(at, all_positive.size - 1)]) == key
            if candidate not in chosen and not known_positive:
                chosen.add(candidate)
            attempts += 1
        sampled = sorted(chosen - {int(positives[i])})
        if len(sampled) < negative_count:
            raise ValueError(
                f"Could only sample {len(sampled)} distinct negatives for a query; "
                f"requested {negative_count}. Reduce --eval-negatives or use a larger catalogue."
            )
        candidates[i, 1:] = np.asarray(sampled[:negative_count], dtype=np.int32)
    return rows.astype(np.int64), candidates


def user_features_from_raw_id(data_dir: Path, metadata: dict[str, Any], raw_user_id: int,
                              tab_raw_id: int, hour_utc: float) -> tuple[np.ndarray, np.ndarray, bool]:
    maps_path = data_dir / "id_maps.npz"
    with np.load(maps_path, allow_pickle=False) as maps:
        user_ids = np.asarray(maps["user_ids"], dtype=np.int64)
        tab_ids = np.asarray(maps["tab_ids"], dtype=np.int64)
        user_cat = np.asarray(maps["user_cat_by_index"], dtype=np.int32)
        user_num = np.asarray(maps["user_numeric_by_index"], dtype=np.float32)
    u_pos = int(np.searchsorted(user_ids, raw_user_id))
    known_user = u_pos < len(user_ids) and int(user_ids[u_pos]) == int(raw_user_id)
    user_index = u_pos + 1 if known_user else 0
    t_pos = int(np.searchsorted(tab_ids, tab_raw_id))
    tab_index = t_pos + 1 if t_pos < len(tab_ids) and int(tab_ids[t_pos]) == int(tab_raw_id) else 0
    hour = float(hour_utc) % 24.0
    angle = 2.0 * np.pi * hour / 24.0
    raw_numeric = np.concatenate([
        np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float32),
        user_num[user_index].astype(np.float32, copy=False),
    ])
    means = np.asarray(metadata["numeric_mean"], dtype=np.float32)[1:7]
    stds = np.asarray(metadata["numeric_std"], dtype=np.float32)[1:7]
    stds = np.where(stds > 1e-6, stds, 1.0)
    numeric = ((raw_numeric - means) / stds).astype(np.float32)
    categorical = np.concatenate([
        np.asarray([user_index, tab_index], dtype=np.int32), user_cat[user_index].astype(np.int32, copy=False)
    ])[None, :]
    return categorical, numeric[None, :], known_user
