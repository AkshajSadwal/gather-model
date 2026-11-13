"""Frozen KuaiRand-to-Gather zero-shot feature adapter.

Gather identifiers never enter KuaiRand ID embeddings. Only semantically verified
fields are mapped; unknown or unavailable categories use the original OOV row 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_SCHEMA_VERSION = "gather-reco-v1"
SOURCE_SCHEMA_FINGERPRINT = "cd94509ad053c55cb973f5dea69546900e3c6d35fc15e90d7308fdbfb46ab82a"
ADAPTER_VERSION = "kuairand-zero-shot-adapter-v1"

CATEGORICAL_ORDER = (
    "user_id", "video_id", "tab",
    "user:user_active_degree", "user:is_lowactive_period", "user:is_live_streamer",
    "user:is_video_author", "user:follow_user_num_range", "user:fans_user_num_range",
    "user:friend_user_num_range", "user:register_days_range",
    *(f"user:onehot_feat{i}" for i in range(18)),
    "video:author_id", "video:video_type", "video:upload_type",
    "video:visible_status", "video:music_id", "video:music_type", "video:tag",
)
NUMERIC_ORDER = (
    "log1p_video_duration_seconds", "hour_sin_utc", "hour_cos_utc",
    "log1p_user_follow_count", "log1p_user_fan_count",
    "log1p_user_friend_count", "log1p_user_register_days",
    "log1p_video_age_hours", "log1p_video_width", "log1p_video_height",
)
RANGE_FIELDS = (
    ("user:follow_user_num_range", "follow_user_num"),
    ("user:fans_user_num_range", "fans_user_num"),
    ("user:friend_user_num_range", "friend_user_num"),
    ("user:register_days_range", "register_days"),
)
RAW_RANGE_TO_GATHER_LOG = (
    ("user:follow_user_num_range", "log1p_user_following_count"),
    ("user:fans_user_num_range", "log1p_user_follower_count"),
    ("user:friend_user_num_range", "log1p_user_friend_count"),
    ("user:register_days_range", "log1p_user_account_age_days"),
)
SOURCE_TO_MODEL_NUMERIC = (
    ("log1p_video_duration_seconds", "log1p_reel_duration_seconds", "reel"),
    ("hour_sin_utc", "hour_sin_utc", "user"),
    ("hour_cos_utc", "hour_cos_utc", "user"),
    ("log1p_user_follow_count", "log1p_user_following_count", "user"),
    ("log1p_user_fan_count", "log1p_user_follower_count", "user"),
    ("log1p_user_friend_count", "log1p_user_friend_count", "user"),
    ("log1p_user_register_days", "log1p_user_account_age_days", "user"),
    ("log1p_video_age_hours", "log1p_reel_age_hours", "reel"),
)


def canonical_fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_tag(value: Any) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return normalized or None


def _parse_bound(value: str) -> float:
    match = re.fullmatch(r"([0-9]+)([kw]?)", value.strip().casefold())
    if not match:
        raise ValueError(f"Unsupported KuaiRand range boundary: {value}")
    amount = float(match.group(1))
    unit = match.group(2)
    return amount * (1000 if unit == "k" else 10000 if unit == "w" else 1)


def parse_range_label(label: str) -> tuple[float | None, bool, float | None, bool] | None:
    label = str(label).strip()
    if label == "__MISSING__":
        return None
    if label == "0":
        return (0.0, True, 0.0, True)
    if match := re.fullmatch(r"\(([^,]+),([^\]]+)\]", label):
        return (_parse_bound(match.group(1)), False, _parse_bound(match.group(2)), True)
    if match := re.fullmatch(r"\[([^,]+),([^\)]+)\)", label):
        return (_parse_bound(match.group(1)), True, _parse_bound(match.group(2)), False)
    if match := re.fullmatch(r"([0-9]+)-([0-9]+)", label):
        return (_parse_bound(match.group(1)), True, _parse_bound(match.group(2)), True)
    if match := re.fullmatch(r"([0-9]+[kw]?)\+", label.casefold()):
        return (_parse_bound(match.group(1)), True, None, False)
    return None


def range_contains(value: float, bounds: tuple[float | None, bool, float | None, bool]) -> bool:
    lower, lower_inclusive, upper, upper_inclusive = bounds
    if lower is not None and (value < lower or (value == lower and not lower_inclusive)):
        return False
    if upper is not None and (value > upper or (value == upper and not upper_inclusive)):
        return False
    return True


def _category_value(value: Any) -> str:
    if value is None:
        return "__MISSING__"
    try:
        if bool(np.isnan(value)):
            return "__MISSING__"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else "__MISSING__"


def _source_number(value: Any, *, nonnegative: bool = True) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (nonnegative and number < 0):
        return None
    return number


def _raw_count_from_log1p(value: Any) -> float | None:
    logged = _source_number(value)
    if logged is None or logged > 60:
        return None
    try:
        raw = math.expm1(logged)
    except OverflowError:
        return None
    return raw if math.isfinite(raw) and raw >= 0 else None


def _source_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    version = value.get("schemaVersion")
    if version not in (None, SOURCE_SCHEMA_VERSION):
        raise ValueError(f"{label} feature schema version mismatch")
    return value


def _range_index(vocabulary: dict[str, int], raw_value: float | None) -> int:
    if raw_value is None:
        return 0
    integer_value = math.floor(raw_value)
    for label, index in vocabulary.items():
        bounds = parse_range_label(label)
        if bounds is not None and range_contains(integer_value, bounds):
            return int(index)
    return 0


class KuaiRandFeatureAdapter:
    def __init__(self, metadata: dict[str, Any], artifact: dict[str, Any]):
        if tuple(metadata.get("categorical_features", ())) != CATEGORICAL_ORDER:
            raise ValueError("KuaiRand categorical feature ordering mismatch")
        if tuple(metadata.get("numeric_features", ())) != NUMERIC_ORDER:
            raise ValueError("KuaiRand numeric feature ordering mismatch")
        if len(metadata.get("categorical_sizes", ())) != len(CATEGORICAL_ORDER):
            raise ValueError("KuaiRand categorical vocabulary size mismatch")
        if len(metadata.get("numeric_mean", ())) != len(NUMERIC_ORDER) or len(metadata.get("numeric_std", ())) != len(NUMERIC_ORDER):
            raise ValueError("KuaiRand numeric normalization metadata mismatch")

        artifact_payload = dict(artifact)
        fingerprint = artifact_payload.pop("fingerprint", None)
        if fingerprint != canonical_fingerprint(artifact_payload):
            raise ValueError("Adapter vocabulary artifact fingerprint mismatch")
        if artifact.get("version") != ADAPTER_VERSION:
            raise ValueError("Unsupported adapter vocabulary artifact")
        if artifact.get("schemaFingerprint") is None or artifact.get("preparedDataFingerprint") is None:
            raise ValueError("Adapter vocabulary artifact is missing prepared-data fingerprints")

        sizes = metadata["categorical_sizes"]
        for feature, _ in RANGE_FIELDS:
            vocab = artifact["rangeVocabularies"].get(feature)
            if not isinstance(vocab, dict) or len(vocab) + 1 != sizes[CATEGORICAL_ORDER.index(feature)]:
                raise ValueError(f"KuaiRand range vocabulary mismatch for {feature}")
            for label in vocab:
                bounds = parse_range_label(label)
                if label != "__MISSING__" and bounds is None:
                    raise ValueError(f"Unverified KuaiRand range label for {feature}")
        tag_vocab = artifact["tagVocabulary"]
        tag_size = sizes[CATEGORICAL_ORDER.index("video:tag")]
        if any(int(index) <= 0 or int(index) >= tag_size for index in tag_vocab.values()):
            raise ValueError("KuaiRand tag vocabulary exceeds the trained embedding table")

        self.metadata = metadata
        self.artifact = artifact
        self.means = np.asarray(metadata["numeric_mean"], dtype=np.float32)
        self.stds = np.asarray(metadata["numeric_std"], dtype=np.float32)
        self.stds = np.where(self.stds > 1e-6, self.stds, 1.0).astype(np.float32)
        self.numeric_training_medians = np.asarray(artifact["numericTrainingMedianStandardized"], dtype=np.float32)
        if self.numeric_training_medians.shape != (len(NUMERIC_ORDER),) or not np.isfinite(self.numeric_training_medians).all():
            raise ValueError("KuaiRand training numeric medians are invalid")
        self.range_vocabs = artifact["rangeVocabularies"]
        self.tag_vocab = artifact["tagVocabulary"]
        self.mapping_report = artifact["mappingReport"]

    @staticmethod
    def _numeric(source: dict[str, Any], key: str, *, nonnegative: bool = True) -> float | None:
        return _source_number(source.get(key), nonnegative=nonnegative)

    def encode(self, user_features: dict[str, Any], reel_features: dict[str, Any],
               now: datetime | None = None) -> tuple[np.ndarray, np.ndarray]:
        user_features = _source_object(user_features, "User")
        reel_features = _source_object(reel_features, "Reel")
        user_cat = user_features.get("categorical") if isinstance(user_features.get("categorical"), dict) else {}
        user_num = user_features.get("numeric") if isinstance(user_features.get("numeric"), dict) else {}
        reel_cat = reel_features.get("categorical") if isinstance(reel_features.get("categorical"), dict) else {}
        reel_num = reel_features.get("numeric") if isinstance(reel_features.get("numeric"), dict) else {}

        categorical = np.zeros(len(CATEGORICAL_ORDER), dtype=np.int64)
        numeric_values: list[float | None] = [None] * len(NUMERIC_ORDER)

        for feature, source_key in RAW_RANGE_TO_GATHER_LOG:
            count = _raw_count_from_log1p(user_num.get(source_key))
            categorical[CATEGORICAL_ORDER.index(feature)] = _range_index(self.range_vocabs[feature], count)

        tag = normalize_tag(reel_cat.get("video:topic"))
        if tag is not None:
            categorical[CATEGORICAL_ORDER.index("video:tag")] = int(self.tag_vocab.get(tag, 0))

        for model_name, source_name, source_group in SOURCE_TO_MODEL_NUMERIC:
            source = user_num if source_group == "user" else reel_num
            allow_negative = model_name in {"hour_sin_utc", "hour_cos_utc"}
            numeric_values[NUMERIC_ORDER.index(model_name)] = self._numeric(source, source_name, nonnegative=not allow_negative)
        current = now if now is not None else datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        hour = current.astimezone(timezone.utc).hour
        angle = 2 * math.pi * hour / 24
        for feature, value in (("hour_sin_utc", math.sin(angle)), ("hour_cos_utc", math.cos(angle))):
            index = NUMERIC_ORDER.index(feature)
            candidate = numeric_values[index]
            if candidate is None:
                numeric_values[index] = value
            else:
                numeric_values[index] = min(1.0, max(-1.0, candidate))

        normalized = self.numeric_training_medians.copy()
        for index, value in enumerate(numeric_values):
            if value is None:
                continue
            standardized = (np.float32(value) - self.means[index]) / self.stds[index]
            if np.isfinite(standardized):
                normalized[index] = standardized
        return categorical, normalized.astype(np.float32, copy=False)


def _verify_range_source_rows(user_table, train_user_ids, range_vocabs, numeric_medians):
    indexed = user_table.drop_duplicates("user_id", keep="first").set_index("user_id")
    aligned = indexed.reindex(train_user_ids)
    for feature, numeric_field in RANGE_FIELDS:
        vocab = range_vocabs[feature]
        source_field = feature.split(":", 1)[1]
        for raw_label, raw_count in zip(aligned[source_field].tolist(), aligned[numeric_field].tolist()):
            label = _category_value(raw_label)
            count = _source_number(raw_count)
            if label == "__MISSING__" or count is None:
                continue
            bounds = parse_range_label(label)
            if bounds is None or not range_contains(math.floor(count), bounds):
                raise ValueError(f"KuaiRand {feature} label does not match its numeric training value")
            if label not in vocab:
                raise ValueError(f"KuaiRand {feature} label is missing from reconstructed vocabulary")


def build_vocab_artifact(root: Path, output_path: Path) -> dict[str, Any]:
    import pandas as pd

    from common.data_version import prepared_data_fingerprint, schema_fingerprint
    from scripts.prepare_data_v2 import (
        USER_CAT, USER_NUM, VIDEO_CAT, clean_cat, encode_cat_lookup, fit_cat_vocab,
    )

    data_dir = root / "data" / "processed" / "ranking" / "1k_v3_catalog"
    raw_dir = root / "data" / "raw" / "KuaiRand-1K" / "data"
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(data_dir / "id_maps.npz", allow_pickle=False) as maps:
        train_user_ids = np.asarray(maps["user_ids"], dtype=np.int64)
        train_video_ids = np.asarray(maps["video_ids"], dtype=np.int64)
        existing_user_categories = np.asarray(maps["user_cat_by_index"], dtype=np.int32)
        existing_video_categories = np.asarray(maps["video_cat_by_index"], dtype=np.int32)

    user_table = pd.read_csv(raw_dir / "user_features_1k.csv", usecols=["user_id", *USER_CAT, *USER_NUM], low_memory=False)
    video_table = pd.read_csv(raw_dir / "video_features_basic_1k.csv", usecols=["video_id", "tag"], low_memory=False)
    user_vocabs, user_sizes = fit_cat_vocab(user_table, "user_id", train_user_ids, USER_CAT)
    tag_vocabs, tag_sizes = fit_cat_vocab(video_table, "video_id", train_video_ids, ["tag"])

    range_vocabs: dict[str, dict[str, int]] = {}
    user_category_columns = {f"user:{name}": index for index, name in enumerate(USER_CAT)}
    for feature, _ in RANGE_FIELDS:
        local_index = user_category_columns[feature]
        range_vocabs[feature] = {str(key): int(value) for key, value in user_vocabs[local_index].items()}
        metadata_index = CATEGORICAL_ORDER.index(feature)
        if user_sizes[local_index] != metadata["categorical_sizes"][metadata_index]:
            raise ValueError(f"Reconstructed KuaiRand vocabulary size differs for {feature}")

    raw_tag_vocab = {str(key): int(value) for key, value in tag_vocabs[0].items()}
    normalized_tag_vocab: dict[str, int] = {}
    for raw_tag, index in raw_tag_vocab.items():
        if raw_tag == "__MISSING__":
            continue
        normalized = normalize_tag(raw_tag)
        if normalized is None:
            continue
        if normalized in normalized_tag_vocab and normalized_tag_vocab[normalized] != index:
            raise ValueError("KuaiRand tag normalization produced an ambiguous category")
        normalized_tag_vocab[normalized] = index
    if tag_sizes[0] != metadata["categorical_sizes"][CATEGORICAL_ORDER.index("video:tag")]:
        raise ValueError("Reconstructed KuaiRand tag vocabulary size mismatch")

    encoded_users = encode_cat_lookup(user_table, "user_id", train_user_ids, USER_CAT, user_vocabs)
    for feature, _ in RANGE_FIELDS:
        local_index = user_category_columns[feature]
        if not np.array_equal(encoded_users[:, local_index], existing_user_categories[:, local_index]):
            raise ValueError(f"Reconstructed {feature} mapping disagrees with id_maps.npz")
    _verify_range_source_rows(user_table, train_user_ids, range_vocabs, metadata["user_numeric_medians"])

    video_indexed = video_table.drop_duplicates("video_id", keep="first").set_index("video_id")
    sample_positions = np.unique(np.linspace(0, len(train_video_ids) - 1, min(256, len(train_video_ids)), dtype=np.int64))
    sample_ids = train_video_ids[sample_positions]
    aligned_tags = video_indexed.reindex(sample_ids)["tag"]
    encoded_tags = clean_cat(aligned_tags).map(tag_vocabs[0]).fillna(0).to_numpy(dtype=np.int32)
    if not np.array_equal(encoded_tags, existing_video_categories[sample_positions + 1, 6]):
        raise ValueError("Reconstructed tag mapping disagrees with sampled id_maps.npz rows")

    with np.load(data_dir / "train.npz", allow_pickle=False) as train:
        numeric_training_medians = np.median(np.asarray(train["numeric"], dtype=np.float32), axis=0).astype(np.float32).tolist()
    if len(numeric_training_medians) != len(NUMERIC_ORDER) or not np.isfinite(numeric_training_medians).all():
        raise ValueError("Could not reconstruct original training-set numeric medians")

    base = {
        "version": ADAPTER_VERSION,
        "schemaFingerprint": schema_fingerprint(data_dir),
        "preparedDataFingerprint": prepared_data_fingerprint(data_dir),
        "rangeVocabularies": range_vocabs,
        "tagVocabulary": normalized_tag_vocab,
        "numericTrainingMedianStandardized": [float(value) for value in numeric_training_medians],
        "mappingReport": {
            "mapped": [
                "user following/follower/friend/account-age ranges using verified KuaiRand labels",
                "user follow/fan/friend/register numeric signals from Gather log1p counts",
                "current UTC hour sine/cosine",
                "Reel duration and age from Gather log1p fields",
                "Reel topic only when its normalized value exactly matches a KuaiRand tag",
            ],
            "approximated": [
                "register_days_range uses floor(account age in days) with verified source range labels",
                "Reel width and height use the original standardized training-set medians because Gather snapshots do not provide dimensions",
            ],
            "unavailable": [
                "user activity flags, live-streamer/video-author status, and onehot_feat0-17",
                "Reel type, upload type, visibility category, music ID, and music type",
            ],
            "oov": [
                "all Gather user IDs, Reel IDs, creator IDs, and the REELS tab value",
                "unknown or missing categories, including unmatched topic tags",
                "unavailable categorical features listed above",
            ],
        },
    }
    artifact = {**base, "fingerprint": canonical_fingerprint(base)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "kuairand_v1_vocab.json")
    args = parser.parse_args()
    artifact = build_vocab_artifact(args.root.resolve(), args.output.resolve())
    print(
        f"adapter={artifact['version']} "
        f"schema={artifact['schemaFingerprint']} "
        f"prepared={artifact['preparedDataFingerprint']} "
        f"tags={len(artifact['tagVocabulary'])} "
        f"fingerprint={artifact['fingerprint']}"
    )


if __name__ == "__main__":
    main()
