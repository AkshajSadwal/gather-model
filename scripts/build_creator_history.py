"""Build a leakage-safe chronological user-to-creator positive-event history index.

The artifact contains positive interaction events from train, validation and test,
ordered per user by timestamp. A consumer MUST query it with query_time_ms; the
index strictly excludes events at or after that timestamp. This supports replaying
validation/test chronologically without letting a row see itself, simultaneous
rows, or future interactions.

Example:
    python -m scripts.build_creator_history \
        --data-dir data/processed/ranking/1k_v3_catalog \
        --output-dir data/processed/history/creator_history_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common.creator_history import CreatorHistoryIndex, build_creator_events
from common.data_version import prepared_data_fingerprint, schema_fingerprint

SPLITS = (("train", 0), ("validation", 1), ("test", 2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(data_dir: Path, output_dir: Path, overwrite: bool = False) -> tuple[Path, Path]:
    metadata_path = data_dir / "metadata.json"
    maps_path = data_dir / "catalog_maps.npz"
    times_path = data_dir / "row_times.npz"
    for path in (metadata_path, maps_path, times_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. The V3 catalogue data and row_times.npz are required."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "creator_history.npz"
    manifest_path = output_dir / "metadata.json"
    if not overwrite and (artifact_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Creator history already exists in {output_dir}; pass --overwrite to replace it."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_names = list(metadata["binary_targets"])
    feature_names = list(metadata["categorical_features"])
    try:
        author_column = feature_names.index("video:author_id")
    except ValueError as exc:
        raise ValueError("Prepared schema has no video:author_id feature") from exc
    num_users = int(metadata["num_users"])
    num_authors = int(metadata["categorical_sizes"][author_column])

    with np.load(maps_path, allow_pickle=False) as maps:
        video_cat = np.asarray(maps["video_cat_by_index"], dtype=np.int32)
    # catalog_maps stores VIDEO_CAT in the order declared by prepare_data_v2;
    # author_id is its first feature (full ranker feature column 29).
    if video_cat.ndim != 2 or video_cat.shape[1] < 1:
        raise ValueError("catalog_maps.video_cat_by_index has no author feature")
    catalog_author_ids = video_cat[:, 0].astype(np.int32, copy=False)
    if catalog_author_ids.size == 0 or int(catalog_author_ids.max(initial=0)) >= num_authors:
        raise ValueError("Catalogue author indices do not match the metadata vocabulary")

    blocks: list[dict[str, np.ndarray]] = []
    total_rows = 0
    split_counts: dict[str, int] = {}
    with np.load(times_path, allow_pickle=False) as row_times:
        for split_name, split_code in SPLITS:
            split_path = data_dir / f"{split_name}.npz"
            if not split_path.exists():
                raise FileNotFoundError(split_path)
            times = np.asarray(row_times[split_name], dtype=np.int64)
            with np.load(split_path, allow_pickle=False) as payload:
                categorical = np.asarray(payload["categorical"], dtype=np.int32)
                labels = np.asarray(payload["labels"], dtype=np.float32)
                if "catalog_video_indices" not in payload.files:
                    raise ValueError(
                        f"{split_path} lacks catalog_video_indices; rebuild the V3 catalogue dataset."
                    )
                video_indices = np.asarray(payload["catalog_video_indices"], dtype=np.int32)
            if categorical.shape[0] != times.size or labels.shape[0] != times.size or video_indices.shape != times.shape:
                raise ValueError(f"{split_name}: arrays and row_times.npz are not row-aligned")
            if categorical.ndim != 2 or categorical.shape[1] < 1:
                raise ValueError(f"{split_name}: malformed categorical array")

            block = build_creator_events(
                users=categorical[:, 0],
                catalog_video_indices=video_indices,
                labels=labels,
                times_ms=times,
                catalog_author_ids=catalog_author_ids,
                target_names=target_names,
                num_users=num_users,
                num_authors=num_authors,
                split_code=split_code,
            )
            split_counts[split_name] = int(block["users"].size)
            total_rows += int(times.size)
            print(
                f"{split_name}: {times.size:,} rows -> "
                f"{block['users'].size:,} valid positive creator events",
                flush=True,
            )
            blocks.append(block)

    if not blocks or sum(split_counts.values()) == 0:
        raise ValueError("No valid positive creator events were produced")

    users = np.concatenate([b["users"] for b in blocks])
    authors = np.concatenate([b["authors"] for b in blocks])
    times = np.concatenate([b["times_ms"] for b in blocks])
    weights = np.concatenate([b["weights"] for b in blocks])
    split_codes = np.concatenate([b["split_codes"] for b in blocks])

    index = CreatorHistoryIndex.from_events(
        users, authors, times, weights, num_users=num_users, num_authors=num_authors
    )
    # Store the event split code in the same per-user/time order as the index.
    order = np.lexsort((times, users))
    split_codes = split_codes[order]

    np.savez_compressed(
        artifact_path,
        offsets=index.offsets,
        times_ms=index.times_ms,
        author_ids=index.author_ids,
        weights=index.weights,
        split_codes=split_codes,
    )
    manifest = {
        "artifact_version": 1,
        "artifact": artifact_path.name,
        "data_dir": str(data_dir),
        "schema_fingerprint": schema_fingerprint(data_dir),
        "prepared_data_fingerprint": prepared_data_fingerprint(data_dir),
        "row_times_sha256": sha256_file(times_path),
        "num_users": num_users,
        "num_authors": num_authors,
        "total_source_rows": total_rows,
        "positive_creator_events_by_split": split_counts,
        "total_positive_creator_events": int(users.size),
        "minimum_event_time_ms": int(index.times_ms.min()),
        "maximum_event_time_ms": int(index.times_ms.max()),
        "split_codes": {"train": 0, "validation": 1, "test": 2},
        "event_definition": (
            "Same positive weights and short_view penalty as model1_retrieval.data.positive_rows; "
            "explicit is_hate rows excluded; strength clipped to [0.25, 5.0] without split-level normalisation."
        ),
        "time_policy": (
            "Queries must use event_time_ms < query_time_ms. Events at exactly the query timestamp "
            "are excluded together; this prevents same-time and self-event leakage."
        ),
        "validation_test_policy": (
            "Validation/test events are stored for chronological replay only. A query may use an "
            "earlier event from those splits only when its timestamp is strictly before the query."
        ),
        "author_policy": "Only known nonzero author-vocabulary IDs are stored; OOV/missing authors are excluded.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved history artifact: {artifact_path.resolve()}")
    print(f"Saved history metadata: {manifest_path.resolve()}")
    print("Strict as-of policy: event_time_ms < query_time_ms")
    return artifact_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v3_catalog")
    parser.add_argument("--output-dir", default="data/processed/history/creator_history_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(Path(args.data_dir), Path(args.output_dir), overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
