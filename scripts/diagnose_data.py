"""Diagnose the prepared ranking/retrieval arrays before spending hours on retraining.

Answers, with numbers from your own data:
  * How many validation/test rows use a video (or user) the model never saw in training,
    and therefore lose all video metadata (cold start)?
  * How concentrated is exposure on a few popular videos (how strong is a popularity baseline)?
  * How often is a validation (user, video) pair already present in training?
  * Do event rates drift between train, validation and test?
  * Do duration-sensitive labels (complete/long/short view) mostly just encode video length?

Uses NumPy only (no torch) and never touches model checkpoints.

    python -m scripts.diagnose_data --data-dir data/processed/ranking/1k_v2 \
        --output reports/experiments/data_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.eval_metrics import relevance_gains

VIDEO_META_COLUMNS = slice(29, 36)  # author, type, upload type, visibility, music, music type, tag
DURATION_COLUMN = 0                  # standardised log1p(video duration seconds)


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "categorical": np.asarray(data["categorical"], dtype=np.int32),
            "numeric": np.asarray(data["numeric"], dtype=np.float32),
            "labels": np.asarray(data["labels"], dtype=np.float32),
        }


def pct(x: float) -> float:
    return round(100.0 * float(x), 3)


def split_summary(name: str, split: dict, targets: list[str], train_counts: np.ndarray) -> dict:
    cat, labels = split["categorical"], split["labels"]
    gain = relevance_gains(labels, targets)
    positive = gain > 0
    video = cat[:, 1]
    unseen_video = video == 0
    meta_missing = ~np.any(cat[:, VIDEO_META_COLUMNS] != 0, axis=1)
    rare = (video > 0) & (train_counts[video] < 5)
    return {
        "rows": int(cat.shape[0]),
        "pct_unseen_user": pct(np.mean(cat[:, 0] == 0)),
        "pct_unseen_video": pct(np.mean(unseen_video)),
        "pct_rows_with_no_video_metadata": pct(np.mean(meta_missing)),
        "pct_video_seen_fewer_than_5_times_in_train": pct(np.mean(rare)),
        "pct_positive_rows_with_unseen_video": pct(np.mean(unseen_video[positive])) if positive.any() else None,
        "pct_positive_rows_with_rare_video": pct(np.mean(rare[positive])) if positive.any() else None,
        "event_rates_pct": {t: pct(labels[:, i].mean()) for i, t in enumerate(targets)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v2")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    targets = list(metadata["binary_targets"])
    n_users = int(metadata["categorical_sizes"][0])
    n_videos = int(metadata["categorical_sizes"][1])

    print("Loading prepared splits (a minute or two for the 1K data)...")
    train = load_split(data_dir / "train.npz")
    validation = load_split(data_dir / "validation.npz")
    test = load_split(data_dir / "test.npz")
    report: dict = {"data_dir": str(data_dir), "dataset": metadata.get("dataset")}

    train_video = train["categorical"][:, 1]
    train_counts = np.bincount(train_video, minlength=n_videos)
    report["train"] = split_summary("train", train, targets, train_counts)
    report["validation"] = split_summary("validation", validation, targets, train_counts)
    report["test"] = split_summary("test", test, targets, train_counts)

    # Exposure concentration (how strong would "just show popular videos" be?)
    positive_counts = np.sort(train_counts[train_counts > 0])[::-1]
    total = positive_counts.sum()
    top1 = max(1, int(0.01 * positive_counts.size))
    top10 = max(1, int(0.10 * positive_counts.size))
    popular_cut = positive_counts[top1 - 1]
    val_video = validation["categorical"][:, 1]
    report["concentration"] = {
        "unique_train_videos": int(positive_counts.size),
        "median_train_exposures_per_video": float(np.median(positive_counts)),
        "pct_train_rows_on_top_1pct_videos": pct(positive_counts[:top1].sum() / total),
        "pct_train_rows_on_top_10pct_videos": pct(positive_counts[:top10].sum() / total),
        "pct_validation_rows_on_train_top_1pct_videos": pct(np.mean(train_counts[val_video] >= popular_cut)),
    }

    # Re-exposure: pairs seen in training also appearing later
    def pair_keys(split: dict) -> np.ndarray:
        c = split["categorical"]
        return c[:, 0].astype(np.int64) * n_videos + c[:, 1].astype(np.int64)

    train_keys = np.unique(pair_keys(train))
    seen_pairs = {}
    for name, split in (("validation", validation), ("test", test)):
        keys = pair_keys(split)
        known = (split["categorical"][:, 1] > 0) & np.isin(keys, train_keys)
        seen_pairs[name] = pct(np.mean(known))
    report["pct_rows_whose_user_video_pair_is_in_train"] = seen_pairs

    # Per-user volume
    per_user = np.bincount(train["categorical"][:, 0], minlength=n_users)[1:]
    per_user = per_user[per_user > 0]
    report["train_rows_per_user"] = {
        "users": int(per_user.size), "min": int(per_user.min()),
        "median": int(np.median(per_user)), "max": int(per_user.max()),
    }

    # Duration bias of duration-sensitive labels, by decile of video duration
    duration = train["numeric"][:, DURATION_COLUMN]
    edges = np.quantile(duration, np.linspace(0, 1, 11))
    bucket = np.clip(np.searchsorted(edges, duration, side="right") - 1, 0, 9)
    names = ["complete_view", "long_view", "short_view", "is_like"]
    rows = []
    for b in range(10):
        mask = bucket == b
        rows.append({"duration_decile": b + 1,
                     **{n: pct(train["labels"][mask, targets.index(n)].mean()) for n in names}})
    report["event_rate_pct_by_duration_decile"] = rows

    # ---- readable summary
    print("\n=== Cold start (rows using IDs the model never trained on) ===")
    for name in ("validation", "test"):
        s = report[name]
        print(f"{name:<11} unseen video {s['pct_unseen_video']:>6}% | no video metadata {s['pct_rows_with_no_video_metadata']:>6}% | "
              f"video seen <5x in train {s['pct_video_seen_fewer_than_5_times_in_train']:>6}% | "
              f"positives on unseen video {s['pct_positive_rows_with_unseen_video']}%")
    print("\n=== Popularity concentration ===")
    for key, value in report["concentration"].items():
        print(f"  {key}: {value}")
    print(f"\n=== Rows whose (user, video) pair already occurred in train (%) === {seen_pairs}")
    print(f"=== Train rows per user === {report['train_rows_per_user']}")
    print("\n=== Event rates (%) by split ===")
    print(f"{'target':<18}{'train':>9}{'validation':>12}{'test':>9}")
    for t in targets:
        print(f"{t:<18}{report['train']['event_rates_pct'][t]:>9}{report['validation']['event_rates_pct'][t]:>12}"
              f"{report['test']['event_rates_pct'][t]:>9}")
    print("\n=== Event rates (%) by video-duration decile (1 = shortest) ===")
    print(f"{'decile':<8}" + "".join(f"{n:>15}" for n in names))
    for row in rows:
        print(f"{row['duration_decile']:<8}" + "".join(f"{row[n]:>15}" for n in names))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
