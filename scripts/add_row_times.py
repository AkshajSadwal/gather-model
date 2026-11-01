"""Create row_times.npz: the timestamp of every row in train/validation/test.

The prepared arrays do not store timestamps, but pools and time-local batches need them.
This script re-reads the raw logs, applies exactly the sort and split used by
scripts/prepare_data_v2.py, and then CHECKS that the user and video of every row match the
prepared arrays. If anything differs it stops without writing a file.

    python -m scripts.add_row_times --raw-dir data/raw/kuairand_1k --data-dir data/processed/ranking/1k_v3_catalog

It adds one new file and changes nothing else (model/index fingerprints are unaffected).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_data_v2 import ids_to_index, read_log

SPLIT_NAMES = ("train", "validation", "test")


def split_logs(older: pd.DataFrame, newer: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The same chronological split as prepare_data_v2.main (frames must be time-sorted)."""
    n = len(newer)
    train_cut, val_cut = int(n * .80), int(n * .90)
    return {
        "train": pd.concat([older, newer.iloc[:train_cut]], ignore_index=True),
        "validation": newer.iloc[train_cut:val_cut],
        "test": newer.iloc[val_cut:],
    }


def check_alignment(frames: dict[str, pd.DataFrame], data_dir: Path) -> None:
    """Raise if the re-created splits are not row-for-row the prepared splits."""
    with np.load(data_dir / "id_maps.npz", allow_pickle=False) as maps:
        user_ids = np.asarray(maps["user_ids"], dtype=np.int64)
        video_ids = np.asarray(maps["video_ids"], dtype=np.int64)
    for name in SPLIT_NAMES:
        with np.load(data_dir / f"{name}.npz", allow_pickle=False) as payload:
            categorical = np.asarray(payload["categorical"])
        frame = frames[name]
        if len(frame) != categorical.shape[0]:
            raise ValueError(f"{name}: raw logs give {len(frame):,} rows but the prepared split has "
                             f"{categorical.shape[0]:,}. Use the same raw files that prepare_data_v2 used.")
        if not np.array_equal(ids_to_index(frame["user_id"].to_numpy(), user_ids), categorical[:, 0]):
            raise ValueError(f"{name}: user IDs differ from the prepared split; row order does not match")
        if not np.array_equal(ids_to_index(frame["video_id"].to_numpy(), video_ids), categorical[:, 1]):
            raise ValueError(f"{name}: video IDs differ from the prepared split; row order does not match")
        print(f"  {name}: {len(frame):,} rows match the prepared arrays")


def build_row_times(raw_dir: Path, dataset: str) -> dict[str, pd.DataFrame]:
    suffix = "pure" if dataset == "pure" else "1k"
    older = read_log(raw_dir / f"log_standard_4_08_to_4_21_{suffix}.csv")
    newer = read_log(raw_dir / f"log_standard_4_22_to_5_08_{suffix}.csv")
    return split_logs(older, newer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="Folder with the KuaiRand CSV files used for preparation")
    parser.add_argument("--data-dir", required=True, help="Prepared dataset folder, e.g. data/processed/ranking/1k_v3_catalog")
    parser.add_argument("--dataset", choices=["pure", "1k"], default="1k")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    target = data_dir / "row_times.npz"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"{target} already exists; pass --overwrite to recreate it")

    print("Reading raw logs (same sort and split as prepare_data_v2)...")
    frames = build_row_times(Path(args.raw_dir), args.dataset)
    print("Checking row-for-row alignment with the prepared arrays...")
    check_alignment(frames, data_dir)
    times = {name: frames[name]["time_ms"].to_numpy(dtype=np.int64) for name in SPLIT_NAMES}
    for name, values in times.items():
        if values.size and np.any(np.diff(values) < 0):
            print(f"  note: {name} is not globally time-sorted (older log followed by newer log is expected for train)")
    np.savez_compressed(target, **times)
    hours = {n: (v.max() - v.min()) / 3.6e6 for n, v in times.items() if v.size}
    print(f"Saved {target}  (time span in hours: " + ", ".join(f"{n} {h:.1f}" for n, h in hours.items()) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
