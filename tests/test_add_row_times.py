"""Tests for recreating row timestamps and refusing misaligned data."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from common.pools import load_row_times
from scripts.add_row_times import SPLIT_NAMES, check_alignment, split_logs
from scripts.prepare_data_v2 import ids_to_index

RAW = ["user_id", "video_id", "time_ms", "tab", "duration_ms", "play_time_ms", "is_click", "is_like",
       "is_follow", "is_comment", "is_forward", "is_hate", "long_view", "is_profile_enter"]


def frame(times, users, videos):
    n = len(times)
    data = {c: np.zeros(n) for c in RAW}
    data.update(user_id=users, video_id=videos, time_ms=times)
    return pd.DataFrame(data).sort_values("time_ms", kind="mergesort").reset_index(drop=True)


class RowTimeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.older = frame(np.sort(rng.integers(0, 1000, 40)), rng.integers(1, 6, 40) * 10, rng.integers(1, 30, 40))
        self.newer = frame(np.sort(rng.integers(1000, 2000, 50)), rng.integers(1, 6, 50) * 10, rng.integers(1, 30, 50))

    def test_split_sizes_and_order_follow_prepare_data(self):
        parts = split_logs(self.older, self.newer)
        self.assertEqual([len(parts[n]) for n in SPLIT_NAMES], [40 + 40, 5, 5])
        self.assertLessEqual(parts["train"]["time_ms"].iloc[-1], parts["validation"]["time_ms"].iloc[0])
        self.assertLessEqual(parts["validation"]["time_ms"].iloc[-1], parts["test"]["time_ms"].iloc[0])

    def _write_prepared(self, directory: Path, parts, corrupt_split=None):
        users = np.sort(pd.concat([parts["train"]["user_id"]]).unique().astype(np.int64))
        videos = np.sort(parts["train"]["video_id"].unique().astype(np.int64))
        np.savez(directory / "id_maps.npz", user_ids=users, video_ids=videos)
        for name in SPLIT_NAMES:
            frame_ = parts[name]
            cat = np.zeros((len(frame_), 36), dtype=np.int32)
            cat[:, 0] = ids_to_index(frame_["user_id"].to_numpy(), users)
            cat[:, 1] = ids_to_index(frame_["video_id"].to_numpy(), videos)
            if name == corrupt_split:
                cat[[0, 1], 1] = cat[[1, 0], 1] + 1   # scramble two rows
            np.savez(directory / f"{name}.npz", categorical=cat)

    def test_alignment_check_accepts_matching_and_rejects_scrambled_rows(self):
        parts = split_logs(self.older, self.newer)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_prepared(d, parts)
            check_alignment(parts, d)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_prepared(d, parts, corrupt_split="validation")
            with self.assertRaises(ValueError):
                check_alignment(parts, d)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_prepared(d, parts)
            shorter = {**parts, "test": parts["test"].iloc[:-1]}
            with self.assertRaises(ValueError):
                check_alignment(shorter, d)

    def test_loader_reads_saved_times_and_explains_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            with self.assertRaises(FileNotFoundError) as caught:
                load_row_times(d, "train")
            self.assertIn("add_row_times", str(caught.exception))
            np.savez(d / "row_times.npz", train=np.array([1, 2, 3]), validation=np.array([4]), test=np.array([5]))
            np.testing.assert_array_equal(load_row_times(d, "validation"), [4])
            self.assertEqual(load_row_times(d, "train").dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
