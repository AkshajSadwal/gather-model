import tempfile
import unittest
from pathlib import Path

import numpy as np

from model1_retrieval.data import load_split


class PreparedSplitValidationTests(unittest.TestCase):
    def _write_split(self, root: Path, *, numeric=None, labels=None, categorical=None,
                     catalog_video_indices=None):
        np.savez(
            root / "train.npz",
            categorical=(np.zeros((2, 36), dtype=np.int32) if categorical is None else categorical),
            numeric=(np.zeros((2, 10), dtype=np.float32) if numeric is None else numeric),
            labels=(np.zeros((2, 10), dtype=np.float32) if labels is None else labels),
            **({} if catalog_video_indices is None else {"catalog_video_indices": catalog_video_indices}),
        )

    def test_accepts_valid_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_split(root)
            split = load_split(root, "train")
            self.assertEqual(split.rows, 2)

    def test_rejects_nonfinite_numeric_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            numeric = np.zeros((2, 10), dtype=np.float32)
            numeric[0, 0] = np.nan
            self._write_split(root, numeric=numeric)
            with self.assertRaisesRegex(ValueError, "numeric features"):
                load_split(root, "train")

    def test_rejects_invalid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.zeros((2, 10), dtype=np.float32)
            labels[0, 0] = 2.0
            self._write_split(root, labels=labels)
            with self.assertRaisesRegex(ValueError, "labels"):
                load_split(root, "train")

    def test_rejects_negative_categorical_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            categorical = np.zeros((2, 36), dtype=np.int32)
            categorical[0, 0] = -1
            self._write_split(root, categorical=categorical)
            with self.assertRaisesRegex(ValueError, "categorical indices"):
                load_split(root, "train")


if __name__ == "__main__":
    unittest.main()
