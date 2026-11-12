"""Adapter tests use synthetic feature fixtures only; they are not model training data."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.prepare_gather_dataset import invalid_label_rows, write_model_arrays

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "configs" / "gather_recommendation_schema_v1.json").read_text(encoding="utf-8"))
PAYLOAD = dict(MANIFEST); PAYLOAD.pop("fingerprint")
FINGERPRINT = hashlib.sha256(json.dumps(PAYLOAD, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def row(viewer, reel, category="known", label=True):
    user_cat = {name: category for name in MANIFEST["categoricalFeatures"] if name.startswith("user:")}
    reel_cat = {name: category for name in MANIFEST["categoricalFeatures"] if name.startswith("video:")}
    user_cat.update({"user_id": viewer, "tab": "REELS"})
    reel_cat["reel_id"] = reel
    user_num = {name: 0.5 for name in MANIFEST["numericFeatures"] if name.startswith("hour_") or name.startswith("log1p_user_")}
    reel_num = {name: 0.5 for name in MANIFEST["numericFeatures"] if name.startswith("log1p_reel_") or name.startswith("log1p_creator_")}
    labels = {name: (int(label) if name == "is_like" else 0) for name in MANIFEST["ranking"]["binaryTargets"]}
    return {"viewerKey": viewer, "reelKey": reel, "creatorKey": "creator-hash", "requestId": "req", "servedAt": "2026-01-01T00:00:00Z", "userFeatures": {"categorical": user_cat, "numeric": user_num}, "reelFeatures": {"categorical": reel_cat, "numeric": reel_num}, "labels": labels, "labelsAvailable": {name: name == "is_like" for name in labels}, "watchLogSeconds": 0.7, "watchLabelAvailable": True}


class GatherAdapterTests(unittest.TestCase):
    def test_invalid_watch_labels_are_rejected(self):
        invalid = row("viewer-a", "reel-a")
        invalid["watchLogSeconds"] = -1
        self.assertEqual(invalid_label_rows([invalid], MANIFEST["ranking"]["binaryTargets"]), 1)

    def test_train_only_vocab_oov_masks_and_model_shapes(self):
        splits = {"train": [row("viewer-a", "reel-a"), row("viewer-b", "reel-b")], "validation": [row("viewer-a", "reel-c", "validation-only")], "test": [row("viewer-b", "reel-d")]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            metadata = write_model_arrays(splits, out, MANIFEST, FINGERPRINT, "source-fingerprint")
            with np.load(out / "validation.npz", allow_pickle=False) as validation:
                self.assertEqual(validation["categorical"].shape, (1, 36))
                self.assertEqual(validation["numeric"].shape, (1, 10))
                self.assertEqual(int(validation["categorical"][0, 3]), 0)
                self.assertEqual(int(validation["label_mask"][0, 1]), 1)
                self.assertEqual(int(validation["label_mask"][0, 0]), 0)
            maps = json.loads((out / "feature_mappings.json").read_text(encoding="utf-8"))
            self.assertEqual(set(maps["userIdToIndex"]), {"viewer-a", "viewer-b"})
            self.assertEqual(metadata["schema_fingerprint"], FINGERPRINT)
            self.assertEqual(len(metadata["categorical_sizes"]), 36)
            self.assertEqual(metadata["numeric_std"][0], 1.0)

if __name__ == "__main__": unittest.main()