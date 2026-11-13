"""Focused tests for the frozen KuaiRand-to-Gather feature adapter."""
import json
import math
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from serving.gather_kuairand_adapter import (
    CATEGORICAL_ORDER,
    NUMERIC_ORDER,
    KuaiRandFeatureAdapter,
    parse_range_label,
    range_contains,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "ranking" / "1k_v3_catalog"
ARTIFACT_PATH = ROOT / "serving" / "artifacts" / "kuairand_v1_vocab.json"


class KuaiRandGatherAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
        cls.artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        cls.adapter = KuaiRandFeatureAdapter(cls.metadata, cls.artifact)

    def test_exact_training_feature_order(self):
        self.assertEqual(tuple(self.metadata["categorical_features"]), CATEGORICAL_ORDER)
        self.assertEqual(tuple(self.metadata["numeric_features"]), NUMERIC_ORDER)
        self.assertEqual(len(CATEGORICAL_ORDER), 36)
        self.assertEqual(len(NUMERIC_ORDER), 10)

    def test_verified_range_mapping_tag_and_id_oov(self):
        tag, tag_index = next(iter(self.artifact["tagVocabulary"].items()))
        user = {
            "schemaVersion": "gather-reco-v1",
            "numeric": {
                "hour_sin_utc": -0.25,
                "hour_cos_utc": 0.75,
                "log1p_user_following_count": math.log1p(11),
                "log1p_user_follower_count": math.log1p(10),
                "log1p_user_friend_count": math.log1p(4),
                "log1p_user_account_age_days": math.log1p(21),
            },
        }
        reel = {
            "schemaVersion": "gather-reco-v1",
            "categorical": {"video:topic": tag},
            "numeric": {
                "log1p_reel_duration_seconds": math.log1p(12),
                "log1p_reel_age_hours": math.log1p(4),
            },
        }
        categorical, numeric = self.adapter.encode(user, reel, datetime(2026, 11, 14, tzinfo=timezone.utc))
        self.assertEqual(categorical[0], 0)
        self.assertEqual(categorical[1], 0)
        self.assertEqual(categorical[2], 0)
        self.assertEqual(categorical[7], self.artifact["rangeVocabularies"]["user:follow_user_num_range"]["(10,50]"])
        self.assertEqual(categorical[8], self.artifact["rangeVocabularies"]["user:fans_user_num_range"]["[10,100)"])
        self.assertEqual(categorical[9], self.artifact["rangeVocabularies"]["user:friend_user_num_range"]["[1,5)"])
        self.assertEqual(categorical[10], self.artifact["rangeVocabularies"]["user:register_days_range"]["15-30"])
        self.assertEqual(categorical[29], 0)
        self.assertEqual(categorical[35], tag_index)
        self.assertAlmostEqual(numeric[0], (math.log1p(12) - self.metadata["numeric_mean"][0]) / self.metadata["numeric_std"][0], places=5)
        self.assertAlmostEqual(numeric[1], (-0.25 - self.metadata["numeric_mean"][1]) / self.metadata["numeric_std"][1], places=5)
        self.assertAlmostEqual(numeric[3], (math.log1p(11) - self.metadata["numeric_mean"][3]) / self.metadata["numeric_std"][3], places=5)
        self.assertAlmostEqual(numeric[8], self.artifact["numericTrainingMedianStandardized"][8], places=5)
        self.assertAlmostEqual(numeric[9], self.artifact["numericTrainingMedianStandardized"][9], places=5)

    def test_missing_invalid_and_unknown_values_use_oov_or_training_medians(self):
        user = {"numeric": {"log1p_user_friend_count": -1, "hour_sin_utc": float("nan")}}
        reel = {
            "categorical": {"video:topic": "not-a-kuairand-tag"},
            "numeric": {"log1p_reel_duration_seconds": "invalid", "log1p_reel_age_hours": float("inf")},
        }
        categorical, numeric = self.adapter.encode(user, reel, datetime(2026, 11, 14, 12, tzinfo=timezone.utc))
        self.assertEqual(categorical[9], 0)
        self.assertEqual(categorical[35], 0)
        for index in (0, 5, 7, 8, 9):
            self.assertAlmostEqual(numeric[index], self.artifact["numericTrainingMedianStandardized"][index], places=5)
        self.assertAlmostEqual(numeric[1], (math.sin(math.pi) - self.metadata["numeric_mean"][1]) / self.metadata["numeric_std"][1], places=5)
        self.assertAlmostEqual(numeric[2], (math.cos(math.pi) - self.metadata["numeric_mean"][2]) / self.metadata["numeric_std"][2], places=5)

    def test_range_label_parser_respects_boundaries(self):
        exclusive = parse_range_label("(10,50]")
        self.assertFalse(range_contains(10, exclusive))
        self.assertTrue(range_contains(50, exclusive))
        fans = parse_range_label("[10w,100w)")
        self.assertTrue(range_contains(100_000, fans))
        self.assertFalse(range_contains(1_000_000, fans))

    def test_nested_source_schema_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema version mismatch"):
            self.adapter.encode({"schemaVersion": "other"}, {})


if __name__ == "__main__":
    unittest.main()
