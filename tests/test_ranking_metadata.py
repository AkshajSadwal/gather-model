import unittest
from types import SimpleNamespace

from model2_ranking.train_v2 import split_row_metadata


class RankingSplitMetadataTests(unittest.TestCase):
    def test_skipped_test_split_uses_preparation_metadata_without_loading_arrays(self):
        result = split_row_metadata(100, 20, None, {"test_rows": 30})
        self.assertEqual(result["rows"], {"train": 100, "validation": 20, "test": 30})
        self.assertFalse(result["test_split_loaded"])

    def test_loaded_test_split_uses_actual_loaded_row_count(self):
        test_split = SimpleNamespace(n=29)
        result = split_row_metadata(100, 20, test_split, {"test_rows": 30})
        self.assertEqual(result["rows"]["test"], 29)
        self.assertTrue(result["test_split_loaded"])

    def test_missing_legacy_test_row_metadata_does_not_crash(self):
        result = split_row_metadata(100, 20, None, {})
        self.assertIsNone(result["rows"]["test"])
        self.assertFalse(result["test_split_loaded"])


if __name__ == "__main__":
    unittest.main()
