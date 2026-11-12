"""Readiness guard tests use synthetic fixtures only; they are never model data."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_gather_dataset.py"

class GatherReadinessTests(unittest.TestCase):
    def test_missing_point_in_time_snapshots_prevent_training_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "one-row.jsonl"
            source.write_text(json.dumps({
                "viewerKey": "test-user", "reelKey": "test-reel", "creatorKey": "test-creator",
                "servedAt": "2026-01-01T00:00:00Z", "actions": ["LIKE"],
                "labelSchemaVersion": "gather-labels-v1", "labelsAvailable": {"is_like": True}
            }) + "\n", encoding="utf-8")
            output = root / "prepared"
            result = subprocess.run([
                sys.executable, str(SCRIPT), str(source), "--output-dir", str(output),
                "--min-users", "1", "--min-impressions", "1", "--min-positive-rows", "1",
                "--min-reels", "1", "--min-creators", "1", "--min-coverage-days", "0"
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "NOT_READY")
            self.assertFalse((output / "train.jsonl").exists())

if __name__ == "__main__":
    unittest.main()
