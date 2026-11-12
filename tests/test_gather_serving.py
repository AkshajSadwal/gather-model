"""Serving/promotion tests run without Gather model artifacts or a database."""
import ast
import unittest
from pathlib import Path
from scripts.promote_gather_model import eligible

ROOT=Path(__file__).resolve().parents[1]
class GatherServingTests(unittest.TestCase):
    def test_missing_artifacts_have_cold_start_readiness_path(self):
        tree=ast.parse((ROOT/"serving"/"app.py").read_text(encoding="utf-8"))
        source=(ROOT/"serving"/"app.py").read_text(encoding="utf-8")
        self.assertIn('self.status="COLD_START"',source)
        self.assertIn('if not runtime.ready',source)
        self.assertTrue(any(isinstance(node,ast.ClassDef) and node.name=="InferenceRuntime" for node in tree.body))

    def test_promotion_requires_matching_artifact_hashes_and_metrics(self):
        report={"status":"READY","ranking":{"is_like":{"auc":0.71}},"retrieval":{"recall_at_100":0.18},"artifacts":{"retrieval_sha256":"r","ranking_sha256":"m"}}
        self.assertTrue(eligible(report,"r","m",0.65,0.1))
        self.assertFalse(eligible(report,"wrong","m",0.65,0.1))
        self.assertFalse(eligible(report,"r","m",0.75,0.1))
        self.assertFalse(eligible(report,"r","m",0.65,0.2))

if __name__ == "__main__": unittest.main()