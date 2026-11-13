"""Serving contract and safety tests; no database or model training is required."""
import ast
import json
import math
import os
import unittest
from unittest.mock import patch

import torch
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from serving.app import (
    EXPECTED_MODEL_SCHEMA_FINGERPRINT,
    EXPECTED_PREPARED_DATA_FINGERPRINT,
    EXPECTED_ADAPTER_FINGERPRINT,
    EXPECTED_RANKING_SHA256,
    EXPECTED_RETRIEVAL_SHA256,
    MODE,
    Candidate,
    InferenceRuntime,
    RecommendationRequest,
    recommendations,
    require_service_token,
    runtime,
)
from scripts.promote_gather_model import eligible
from serving.gather_kuairand_adapter import KuaiRandFeatureAdapter, SOURCE_SCHEMA_FINGERPRINT

ROOT = Path(__file__).resolve().parents[1]


class GatherServingTests(unittest.TestCase):
    def test_selected_frozen_model_and_cold_start_paths_are_explicit(self):
        source = (ROOT / "serving" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn('self.status = "COLD_START"', source)
        self.assertIn('if not runtime.ready', source)
        self.assertIn('MODE = "kuairand_zero_shot_v1"', source)
        self.assertIn("encode_video", source)
        self.assertIn("prepared_data_fingerprint", source)
        self.assertNotIn("self.catalog_ids", source)
        self.assertTrue(any(isinstance(node, ast.ClassDef) and node.name == "InferenceRuntime" for node in tree.body))
        self.assertEqual(MODE, "kuairand_zero_shot_v1")
        self.assertEqual(len(EXPECTED_MODEL_SCHEMA_FINGERPRINT), 64)
        self.assertEqual(len(EXPECTED_PREPARED_DATA_FINGERPRINT), 64)
        self.assertEqual(len(EXPECTED_ADAPTER_FINGERPRINT), 64)
        self.assertEqual(len(EXPECTED_RETRIEVAL_SHA256), 64)
        self.assertEqual(len(EXPECTED_RANKING_SHA256), 64)

    def test_service_token_is_required(self):
        with patch.dict(os.environ, {"REELS_RECOMMENDER_TOKEN": "test-token"}):
            require_service_token("Bearer test-token")
            with self.assertRaises(HTTPException) as error:
                require_service_token("Bearer wrong")
        self.assertEqual(error.exception.status_code, 401)

    def test_source_schema_mismatch_is_rejected_before_model_loading(self):
        request = RecommendationRequest(
            requestId="request-1",
            sessionId="session-1",
            userKey="opaque-user",
            featureSchemaVersion="gather-reco-v1",
            featureSchemaFingerprint="0" * 64,
            userFeatures={},
            candidates=[],
        )
        with patch.dict(os.environ, {"REELS_RECOMMENDER_TOKEN": "test-token"}):
            with self.assertRaises(HTTPException) as error:
                recommendations(request, "Bearer test-token")
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(error.exception.detail["code"], "FEATURE_SCHEMA_MISMATCH")

    def test_cold_start_returns_service_unavailable_for_node_fallback(self):
        request = RecommendationRequest(
            requestId="request-cold",
            sessionId="session-cold",
            userKey="opaque-user",
            featureSchemaVersion="gather-reco-v1",
            featureSchemaFingerprint=SOURCE_SCHEMA_FINGERPRINT,
            userFeatures={},
            candidates=[],
        )
        with patch.dict(os.environ, {"REELS_RECOMMENDER_TOKEN": "test-token"}):
            with patch.object(runtime, "load", lambda: None):
                with self.assertRaises(HTTPException) as error:
                    recommendations(request, "Bearer test-token")
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(error.exception.detail["status"], "COLD_START")

    def test_model1_similarity_and_model2_is_like_rank_keep_real_gather_ids(self):
        metadata = json.loads((ROOT / "data" / "processed" / "ranking" / "1k_v3_catalog" / "metadata.json").read_text(encoding="utf-8"))
        artifact = json.loads((ROOT / "serving" / "artifacts" / "kuairand_v1_vocab.json").read_text(encoding="utf-8"))
        adapter = KuaiRandFeatureAdapter(metadata, artifact)
        matched_tag = next(iter(artifact["tagVocabulary"]))

        class FakeRetrieval:
            def encode_user(self, categorical, numeric):
                return torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=categorical.device)

            def encode_video(self, categorical, numeric):
                known = (categorical[:, 35] > 0).to(dtype=torch.float32)
                return torch.stack((known, 1.0 - known), dim=1)

        class FakeRanker(torch.nn.Module):
            def __init__(self, like_index):
                super().__init__()
                self.like_index = like_index

            def forward(self, categorical, numeric):
                logits = torch.zeros((categorical.shape[0], len(metadata["binary_targets"])), device=categorical.device)
                logits[:, self.like_index] = numeric[:, 0]
                watch = torch.full((categorical.shape[0],), math.log1p(7.0), device=categorical.device)
                return logits, watch

        test_runtime = InferenceRuntime()
        test_runtime.ready = True
        test_runtime.device = torch.device("cpu")
        test_runtime.adapter = adapter
        test_runtime.retrieval_model = FakeRetrieval()
        test_runtime.like_index = metadata["binary_targets"].index("is_like")
        test_runtime.rank_model = FakeRanker(test_runtime.like_index)
        user = {
            "schemaVersion": "gather-reco-v1",
            "numeric": {
                "hour_sin_utc": -0.5, "hour_cos_utc": 0.8660254,
                "log1p_user_following_count": math.log1p(12),
                "log1p_user_follower_count": math.log1p(4),
                "log1p_user_friend_count": math.log1p(2),
                "log1p_user_account_age_days": math.log1p(120),
            },
        }
        candidates = [
            Candidate(
                reelId="gather-reel-known-short", creatorId="creator-hash", reelKey="opaque-reel-a",
                candidateSource="AFFINITY",
                features={"schemaVersion": "gather-reco-v1", "categorical": {"video:topic": matched_tag},
                          "numeric": {"log1p_reel_duration_seconds": math.log1p(1), "log1p_reel_age_hours": math.log1p(1)}},
            ),
            Candidate(
                reelId="gather-reel-new-long", creatorId="creator-hash", reelKey="opaque-reel-b",
                candidateSource="FRESH",
                features={"schemaVersion": "gather-reco-v1", "categorical": {"video:topic": "unseen-gather-topic"},
                          "numeric": {"log1p_reel_duration_seconds": math.log1p(20), "log1p_reel_age_hours": math.log1p(1)}},
            ),
        ]
        request = RecommendationRequest(
            requestId="request-ranking", sessionId="session-ranking", userKey="opaque-user",
            featureSchemaVersion="gather-reco-v1", featureSchemaFingerprint=SOURCE_SCHEMA_FINGERPRINT,
            userFeatures=user, candidates=candidates,
        )
        result = test_runtime.recommend(request)
        self.assertEqual([item["reelId"] for item in result["items"]], ["gather-reel-new-long", "gather-reel-known-short"])
        self.assertLess(result["items"][0]["retrievalScore"], result["items"][1]["retrievalScore"])
        self.assertGreater(result["items"][0]["isLikeScore"], result["items"][1]["isLikeScore"])
        self.assertEqual(result["items"][0]["candidateSource"], "FRESH")
        self.assertAlmostEqual(result["items"][0]["predictedWatchSeconds"], 7.0, places=5)

    def test_promotion_requires_matching_artifact_hashes_and_metrics(self):
        report = {
            "status": "READY",
            "ranking": {"is_like": {"auc": 0.71}},
            "retrieval": {"recall_at_100": 0.18},
            "artifacts": {"retrieval_sha256": "r", "ranking_sha256": "m"},
        }
        self.assertTrue(eligible(report, "r", "m", 0.65, 0.1))
        self.assertFalse(eligible(report, "wrong", "m", 0.65, 0.1))
        self.assertFalse(eligible(report, "r", "m", 0.75, 0.1))
        self.assertFalse(eligible(report, "r", "m", 0.65, 0.2))


if __name__ == "__main__":
    unittest.main()
