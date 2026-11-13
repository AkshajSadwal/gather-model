"""Private, persistent KuaiRand zero-shot inference for Gather Reels."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from common.data_version import prepared_data_fingerprint, schema_fingerprint
from model1_retrieval.model import TwoTowerRetrieval
from model2_ranking.model_v2 import ReelsRankerV2
from serving.gather_kuairand_adapter import (
    ADAPTER_VERSION,
    CATEGORICAL_ORDER,
    NUMERIC_ORDER,
    SOURCE_SCHEMA_FINGERPRINT,
    SOURCE_SCHEMA_VERSION,
    KuaiRandFeatureAdapter,
)

MODE = "kuairand_zero_shot_v1"
TRAINING_DOMAIN = "KuaiRand-1K"
QUALITY_STATUS = "NOT_VALIDATED_ON_GATHER"
MODEL_SCHEMA_VERSION = "kuairand-v2"
EXPECTED_MODEL_SCHEMA_FINGERPRINT = "7bbaadb7410036832014bdf7f3a225671f84bf60c8226c2f98ae78ca5ebb67e7"
EXPECTED_PREPARED_DATA_FINGERPRINT = "a9e0107ff2818fba39e378746664cf1a1e70402af11ca0e62d85097c2da1195f"
EXPECTED_ADAPTER_FINGERPRINT = "1fc0fea6c5e75d6948183afb9469dc34bfe62d6c67bd8a1e41ad05dbdcff0bf1"
EXPECTED_RETRIEVAL_SHA256 = "6f3114de877b4c2a3770b750704bb0cae6e0a7e2441b6ac7e17ee1162946888e"
EXPECTED_RANKING_SHA256 = "a4c91097fd409145253e9f43454b80852018286e9da5c26b870a75872d47ba8e"
ROOT = Path(os.getenv("REELS_RECOMMENDER_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
DATA_DIR = Path(os.getenv("REELS_RECOMMENDER_DATA_DIR", str(ROOT / "data" / "processed" / "ranking" / "1k_v3_catalog"))).resolve()
RETRIEVAL_PATH = Path(os.getenv("REELS_RECOMMENDER_RETRIEVAL_CHECKPOINT", str(ROOT / "artifacts" / "retrieval" / "v3_linear_output_seed42_14ep" / "best_model.pt"))).resolve()
RANKING_PATH = Path(os.getenv("REELS_RECOMMENDER_RANKING_CHECKPOINT", str(ROOT / "artifacts" / "two_stage_runs_parallel" / "parallel_v3catalog_warmstart_r42_20261101_203431" / "model2_ranking" / "warmstart_v3_catalog_baseline.pt"))).resolve()
REFERENCE_INDEX_DIR = Path(os.getenv("REELS_RECOMMENDER_REFERENCE_INDEX", str(ROOT / "data" / "indexes" / "retrieval" / "1k_v3_v3_linear_output_seed42_14ep"))).resolve()
ADAPTER_VOCAB_PATH = Path(os.getenv("REELS_RECOMMENDER_ADAPTER_VOCAB", str(Path(__file__).resolve().parent / "artifacts" / "kuairand_v1_vocab.json"))).resolve()
started = time.monotonic()

app = FastAPI(title="Gather KuaiRand Zero-shot Recommender", version=MODE)


class Candidate(BaseModel):
    reelId: str = Field(min_length=1, max_length=64)
    creatorId: str = Field(min_length=1, max_length=64)
    reelKey: str = Field(min_length=1, max_length=64)
    candidateSource: str = Field(default="EXPLORE", max_length=32)
    features: dict[str, Any] = Field(default_factory=dict)


class RecommendationRequest(BaseModel):
    requestId: str = Field(min_length=1, max_length=128)
    sessionId: str = Field(min_length=1, max_length=128)
    userKey: str = Field(min_length=1, max_length=64)
    featureSchemaVersion: str = Field(min_length=1, max_length=64)
    featureSchemaFingerprint: str = Field(min_length=64, max_length=64)
    userFeatures: dict[str, Any] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(max_length=300)


def require_service_token(value: str | None) -> None:
    expected = os.getenv("REELS_RECOMMENDER_TOKEN", "")
    if not expected or not value or not hmac.compare_digest(value, f"Bearer {expected}"):
        raise HTTPException(
            status_code=401,
            detail={"code": "SERVICE_AUTH_REQUIRED", "message": "Service authentication failed."},
        )


class InferenceRuntime:
    def __init__(self):
        self.status = "COLD_START"
        self.reason = "Selected KuaiRand artifacts have not been validated."
        self.device: torch.device | None = None
        self.ready = False
        self._loaded = False
        self.startup_load_ms: float | None = None
        self.metadata: dict[str, Any] = {}
        self.adapter: KuaiRandFeatureAdapter | None = None
        self.retrieval_model = None
        self.rank_model = None
        self.retrieval_sha256: str | None = None
        self.ranking_sha256: str | None = None
        self.reference_index_verified = False
        self.like_index: int | None = None

    @staticmethod
    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_reference_index(self, data_fingerprint: str) -> None:
        index_meta_path = REFERENCE_INDEX_DIR / "index_metadata.json"
        if not index_meta_path.is_file():
            raise FileNotFoundError("Selected Model 1 reference index metadata is missing.")
        index = json.loads(index_meta_path.read_text(encoding="utf-8"))
        catalog_maps = DATA_DIR / "catalog_maps.npz"
        if index.get("data_fingerprint") != data_fingerprint:
            raise ValueError("Reference index dataset fingerprint mismatch.")
        if index.get("checkpoint_sha256") != self.retrieval_sha256:
            raise ValueError("Reference index belongs to another Model 1 checkpoint.")
        if index.get("catalog_maps_sha256") != self.file_sha256(catalog_maps):
            raise ValueError("Reference index catalogue-map fingerprint mismatch.")
        if index.get("embedding_dim") != 128 or index.get("dtype") != "float16":
            raise ValueError("Reference index dimensions or dtype mismatch.")
        if index.get("normalization") != "L2-normalized by video tower":
            raise ValueError("Reference index normalization mismatch.")
        files = index.get("files", {})
        for name in (files.get("embeddings"), files.get("video_ids")):
            if not isinstance(name, str) or Path(name).name != name or not (REFERENCE_INDEX_DIR / name).is_file():
                raise ValueError("Reference index file metadata is invalid.")
        self.reference_index_verified = True
        self.index_metadata = {
            "dataFingerprint": index["data_fingerprint"],
            "checkpointSha256": index["checkpoint_sha256"],
            "catalogMapsSha256": index["catalog_maps_sha256"],
            "embeddingDim": index["embedding_dim"],
            "dtype": index["dtype"],
        }

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        load_started = time.perf_counter()
        try:
            for path in (DATA_DIR / "metadata.json", DATA_DIR / "id_maps.npz", DATA_DIR / "catalog_maps.npz",
                         DATA_DIR / "train.npz", DATA_DIR / "validation.npz", DATA_DIR / "test.npz",
                         RETRIEVAL_PATH, RANKING_PATH, ADAPTER_VOCAB_PATH):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing selected zero-shot artifact: {path.name}")

            self.metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
            if tuple(self.metadata.get("categorical_features", ())) != CATEGORICAL_ORDER:
                raise ValueError("Prepared KuaiRand categorical schema/order mismatch.")
            if tuple(self.metadata.get("numeric_features", ())) != NUMERIC_ORDER:
                raise ValueError("Prepared KuaiRand numeric schema/order mismatch.")

            data_fingerprint = schema_fingerprint(DATA_DIR)
            if data_fingerprint != EXPECTED_MODEL_SCHEMA_FINGERPRINT:
                raise ValueError("Prepared KuaiRand schema/ID-map fingerprint mismatch.")
            prepared_fingerprint = prepared_data_fingerprint(DATA_DIR)
            if prepared_fingerprint != EXPECTED_PREPARED_DATA_FINGERPRINT:
                raise ValueError("Prepared KuaiRand dataset fingerprint mismatch.")

            vocab_artifact = json.loads(ADAPTER_VOCAB_PATH.read_text(encoding="utf-8"))
            if vocab_artifact.get("fingerprint") != EXPECTED_ADAPTER_FINGERPRINT:
                raise ValueError("Pinned KuaiRand adapter fingerprint mismatch.")
            if vocab_artifact.get("schemaFingerprint") != data_fingerprint:
                raise ValueError("Adapter schema fingerprint mismatch.")
            if vocab_artifact.get("preparedDataFingerprint") != prepared_fingerprint:
                raise ValueError("Adapter prepared-data fingerprint mismatch.")
            self.adapter = KuaiRandFeatureAdapter(self.metadata, vocab_artifact)

            self.retrieval_sha256 = self.file_sha256(RETRIEVAL_PATH)
            self.ranking_sha256 = self.file_sha256(RANKING_PATH)
            if self.retrieval_sha256 != EXPECTED_RETRIEVAL_SHA256:
                raise ValueError("Selected Model 1 checkpoint SHA-256 mismatch.")
            if self.ranking_sha256 != EXPECTED_RANKING_SHA256:
                raise ValueError("Selected Model 2 checkpoint SHA-256 mismatch.")
            self._validate_reference_index(data_fingerprint)

            retrieval_checkpoint = torch.load(RETRIEVAL_PATH, map_location="cpu", weights_only=True)
            ranking_checkpoint = torch.load(RANKING_PATH, map_location="cpu", weights_only=True)
            if retrieval_checkpoint.get("data_fingerprint") != data_fingerprint:
                raise ValueError("Model 1 checkpoint schema fingerprint mismatch.")
            if retrieval_checkpoint.get("prepared_data_fingerprint") != prepared_fingerprint:
                raise ValueError("Model 1 checkpoint prepared-data fingerprint mismatch.")
            if retrieval_checkpoint.get("categorical_sizes") != self.metadata.get("categorical_sizes"):
                raise ValueError("Model 1 checkpoint vocabulary sizes mismatch.")
            if ranking_checkpoint.get("schema_fingerprint") != data_fingerprint:
                raise ValueError("Model 2 checkpoint schema fingerprint mismatch.")
            if ranking_checkpoint.get("prepared_data_fingerprint") != prepared_fingerprint:
                raise ValueError("Model 2 checkpoint prepared-data fingerprint mismatch.")
            if ranking_checkpoint.get("dataset") != "1k":
                raise ValueError("Model 2 checkpoint training domain mismatch.")
            if ranking_checkpoint.get("categorical_sizes") != self.metadata.get("categorical_sizes"):
                raise ValueError("Model 2 checkpoint vocabulary sizes mismatch.")
            if ranking_checkpoint.get("binary_targets") != self.metadata.get("binary_targets"):
                raise ValueError("Model 2 checkpoint target order mismatch.")
            if ranking_checkpoint.get("num_numeric_features") != len(NUMERIC_ORDER):
                raise ValueError("Model 2 checkpoint numeric feature count mismatch.")

            device_choice = os.getenv("REELS_RECOMMENDER_DEVICE", "auto").strip().lower()
            if device_choice not in {"auto", "cpu", "cuda"}:
                raise ValueError("REELS_RECOMMENDER_DEVICE must be auto, cpu, or cuda.")
            if device_choice == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable.")
            self.device = torch.device("cuda" if device_choice == "cuda" or
                                       (device_choice == "auto" and torch.cuda.is_available()) else "cpu")
            if self.device.type == "cpu":
                cpu_threads = max(1, min(8, int(os.getenv("REELS_RECOMMENDER_CPU_THREADS", "2"))))
                torch.set_num_threads(cpu_threads)

            self.retrieval_model = TwoTowerRetrieval(
                self.metadata, linear_output=bool(retrieval_checkpoint.get("linear_output", False))
            ).to(self.device)
            self.retrieval_model.load_state_dict(retrieval_checkpoint["model_state_dict"], strict=True)
            self.retrieval_model.eval().requires_grad_(False)
            self.rank_model = ReelsRankerV2(
                self.metadata["categorical_sizes"], len(NUMERIC_ORDER),
                num_targets=len(self.metadata["binary_targets"]),
                embedding_dim=int(ranking_checkpoint["embedding_dim"]),
            ).to(self.device)
            self.rank_model.load_state_dict(ranking_checkpoint["model_state_dict"], strict=True)
            self.rank_model.eval().requires_grad_(False)
            self.like_index = self.metadata["binary_targets"].index("is_like")
            self.status = "READY"
            self.reason = None
            self.ready = True
        except Exception as exc:
            self.status = "COLD_START"
            self.reason = f"{type(exc).__name__}: {str(exc)[:180]}"
            self.ready = False
        finally:
            self.startup_load_ms = round((time.perf_counter() - load_started) * 1000, 2)

    def recommend(self, request: RecommendationRequest) -> dict[str, Any]:
        if not self.ready or self.adapter is None or self.device is None:
            raise RuntimeError("Zero-shot runtime is not ready.")
        started_at = time.perf_counter()
        candidate_ids = [candidate.reelId for candidate in request.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Candidate Reel IDs must be unique.")

        now = datetime.now(timezone.utc)
        user_cat, user_num = self.adapter.encode(request.userFeatures, {}, now=now)
        encoded_candidates = [
            self.adapter.encode(request.userFeatures, candidate.features, now=now)
            for candidate in request.candidates
        ]
        if not encoded_candidates:
            return self._response([], 0.0, 0)

        categorical = np.stack([item[0] for item in encoded_candidates])
        numeric = np.stack([item[1] for item in encoded_candidates])
        user_cat_tensor = torch.as_tensor(user_cat[None, :], dtype=torch.int64, device=self.device)
        user_num_tensor = torch.as_tensor(user_num[None, :], dtype=torch.float32, device=self.device)

        retrieval_scores: list[float] = []
        like_scores: list[float] = []
        watch_seconds: list[float] = []
        batch_size = 64
        with torch.inference_mode():
            user_vector = self.retrieval_model.encode_user(user_cat_tensor, user_num_tensor)
            for offset in range(0, len(request.candidates), batch_size):
                end = min(offset + batch_size, len(request.candidates))
                cat_batch = torch.as_tensor(categorical[offset:end], dtype=torch.int64, device=self.device)
                num_batch = torch.as_tensor(numeric[offset:end], dtype=torch.float32, device=self.device)
                reel_vectors = self.retrieval_model.encode_video(cat_batch, num_batch)
                similarity = torch.sum(reel_vectors * user_vector, dim=1)
                logits, watch_log_seconds = self.rank_model(cat_batch, num_batch)
                is_like = torch.sigmoid(logits[:, self.like_index])
                watch_log_seconds = watch_log_seconds.clamp(min=0, max=math.log1p(3600))
                watch = torch.expm1(watch_log_seconds).clamp(min=0, max=3600)
                retrieval_scores.extend(float(value) for value in similarity.cpu().tolist())
                like_scores.extend(float(value) for value in is_like.cpu().tolist())
                watch_seconds.extend(float(value) for value in watch.cpu().tolist())

        ranked = []
        for index, candidate in enumerate(request.candidates):
            retrieval_score = retrieval_scores[index]
            like_score = like_scores[index]
            if not math.isfinite(retrieval_score) or not math.isfinite(like_score):
                continue
            ranked.append({
                "reelId": candidate.reelId,
                "retrievalScore": retrieval_score,
                "isLikeScore": like_score,
                "rankingScore": like_score,
                "predictedWatchSeconds": watch_seconds[index],
                "candidateSource": candidate.candidateSource,
                "_inputOrder": index,
            })
        ranked.sort(key=lambda item: (-item["rankingScore"], -item["retrievalScore"], item["_inputOrder"]))
        for item in ranked:
            item.pop("_inputOrder", None)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return self._response(ranked, elapsed_ms, len(request.candidates))

    def _response(self, items: list[dict[str, Any]], inference_ms: float, candidate_count: int) -> dict[str, Any]:
        return {
            "status": "READY",
            "mode": MODE,
            "trainingDomain": TRAINING_DOMAIN,
            "recommendationQuality": QUALITY_STATUS,
            "modelVersion": MODE,
            "featureSchemaVersion": SOURCE_SCHEMA_VERSION,
            "featureSchemaFingerprint": SOURCE_SCHEMA_FINGERPRINT,
            "modelSchemaVersion": MODEL_SCHEMA_VERSION,
            "modelSchemaFingerprint": EXPECTED_MODEL_SCHEMA_FINGERPRINT,
            "preparedDataFingerprint": EXPECTED_PREPARED_DATA_FINGERPRINT,
            "inferenceMs": inference_ms,
            "candidateCount": candidate_count,
            "items": items,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": MODE,
            "provider": MODE if self.ready else "cold_start_v1",
            "trainingDomain": TRAINING_DOMAIN,
            "recommendationQuality": QUALITY_STATUS,
            "featureSchemaVersion": SOURCE_SCHEMA_VERSION,
            "featureSchemaFingerprint": SOURCE_SCHEMA_FINGERPRINT,
            "modelSchemaVersion": MODEL_SCHEMA_VERSION,
            "modelSchemaFingerprint": EXPECTED_MODEL_SCHEMA_FINGERPRINT,
            "device": str(self.device) if self.device else None,
            "startupLoadMs": self.startup_load_ms,
            "referenceIndexVerified": self.reference_index_verified,
            "reason": self.reason,
        }


runtime = InferenceRuntime()


@app.on_event("startup")
def initialize_runtime() -> None:
    runtime.load()


@app.get("/health/live")
def live() -> dict[str, Any]:
    return {"status": "LIVE", "uptimeSeconds": round(time.monotonic() - started, 1)}


@app.get("/health/ready")
def ready() -> dict[str, Any]:
    runtime.load()
    return runtime.health()


@app.get("/v1/model-info")
def model_info(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_service_token(authorization)
    runtime.load()
    info = runtime.health()
    info.update({
        "modelVersion": MODE if runtime.ready else None,
        "model1Checkpoint": "v3_linear_output_seed42_14ep/best_model.pt",
        "model1CheckpointSha256": runtime.retrieval_sha256,
        "model2Checkpoint": "parallel_v3catalog_warmstart_r42_20261101_203431/model2_ranking/warmstart_v3_catalog_baseline.pt",
        "model2CheckpointSha256": runtime.ranking_sha256,
        "adapterVersion": ADAPTER_VERSION,
        "adapterFingerprint": runtime.adapter.artifact["fingerprint"] if runtime.adapter else None,
        "mappingReport": runtime.adapter.mapping_report if runtime.adapter else None,
    })
    return info


@app.post("/v1/recommendations")
def recommendations(
    request: RecommendationRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require_service_token(authorization)
    if request.featureSchemaVersion != SOURCE_SCHEMA_VERSION or request.featureSchemaFingerprint != SOURCE_SCHEMA_FINGERPRINT:
        raise HTTPException(
            status_code=409,
            detail={"code": "FEATURE_SCHEMA_MISMATCH", "mode": MODE, "items": []},
        )
    runtime.load()
    if not runtime.ready:
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_NOT_READY", "status": "COLD_START", "requestId": request.requestId, "items": []},
        )
    try:
        return runtime.recommend(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_RECOMMENDATION_INPUT", "message": str(exc)}) from exc
