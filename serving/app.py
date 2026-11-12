"""Authenticated, persistent two-stage Gather inference. Missing/mismatched artifacts stay COLD_START."""
from __future__ import annotations
import hashlib, hmac, json, os, time
from pathlib import Path
from typing import Any
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Gather Recommender", version="2.0.0")
started = time.monotonic()
ROOT = Path(os.getenv("REELS_RECOMMENDER_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
DATA_DIR = Path(os.getenv("GATHER_DATA_DIR", str(ROOT / "data" / "processed" / "gather_v1"))).resolve()
ARTIFACT_DIR = Path(os.getenv("GATHER_ARTIFACT_DIR", str(ROOT / "artifacts" / "gather"))).resolve()
MANIFEST_PATH = Path(os.getenv("GATHER_MODEL_MANIFEST", str(ROOT / "configs" / "gather_recommendation_schema_v1.json"))).resolve()
RETRIEVAL_PATH = Path(os.getenv("GATHER_RETRIEVAL_CHECKPOINT", str(ARTIFACT_DIR / "retrieval" / "best_model.pt"))).resolve()
RANKING_PATH = Path(os.getenv("GATHER_RANKING_CHECKPOINT", str(ARTIFACT_DIR / "ranking" / "best_model.pt"))).resolve()
INDEX_DIR = Path(os.getenv("GATHER_RETRIEVAL_INDEX", str(ARTIFACT_DIR / "index"))).resolve()
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
manifest_payload = dict(manifest); EXPECTED_FINGERPRINT = manifest_payload.pop("fingerprint", None)
ACTUAL_FINGERPRINT = hashlib.sha256(json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
if EXPECTED_FINGERPRINT != ACTUAL_FINGERPRINT: raise RuntimeError("Gather feature schema fingerprint mismatch")
ACTIVE_MANIFEST_PATH=Path(os.getenv("GATHER_ACTIVE_MODEL_MANIFEST",str(ARTIFACT_DIR/"active_gather_model.json"))).resolve()
ACTIVE_MANIFEST_ERROR=None
ACTIVE_POINTER=None
if ACTIVE_MANIFEST_PATH.is_file():
    try:
        active_pointer=json.loads(ACTIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
        ACTIVE_POINTER=active_pointer
        if active_pointer.get("schemaFingerprint") != EXPECTED_FINGERPRINT: raise ValueError("Active model pointer schema fingerprint mismatch")
        if "GATHER_RETRIEVAL_CHECKPOINT" not in os.environ: RETRIEVAL_PATH=Path(active_pointer["retrievalCheckpoint"]).resolve()
        if "GATHER_RANKING_CHECKPOINT" not in os.environ: RANKING_PATH=Path(active_pointer["rankingCheckpoint"]).resolve()
        if "GATHER_RETRIEVAL_INDEX" not in os.environ: INDEX_DIR=Path(active_pointer["indexDirectory"]).resolve()
    except Exception as exc:
        ACTIVE_MANIFEST_ERROR=str(exc)[:240]

class Candidate(BaseModel):
    reelId: str = Field(min_length=1, max_length=64)
    creatorId: str = Field(min_length=1, max_length=64)
    reelKey: str = Field(min_length=1, max_length=64)
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
        raise HTTPException(status_code=401, detail={"code": "SERVICE_AUTH_REQUIRED", "message": "Service authentication failed."})

class InferenceRuntime:
    def __init__(self):
        self.status="COLD_START"; self.reason="Gather-trained artifacts are not installed."; self.device=None
        self.ready=False; self._loaded=False
    def load(self):
        if self._loaded: return
        self._loaded=True
        try:
            if ACTIVE_MANIFEST_ERROR: raise ValueError(ACTIVE_MANIFEST_ERROR)
            import torch
            from common.data_version import schema_fingerprint
            from model1_retrieval.model import TwoTowerRetrieval
            from model2_ranking.model_v2 import ReelsRankerV2
            for path in (DATA_DIR / "metadata.json", DATA_DIR / "feature_mappings.json", RETRIEVAL_PATH, RANKING_PATH, INDEX_DIR / "index_metadata.json"):
                if not path.is_file(): raise FileNotFoundError(f"Missing required Gather artifact: {path.name}")
            self.metadata=json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
            if self.metadata.get("status") != "READY" or self.metadata.get("schema_fingerprint") != EXPECTED_FINGERPRINT:
                raise ValueError("Prepared Gather data is not READY for the active feature schema")
            self.schema_fingerprint=schema_fingerprint(DATA_DIR)
            mappings=json.loads((DATA_DIR / "feature_mappings.json").read_text(encoding="utf-8"))
            if mappings.get("schemaFingerprint") != EXPECTED_FINGERPRINT: raise ValueError("Feature mapping fingerprint mismatch")
            self.vocab=mappings["categoricalVocabularies"]; self.index_to_reel=mappings["indexToReelId"]
            if ACTIVE_POINTER and (self.file_sha256(RETRIEVAL_PATH) != ACTIVE_POINTER.get("retrievalSha256") or self.file_sha256(RANKING_PATH) != ACTIVE_POINTER.get("rankingSha256")): raise ValueError("Active model pointer checkpoint hash mismatch")
            self.m1=torch.load(RETRIEVAL_PATH,map_location="cpu",weights_only=False)
            self.m2=torch.load(RANKING_PATH,map_location="cpu",weights_only=False)
            if self.m1.get("data_fingerprint") != self.schema_fingerprint: raise ValueError("Retrieval checkpoint schema/ID-map fingerprint mismatch")
            if self.m2.get("schema_fingerprint") != EXPECTED_FINGERPRINT or self.m2.get("dataset") != "gather_v1": raise ValueError("Ranking checkpoint schema fingerprint mismatch")
            if self.m2.get("prepared_data_fingerprint") != self.metadata.get("prepared_data_fingerprint"): raise ValueError("Ranking checkpoint prepared-data fingerprint mismatch")
            if self.m2.get("categorical_sizes") != self.metadata.get("categorical_sizes") or self.m2.get("binary_targets") != self.metadata.get("binary_targets"):
                raise ValueError("Ranking checkpoint vocabulary/target metadata mismatch")
            self.index_metadata=json.loads((INDEX_DIR / "index_metadata.json").read_text(encoding="utf-8"))
            if self.index_metadata.get("data_fingerprint") != self.schema_fingerprint: raise ValueError("Retrieval index schema fingerprint mismatch")
            catalog_map=DATA_DIR / "catalog_maps.npz"
            if self.index_metadata.get("catalog_maps_sha256") != self.file_sha256(catalog_map): raise ValueError("Retrieval index catalogue mapping fingerprint mismatch")
            if self.index_metadata.get("checkpoint_sha256") != self.file_sha256(RETRIEVAL_PATH): raise ValueError("Retrieval index was built from another checkpoint")
            files=self.index_metadata.get("files",{})
            self.vectors=np.load(INDEX_DIR / files["embeddings"],mmap_mode="r",allow_pickle=False)
            self.catalog_ids=np.load(INDEX_DIR / files["video_ids"],mmap_mode="r",allow_pickle=False)
            if self.vectors.ndim != 2 or self.vectors.shape[1] != 128 or len(self.vectors) != len(self.catalog_ids): raise ValueError("Invalid retrieval index shape")
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.user_model=TwoTowerRetrieval(self.metadata).to(self.device); self.user_model.load_state_dict(self.m1["model_state_dict"]); self.user_model.eval()
            self.rank_model=ReelsRankerV2(self.metadata["categorical_sizes"],len(self.metadata["numeric_features"]),num_targets=len(self.metadata["binary_targets"])).to(self.device)
            self.rank_model.load_state_dict(self.m2["model_state_dict"]); self.rank_model.eval()
            self.like_index=self.metadata["binary_targets"].index("is_like")
            self.status="READY"; self.reason=None; self.ready=True
        except Exception as exc:
            self.status="COLD_START"; self.reason=str(exc)[:240]; self.ready=False
    @staticmethod
    def file_sha256(path: Path) -> str:
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024),b""): digest.update(chunk)
        return digest.hexdigest()
    def encode(self, features: dict[str, Any]):
        categorical=features.get("categorical",{}); numeric=features.get("numeric",{})
        cat=[self.vocab[name].get(str(categorical.get(name) if categorical.get(name) not in (None,"") else "__MISSING__"),0) for name in self.metadata["categorical_features"]]
        raw=[]
        for i,name in enumerate(self.metadata["numeric_features"]):
            value=numeric.get(name); value=float(value) if value is not None else float(self.metadata["numeric_medians"][i])
            if not np.isfinite(value): value=float(self.metadata["numeric_medians"][i])
            raw.append(value)
        nums=(np.asarray(raw,dtype=np.float32)-np.asarray(self.metadata["numeric_mean"],dtype=np.float32))/np.asarray(self.metadata["numeric_std"],dtype=np.float32)
        return cat,nums
    def recommend(self, request: RecommendationRequest):
        import torch
        user_cat,user_num=self.encode(request.userFeatures)
        cat=torch.tensor([user_cat],dtype=torch.int32,device=self.device); num=torch.tensor([user_num],dtype=torch.float32,device=self.device)
        with torch.no_grad(): user_vec=self.user_model.user_tower(cat[:,self.user_model.USER_COLUMNS],num[:,self.user_model.USER_NUMERIC_COLUMNS]).cpu().numpy()[0]
        best=[]; k=min(max(len(request.candidates)*4,100),1000); chunk=131072
        for start in range(0,len(self.catalog_ids),chunk):
            end=min(start+chunk,len(self.catalog_ids)); ids=np.asarray(self.catalog_ids[start:end],dtype=np.int64)
            scores=np.asarray(self.vectors[start:end],dtype=np.float32) @ user_vec
            take=min(k,len(scores)); idx=np.argpartition(scores,-take)[-take:] if take else np.empty(0,dtype=int)
            best.extend((float(scores[i]),str(self.index_to_reel.get(str(int(ids[i])),""))) for i in idx)
            if len(best)>k*3: best=sorted(best,reverse=True)[:k]
        best.sort(reverse=True); retrieval={key:score for score,key in best if key}
        by_id={candidate.reelKey:candidate for candidate in request.candidates}; ordered=[(key,retrieval[key],by_id[key]) for key in retrieval if key in by_id]
        if not ordered: return {"status":"READY","modelVersion":"gather_two_stage_v1","featureSchemaVersion":manifest["schemaVersion"],"featureSchemaFingerprint":EXPECTED_FINGERPRINT,"items":[]}
        rows=[]
        for key,retrieval_score,candidate in ordered:
            c,n=self.encode(candidate.features); rows.append((candidate.reelId,retrieval_score,c,n))
        c=torch.tensor([row[2] for row in rows],dtype=torch.int32,device=self.device); n=torch.tensor(np.stack([row[3] for row in rows]),dtype=torch.float32,device=self.device)
        with torch.no_grad(): logits,watch=self.rank_model(c,n); like=torch.sigmoid(logits[:,self.like_index]).cpu().numpy(); watch=watch.clamp(min=0,max=float(np.log1p(3600))).exp().sub(1).cpu().numpy()
        ranked=sorted(({"reelId":rows[i][0],"retrievalScore":rows[i][1],"isLikeScore":float(like[i]),"rankingScore":float(like[i]),"predictedWatchSeconds":float(watch[i])} for i in range(len(rows))),key=lambda item:(item["rankingScore"],item["retrievalScore"]),reverse=True)
        return {"status":"READY","modelVersion":"gather_two_stage_v1","featureSchemaVersion":manifest["schemaVersion"],"featureSchemaFingerprint":EXPECTED_FINGERPRINT,"items":ranked[:len(request.candidates)]}

runtime=InferenceRuntime()
@app.on_event("startup")
def initialize_runtime(): runtime.load()
@app.get("/health/live")
def live(): return {"status":"LIVE","uptimeSeconds":round(time.monotonic()-started,1)}
@app.get("/health/ready")
def ready():
    runtime.load(); return {"status":runtime.status,"provider":"two_stage_v1" if runtime.ready else "cold_start_v1","featureSchemaVersion":manifest["schemaVersion"],"featureSchemaFingerprint":EXPECTED_FINGERPRINT,"device":str(runtime.device) if runtime.device else None,"reason":runtime.reason}
@app.get("/v1/model-info")
def model_info(authorization: str | None=Header(default=None)):
    require_service_token(authorization); runtime.load()
    return {"status":runtime.status,"modelVersion":"gather_two_stage_v1" if runtime.ready else None,"featureSchemaVersion":manifest["schemaVersion"],"featureSchemaFingerprint":EXPECTED_FINGERPRINT,"indexVersion":runtime.index_metadata.get("created_at_utc") if runtime.ready else None}
@app.post("/v1/recommendations")
def recommendations(request: RecommendationRequest, authorization: str | None=Header(default=None)):
    require_service_token(authorization); runtime.load()
    if request.featureSchemaVersion != manifest["schemaVersion"] or request.featureSchemaFingerprint != EXPECTED_FINGERPRINT:
        raise HTTPException(status_code=409,detail={"code":"FEATURE_SCHEMA_MISMATCH","status":runtime.status,"items":[]})
    if not runtime.ready: raise HTTPException(status_code=503,detail={"code":"GATHER_MODELS_NOT_READY","status":"COLD_START","requestId":request.requestId,"items":[]})
    return runtime.recommend(request)