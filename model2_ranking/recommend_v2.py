"""Rank KuaiRand video candidates using the metadata-enriched V2 model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common.data_version import schema_fingerprint
from model2_ranking.model_v2 import ReelsRankerV2

SCORE_WEIGHTS={"is_click":0.15,"is_like":0.60,"is_follow":0.75,"is_comment":0.65,
               "is_forward":1.00,"long_view":1.00,"is_profile_enter":0.25,
               "complete_view":0.40,"short_view":-0.90,"is_hate":-2.00}

def find_index(sorted_values, value):
    idx=int(np.searchsorted(sorted_values,value))
    return idx+1 if idx<len(sorted_values) and int(sorted_values[idx])==int(value) else 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",default="data/processed/ranking/1k_v2")
    p.add_argument("--model-path",default="artifacts/ranking/1k_v2_m4/best_model.pt")
    p.add_argument("--user-id",required=True,type=int)
    p.add_argument("--tab",type=int,default=0)
    p.add_argument("--top-k",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=4096)
    p.add_argument("--candidate-file",default=None,help="Optional CSV of raw video_id candidates from Model 1 retrieval; omit to score the full training catalogue")
    p.add_argument("--output",default="")
    args=p.parse_args(); data_dir=Path(args.data_dir)
    metadata=json.loads((data_dir/"metadata.json").read_text(encoding="utf-8"))
    maps=np.load(data_dir/"id_maps.npz",allow_pickle=False)
    user_ids=maps["user_ids"]; video_ids=maps["video_ids"]; tab_ids=maps["tab_ids"]
    user_cat=maps["user_cat_by_index"]; video_cat=maps["video_cat_by_index"]; user_num=maps["user_numeric_by_index"]
    dur=maps["video_duration_seconds_by_index"]; width=maps["video_width_by_index"]; height=maps["video_height_by_index"]; upload=maps["video_upload_ms_by_index"]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Inference device: {device}")
    checkpoint=torch.load(args.model_path,map_location=device,weights_only=False)
    if checkpoint.get("dataset") != metadata.get("dataset"):
        raise ValueError("Ranking checkpoint dataset does not match the selected prepared dataset")
    checkpoint_schema = checkpoint.get("schema_fingerprint")
    if checkpoint_schema is not None and checkpoint_schema != schema_fingerprint(data_dir):
        raise ValueError("Ranking checkpoint feature/ID mappings differ from the selected prepared dataset")
    if checkpoint.get("binary_targets") != metadata.get("binary_targets"):
        raise ValueError("Ranking checkpoint prediction targets do not match the selected prepared dataset")
    model=ReelsRankerV2(checkpoint["categorical_sizes"],checkpoint["num_numeric_features"],
                        num_targets=len(checkpoint["binary_targets"]),embedding_dim=checkpoint["embedding_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    ui=find_index(user_ids,args.user_id); ti=find_index(tab_ids,args.tab)
    if ui==0: print(f"Warning: user {args.user_id} not in training IDs; using OOV user and metadata features.")
    if ti==0: print(f"Warning: tab {args.tab} not in training IDs; using OOV tab.")
    ref=int(metadata["reference_time_ms"]); hours=(ref//3_600_000)%24; angle=2*np.pi*hours/24.0
    sin_h,cos_h=float(np.sin(angle)),float(np.cos(angle))
    means=np.asarray(metadata["numeric_mean"],dtype=np.float32); stds=np.asarray(metadata["numeric_std"],dtype=np.float32)
    if args.candidate_file:
        candidate_path = Path(args.candidate_file)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate CSV not found: {candidate_path}")
        candidate_frame = pd.read_csv(candidate_path, usecols=["video_id"])
        candidate_raw = pd.to_numeric(candidate_frame["video_id"], errors="coerce").dropna().astype(np.int64).unique()
        positions = np.searchsorted(video_ids, candidate_raw)
        valid = positions < len(video_ids)
        matching = np.zeros(len(positions), dtype=bool)
        matching[valid] = video_ids[positions[valid]] == candidate_raw[valid]
        candidate_indices = np.unique(positions[matching] + 1).astype(np.int32)
        if candidate_indices.size == 0:
            raise ValueError("No candidate video IDs were found in this prepared dataset's training catalogue")
        print(f"Scoring {len(candidate_indices):,} Model 1 candidates")
    else:
        candidate_indices = np.arange(1, len(video_ids) + 1, dtype=np.int32)
        print(f"No --candidate-file supplied; scoring the full training catalogue ({len(candidate_indices):,} videos)")

    results=[]; n_candidates=len(candidate_indices)
    with torch.no_grad():
        for start0 in range(0,n_candidates,args.batch_size):
            end0=min(start0+args.batch_size,n_candidates)
            # Translate candidate model indices back to raw IDs; row 0 is reserved for OOV.
            vid_idx=candidate_indices[start0:end0]
            raw_ids=video_ids[vid_idx - 1]
            cat=np.column_stack([np.full(len(raw_ids),ui,dtype=np.int32),vid_idx,
                                 np.full(len(raw_ids),ti,dtype=np.int32),
                                 np.repeat(user_cat[ui:ui+1],len(raw_ids),axis=0),video_cat[vid_idx]]).astype(np.int64)
            d=dur[vid_idx].copy(); d=np.where(d>0,d,float(metadata.get("video_duration_fallback_ms",0))/1000.0)
            age=np.where(upload[vid_idx]>0,np.maximum((ref-upload[vid_idx])/3_600_000.0,0.0),0.0).astype(np.float32)
            un=np.repeat(user_num[ui:ui+1],len(raw_ids),axis=0)
            numeric_raw=np.column_stack([np.log1p(np.maximum(d,0)),np.full(len(raw_ids),sin_h),np.full(len(raw_ids),cos_h),
                                         un,np.log1p(age),np.log1p(np.maximum(width[vid_idx],0)),
                                         np.log1p(np.maximum(height[vid_idx],0))]).astype(np.float32)
            numeric=(numeric_raw-means)/stds
            logits, watch_pred_log = model(torch.from_numpy(cat).to(device),torch.from_numpy(numeric.astype(np.float32)).to(device))
            probs=torch.sigmoid(logits).cpu().numpy(); labels=checkpoint["binary_targets"]
            predicted_watch_seconds = torch.expm1(torch.clamp(watch_pred_log, min=0.0)).cpu().numpy().reshape(-1)
            score=np.zeros(len(raw_ids),dtype=np.float32)
            for label,w in SCORE_WEIGHTS.items():
                if label in labels: score += w*probs[:,labels.index(label)]
            frame=pd.DataFrame({"video_id":raw_ids.astype(np.int64),"score":score,
                                "predicted_watch_time_seconds":predicted_watch_seconds})
            for j,label in enumerate(labels): frame[f"p_{label}"]=probs[:,j]
            results.append(frame.nlargest(max(1,args.top_k),"score"))
    result=pd.concat(results,ignore_index=True).nlargest(max(1,args.top_k),"score")
    print(result.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    if args.output:
        out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False); print(f"\nWrote: {out.resolve()}")
    print("\nNote: these are KuaiRand video IDs, not URLs or media files from your Instagram clone.")
if __name__=="__main__": main()
