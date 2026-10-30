"""Rank KuaiRand video candidates using the metadata-enriched V2 model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model_v2 import ReelsRankerV2

SCORE_WEIGHTS={"is_click":0.15,"is_like":0.60,"is_follow":0.75,"is_comment":0.65,
               "is_forward":1.00,"long_view":1.00,"is_profile_enter":0.25,
               "complete_view":0.40,"short_view":-0.90,"is_hate":-2.00}

def find_index(sorted_values, value):
    idx=int(np.searchsorted(sorted_values,value))
    return idx+1 if idx<len(sorted_values) and int(sorted_values[idx])==int(value) else 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",default="data/processed/pure_v2")
    p.add_argument("--model-path",default="artifacts/pure_v2/best_model.pt")
    p.add_argument("--user-id",required=True,type=int)
    p.add_argument("--tab",type=int,default=0)
    p.add_argument("--top-k",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=2048)
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
    model=ReelsRankerV2(checkpoint["categorical_sizes"],checkpoint["num_numeric_features"],
                        num_targets=len(checkpoint["binary_targets"]),embedding_dim=checkpoint["embedding_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    ui=find_index(user_ids,args.user_id); ti=find_index(tab_ids,args.tab)
    if ui==0: print(f"Warning: user {args.user_id} not in training IDs; using OOV user and metadata features.")
    if ti==0: print(f"Warning: tab {args.tab} not in training IDs; using OOV tab.")
    ref=int(metadata["reference_time_ms"]); hours=(ref//3_600_000)%24; angle=2*np.pi*hours/24.0
    sin_h,cos_h=float(np.sin(angle)),float(np.cos(angle))
    means=np.asarray(metadata["numeric_mean"],dtype=np.float32); stds=np.asarray(metadata["numeric_std"],dtype=np.float32)
    results=[]; nvid=len(video_ids)
    with torch.no_grad():
        for start0 in range(0,nvid,args.batch_size):
            end0=min(start0+args.batch_size,nvid)
            # Recommending by trained video index; row 0 is reserved for OOV.
            vid_idx=np.arange(start0+1,end0+1,dtype=np.int32); raw_ids=video_ids[start0:end0]
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
            logits,_=model(torch.from_numpy(cat).to(device),torch.from_numpy(numeric.astype(np.float32)).to(device))
            probs=torch.sigmoid(logits).cpu().numpy(); labels=checkpoint["binary_targets"]
            score=np.zeros(len(raw_ids),dtype=np.float32)
            for label,w in SCORE_WEIGHTS.items():
                if label in labels: score += w*probs[:,labels.index(label)]
            frame=pd.DataFrame({"video_id":raw_ids.astype(np.int64),"score":score})
            for j,label in enumerate(labels): frame[f"p_{label}"]=probs[:,j]
            results.append(frame.nlargest(max(1,args.top_k),"score"))
    result=pd.concat(results,ignore_index=True).nlargest(max(1,args.top_k),"score")
    print(result.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    if args.output:
        out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False); print(f"\nWrote: {out.resolve()}")
    print("\nNote: these are KuaiRand video IDs, not URLs or media files from your Instagram clone.")
if __name__=="__main__": main()
