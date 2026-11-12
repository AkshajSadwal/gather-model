"""Evaluate a prepared Gather holdout: masked ranking metrics and full-index Recall@100."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from common.data_version import schema_fingerprint
from model1_retrieval.model import TwoTowerRetrieval
from model2_ranking.model_v2 import ReelsRankerV2


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--retrieval",type=Path,required=True); p.add_argument("--ranking",type=Path,required=True); p.add_argument("--index-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--max-queries",type=int,default=200); args=p.parse_args()
    meta=json.loads((args.data_dir/"metadata.json").read_text(encoding="utf-8")); fp=schema_fingerprint(args.data_dir)
    if meta.get("status")!="READY" or meta.get("schema_fingerprint") != json.loads((Path(__file__).resolve().parents[1]/"configs"/"gather_recommendation_schema_v1.json").read_text(encoding="utf-8"))["fingerprint"]: raise ValueError("Gather prepared data is not compatible/READY")
    m1=torch.load(args.retrieval,map_location="cpu",weights_only=False); m2=torch.load(args.ranking,map_location="cpu",weights_only=False)
    if m1.get("data_fingerprint")!=fp or m2.get("schema_fingerprint")!=meta["schema_fingerprint"]: raise ValueError("Checkpoint fingerprint mismatch")
    idx=json.loads((args.index_dir/"index_metadata.json").read_text(encoding="utf-8"))
    if idx.get("data_fingerprint")!=fp or idx.get("checkpoint_sha256")!=sha(args.retrieval): raise ValueError("Index/checkpoint fingerprint mismatch")
    with np.load(args.data_dir/"test.npz",allow_pickle=False) as d: cat=np.asarray(d["categorical"],dtype=np.int32); num=np.asarray(d["numeric"],dtype=np.float32); labels=np.asarray(d["labels"],dtype=np.float32); masks=np.asarray(d["label_mask"],dtype=bool); watch=np.asarray(d["watch_log_seconds"],dtype=np.float32); watch_mask=np.asarray(d["watch_mask"],dtype=bool); item=np.asarray(d["catalog_video_indices"],dtype=np.int64)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank=ReelsRankerV2(meta["categorical_sizes"],10,num_targets=len(meta["binary_targets"])).to(device); rank.load_state_dict(m2["model_state_dict"]); rank.eval()
    with torch.no_grad(): logits,pred_watch=rank(torch.from_numpy(cat).long().to(device),torch.from_numpy(num).float().to(device)); probs=torch.sigmoid(logits).cpu().numpy(); pred_watch=pred_watch.cpu().numpy()
    ranking_metrics={}
    for j,name in enumerate(meta["binary_targets"]):
        valid=masks[:,j]; ranking_metrics[name]={"observed":int(valid.sum()),"auc":float(roc_auc_score(labels[valid,j],probs[valid,j])) if valid.sum() and np.unique(labels[valid,j]).size==2 else None}
    watch_valid=watch_mask
    ranking_metrics["watch_log_mae"]={"observed":int(watch_valid.sum()),"value":float(np.mean(np.abs(pred_watch[watch_valid]-watch[watch_valid]))) if watch_valid.any() else None}
    index_meta=idx; vectors=np.load(args.index_dir/index_meta["files"]["embeddings"],mmap_mode="r",allow_pickle=False); video_ids=np.load(args.index_dir/index_meta["files"]["video_ids"],mmap_mode="r",allow_pickle=False)
    retrieval=TwoTowerRetrieval(meta).to(device); retrieval.load_state_dict(m1["model_state_dict"]); retrieval.eval(); likes=meta["binary_targets"].index("is_like"); rows=np.flatnonzero(masks[:,likes] & (labels[:,likes]>0) & (item>0) & (item<=len(video_ids)))
    if len(rows)>args.max_queries: rows=np.random.default_rng(47).choice(rows,args.max_queries,replace=False)
    hits=0; measured=0
    for row in rows:
        with torch.no_grad(): u=retrieval.user_tower(torch.from_numpy(cat[row:row+1][:,TwoTowerRetrieval.USER_COLUMNS]).long().to(device),torch.from_numpy(num[row:row+1][:,TwoTowerRetrieval.USER_NUMERIC_COLUMNS]).float().to(device)).cpu().numpy()[0]
        k=min(100,len(video_ids)); best_scores=np.empty(0,dtype=np.float32); best_ids=np.empty(0,dtype=np.int64)
        for start in range(0,len(video_ids),131072):
            end=min(start+131072,len(video_ids)); scores=np.asarray(vectors[start:end],dtype=np.float32)@u; local_k=min(k,len(scores)); local=np.argpartition(scores,-local_k)[-local_k:]
            best_scores=np.concatenate((best_scores,scores[local])); best_ids=np.concatenate((best_ids,np.asarray(video_ids[start:end][local],dtype=np.int64)))
            if len(best_scores)>k*2:
                keep=np.argpartition(best_scores,-k)[-k:]; best_scores,best_ids=best_scores[keep],best_ids[keep]
        measured+=1; hits+=int(np.any(best_ids==item[row]))
    report={"status":"READY","schema_fingerprint":meta["schema_fingerprint"],"prepared_data_fingerprint":m1.get("prepared_data_fingerprint"),"device":str(device),"ranking":ranking_metrics,"retrieval":{"queries":measured,"recall_at_100":hits/measured if measured else None},"artifacts":{"retrieval_sha256":sha(args.retrieval),"ranking_sha256":sha(args.ranking),"index_version":index_meta.get("created_at_utc"),"index_checkpoint_sha256":index_meta.get("checkpoint_sha256")}}
    args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix(args.output.suffix+".tmp"); tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8"); os.replace(tmp,args.output); print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()