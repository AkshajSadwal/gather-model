"""Explicitly promote a validated Gather model/index manifest; never runs automatically."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def eligible(evaluation, retrieval_hash, ranking_hash, min_like_auc, min_recall_at_100):
    auc=evaluation.get("ranking",{}).get("is_like",{}).get("auc")
    recall=evaluation.get("retrieval",{}).get("recall_at_100")
    artifacts=evaluation.get("artifacts",{})
    return (evaluation.get("status")=="READY" and artifacts.get("retrieval_sha256")==retrieval_hash and artifacts.get("ranking_sha256")==ranking_hash and isinstance(auc,(int,float)) and auc>=min_like_auc and isinstance(recall,(int,float)) and recall>=min_recall_at_100)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--evaluation",type=Path,required=True); p.add_argument("--retrieval",type=Path,required=True); p.add_argument("--ranking",type=Path,required=True); p.add_argument("--index-dir",type=Path,required=True); p.add_argument("--active-pointer",type=Path,required=True); p.add_argument("--min-like-auc",type=float,required=True); p.add_argument("--min-recall-at-100",type=float,required=True); p.add_argument("--activate",action="store_true"); p.add_argument("--approval",default=""); args=p.parse_args()
    evaluation=json.loads(args.evaluation.read_text(encoding="utf-8")); index=json.loads((args.index_dir/"index_metadata.json").read_text(encoding="utf-8")); retrieval_hash=sha(args.retrieval); ranking_hash=sha(args.ranking)
    if not eligible(evaluation,retrieval_hash,ranking_hash,args.min_like_auc,args.min_recall_at_100): raise SystemExit("Promotion gates failed: fingerprinted artifacts and configured validation thresholds must all pass.")
    if index.get("checkpoint_sha256")!=retrieval_hash or evaluation.get("artifacts",{}).get("index_checkpoint_sha256")!=retrieval_hash: raise SystemExit("Promotion rejected: index is not built from the evaluated retrieval checkpoint.")
    pointer={"provider":"two_stage_v1","modelVersion":"gather_two_stage_v1","schemaFingerprint":evaluation["schema_fingerprint"],"retrievalCheckpoint":str(args.retrieval.resolve()),"retrievalSha256":retrieval_hash,"rankingCheckpoint":str(args.ranking.resolve()),"rankingSha256":ranking_hash,"indexDirectory":str(args.index_dir.resolve()),"indexVersion":index.get("created_at_utc"),"evaluation":str(args.evaluation.resolve()),"promotedAtUtc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    if args.activate:
        if args.approval!="PROMOTE": raise SystemExit("Activation requires explicit --approval PROMOTE")
        args.active_pointer.parent.mkdir(parents=True,exist_ok=True); temp=args.active_pointer.with_suffix(args.active_pointer.suffix+".tmp"); temp.write_text(json.dumps(pointer,indent=2)+"\n",encoding="utf-8"); os.replace(temp,args.active_pointer)
    print(json.dumps({"eligible":True,"activated":args.activate,"manifest":pointer},sort_keys=True))
if __name__=="__main__": main()