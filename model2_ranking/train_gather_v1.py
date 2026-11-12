"""Train the Gather ranker with observed-label masks; never treats unavailable labels as negatives."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from model2_ranking.model_v2 import ReelsRankerV2


def load_split(path: Path):
    with np.load(path, allow_pickle=False) as data:
        required = ("categorical", "numeric", "labels", "label_mask", "watch_log_seconds", "watch_mask")
        if any(name not in data.files for name in required): raise ValueError(f"{path} is missing masked training arrays")
        return {name: torch.from_numpy(np.asarray(data[name])) for name in required}


def masked_bce(logits, labels, mask, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate(model, split, device, pos_weight, like_index, batch_size=2048):
    model.eval(); losses=[]; probs=[]; truths=[]; masks=[]
    with torch.no_grad():
        for start in range(0, len(split["labels"]), batch_size):
            sl = slice(start, start + batch_size)
            cat = split["categorical"][sl].long().to(device); num = split["numeric"][sl].float().to(device)
            labels = split["labels"][sl].float().to(device); mask = split["label_mask"][sl].float().to(device)
            watch = split["watch_log_seconds"][sl].float().to(device); watch_mask = split["watch_mask"][sl].float().to(device)
            logits, watch_pred = model(cat, num)
            action = masked_bce(logits, labels, mask, pos_weight)
            watch_loss = (F.smooth_l1_loss(watch_pred, watch, reduction="none") * watch_mask).sum() / watch_mask.sum().clamp_min(1.0)
            losses.append(float((action + .2 * watch_loss).item()))
            probs.append(torch.sigmoid(logits[:, like_index]).cpu().numpy()); truths.append(labels[:, like_index].cpu().numpy()); masks.append(mask[:, like_index].cpu().numpy())
    p, y, m = np.concatenate(probs), np.concatenate(truths), np.concatenate(masks).astype(bool)
    auc = float(roc_auc_score(y[m], p[m])) if m.sum() and np.unique(y[m]).size == 2 else None
    return {"masked_loss": float(np.mean(losses)) if losses else None, "is_like_auc": auc, "is_like_labels": int(m.sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--batch-size", type=int, default=1024); parser.add_argument("--seed", type=int, default=42); args=parser.parse_args()
    meta_path=args.data_dir / "metadata.json"
    metadata=json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "READY" or metadata.get("schema_version") != "gather-reco-v1": raise ValueError("Gather dataset readiness/schema is not READY")
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train, valid = load_split(args.data_dir / "train.npz"), load_split(args.data_dir / "validation.npz")
    targets=metadata["binary_targets"]; like_index=targets.index("is_like")
    y=train["labels"].float(); m=train["label_mask"].float()
    available=(y*m).sum(0); negatives=m.sum(0)-available
    if available[like_index] < 100: raise ValueError("Not enough observed is_like labels to train a Gather ranker")
    pos_weight=(negatives / available.clamp_min(1)).clamp(1,20)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ReelsRankerV2(metadata["categorical_sizes"], len(metadata["numeric_features"]), num_targets=len(targets)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for _epoch in range(args.epochs):
        model.train(); order=torch.randperm(len(y))
        for start in range(0, len(order), args.batch_size):
            idx=order[start:start+args.batch_size]
            cat=train["categorical"][idx].long().to(device); num=train["numeric"][idx].float().to(device)
            labels=train["labels"][idx].float().to(device); mask=train["label_mask"][idx].float().to(device)
            watch=train["watch_log_seconds"][idx].float().to(device); watch_mask=train["watch_mask"][idx].float().to(device)
            logits, watch_pred=model(cat,num)
            loss=masked_bce(logits,labels,mask,pos_weight.to(device))
            watch_loss=(F.smooth_l1_loss(watch_pred,watch,reduction="none")*watch_mask).sum()/watch_mask.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True); (loss+.2*watch_loss).backward(); optimizer.step()
    metrics=evaluate(model,valid,device,pos_weight.to(device),like_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp=args.output.with_suffix(args.output.suffix+".tmp")
    torch.save({"model_state_dict":model.state_dict(),"dataset":"gather_v1","model_version":1,"schema_fingerprint":metadata["schema_fingerprint"],"prepared_data_fingerprint":metadata["prepared_data_fingerprint"],"categorical_sizes":metadata["categorical_sizes"],"numeric_mean":metadata["numeric_mean"],"numeric_std":metadata["numeric_std"],"num_numeric_features":len(metadata["numeric_features"]),"binary_targets":targets,"primary_objective":"is_like","validation_metrics":metrics,"device":str(device)},temp); os.replace(temp,args.output)
    print(json.dumps({"checkpoint":str(args.output),"device":str(device),"validation":metrics},sort_keys=True))

if __name__ == "__main__": main()