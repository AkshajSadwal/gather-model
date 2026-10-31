"""Retrieve top-K video candidates for one raw KuaiRand user ID."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from model1_retrieval.data import data_fingerprint, load_metadata, user_features_from_raw_id
from model1_retrieval.model import TwoTowerRetrieval


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v2")
    parser.add_argument("--model-path", default="artifacts/retrieval/1k_v2_m4/best_model.pt")
    parser.add_argument("--index-dir", default="data/indexes/retrieval/1k_v2")
    parser.add_argument("--user-id", type=int, required=True, help="Raw user_id from the selected KuaiRand dataset")
    parser.add_argument("--tab", type=int, default=0, help="Raw tab ID; default 0 is mapped through the training tab vocabulary")
    parser.add_argument("--hour-utc", type=float, default=None, help="UTC hour 0..24; defaults to current UTC hour")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=131072)
    parser.add_argument("--exclude-video-ids", default=None, help="Optional text file with raw video IDs to exclude, one per line")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    args = parser.parse_args()
    if args.top_k < 1 or args.chunk_size < 1:
        parser.error("--top-k and --chunk-size must be positive")

    data_dir, model_path, index_dir = Path(args.data_dir), Path(args.model_path), Path(args.index_dir)
    metadata = load_metadata(data_dir)
    schema_fingerprint = data_fingerprint(data_dir)
    model_checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    index_metadata = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
    catalog_map_path = data_dir / "catalog_maps.npz"
    current_catalog_hash = sha256_file(catalog_map_path) if catalog_map_path.exists() else None
    if index_metadata.get("catalog_maps_sha256") != current_catalog_hash:
        raise ValueError("The retrieval index was built from a different video catalogue map. Rebuild the index.")
    if index_metadata.get("dataset") != metadata.get("dataset") or model_checkpoint.get("dataset") != metadata.get("dataset"):
        raise ValueError("Model, index, and prepared dataset do not match")
    if index_metadata.get("data_fingerprint") != schema_fingerprint or model_checkpoint.get("data_fingerprint") != schema_fingerprint:
        raise ValueError("Prepared metadata/ID maps differ from the model/index. Rebuild the retrieval model and index for this data version.")
    if index_metadata.get("checkpoint_sha256") != sha256_file(model_path):
        raise ValueError("This index was built from a different checkpoint. Rebuild it with model1_retrieval.build_index.")

    embeddings = np.load(index_dir / index_metadata["files"]["embeddings"], mmap_mode="r")
    video_ids = np.load(index_dir / index_metadata["files"]["video_ids"], mmap_mode="r", allow_pickle=False)
    if len(embeddings) != len(video_ids):
        raise ValueError("Index embedding rows do not match video ID count")

    hour = float(datetime.now(timezone.utc).hour) if args.hour_utc is None else float(args.hour_utc)
    user_cat, user_num, known_user = user_features_from_raw_id(data_dir, metadata, args.user_id, args.tab, hour)
    if not known_user:
        print("WARNING: user not found in the training vocabulary; using OOV ID with context/profile defaults")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model = TwoTowerRetrieval(metadata).to(device)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        user_vec = model.user_tower(torch.from_numpy(user_cat).to(device), torch.from_numpy(user_num).to(device))
        user_vector = user_vec[0].cpu().numpy().astype(np.float32, copy=False)

    excluded: set[int] = set()
    if args.exclude_video_ids:
        path = Path(args.exclude_video_ids)
        if not path.exists():
            raise FileNotFoundError(f"Exclude list not found: {path}")
        excluded = {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    best_scores = np.empty(0, dtype=np.float32)
    best_ids = np.empty(0, dtype=np.int64)
    for start in range(0, len(video_ids), args.chunk_size):
        end = min(start + args.chunk_size, len(video_ids))
        ids_chunk = np.asarray(video_ids[start:end], dtype=np.int64)
        vectors = np.asarray(embeddings[start:end], dtype=np.float32)
        scores = vectors @ user_vector
        if excluded:
            mask = np.fromiter((int(value) not in excluded for value in ids_chunk), count=len(ids_chunk), dtype=bool)
            scores = np.where(mask, scores, -np.inf)
        eligible = np.isfinite(scores)
        if not eligible.any():
            continue
        eligible_scores = scores[eligible]
        eligible_ids = ids_chunk[eligible]
        keep = min(args.top_k, len(eligible_scores))
        local = np.argpartition(eligible_scores, -keep)[-keep:]
        best_scores = np.concatenate([best_scores, eligible_scores[local]])
        best_ids = np.concatenate([best_ids, eligible_ids[local]])
        if len(best_scores) > args.top_k * 2:
            chosen = np.argpartition(best_scores, -args.top_k)[-args.top_k:]
            best_scores, best_ids = best_scores[chosen], best_ids[chosen]
    finite_best = np.isfinite(best_scores)
    best_scores, best_ids = best_scores[finite_best], best_ids[finite_best]
    if not len(best_ids):
        raise RuntimeError("No eligible videos found in the retrieval index")
    order = np.argsort(best_scores)[::-1][:args.top_k]
    result_ids, result_scores = best_ids[order], best_scores[order]
    print(f"Retrieved {len(result_ids)} candidates for user {args.user_id} on {device}")
    for rank, (video_id, score) in enumerate(zip(result_ids, result_scores), start=1):
        print(f"{rank:4d}. video_id={int(video_id)} similarity={float(score):.6f}")
    if args.output:
        import pandas as pd
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"rank": np.arange(1, len(result_ids) + 1), "video_id": result_ids,
                      "similarity": result_scores}).to_csv(out, index=False)
        print(f"Saved: {out.resolve()}")
    print("Note: this is an offline KuaiRand candidate retriever. It does not fetch video files or represent production serving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
