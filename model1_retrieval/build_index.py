"""Build a chunked, float16 video-vector index from a trained retrieval checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from model1_retrieval.data import build_video_catalog, data_fingerprint, load_metadata
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
    parser.add_argument("--output-dir", default="data/indexes/retrieval/1k_v2")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing index after building a new one")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    data_dir, model_path, output_dir = Path(args.data_dir), Path(args.model_path), Path(args.output_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Retrieval checkpoint not found: {model_path}")
    metadata = load_metadata(data_dir)
    schema_fingerprint = data_fingerprint(data_dir)
    catalog = build_video_catalog(data_dir, metadata)
    catalog_map_path = data_dir / "catalog_maps.npz"
    catalog_maps_sha256 = sha256_file(catalog_map_path) if catalog_map_path.exists() else None
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("dataset") != metadata.get("dataset"):
        raise ValueError("Checkpoint dataset does not match the prepared catalogue dataset")
    if checkpoint.get("data_fingerprint") != schema_fingerprint:
        raise ValueError("Checkpoint was trained with different prepared metadata/ID maps")
    model = TwoTowerRetrieval(metadata).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Index output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Index directory {output_dir} already contains files. Choose another --output-dir "
                "or pass --overwrite to replace it."
            )
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    embedding_path = temporary_dir / "video_embeddings.npy"
    mmap = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float16,
                                     shape=(len(catalog.raw_ids), 128))
    with torch.no_grad():
        for start in range(0, len(catalog.raw_ids), args.batch_size):
            end = min(start + args.batch_size, len(catalog.raw_ids))
            video_indices = np.arange(start + 1, end + 1, dtype=np.int32)
            cat = torch.from_numpy(np.ascontiguousarray(catalog.categorical[video_indices][:, TwoTowerRetrieval.VIDEO_CATALOG_COLUMNS])).to(device)
            num = torch.from_numpy(np.ascontiguousarray(catalog.numeric[video_indices][:, TwoTowerRetrieval.VIDEO_CATALOG_NUMERIC_COLUMNS])).to(device)
            vectors = model.video_tower(cat, num).cpu().numpy().astype(np.float16, copy=False)
            mmap[start:end] = vectors
            if start == 0 or end == len(catalog.raw_ids) or (end // args.batch_size) % 25 == 0:
                print(f"Indexed {end:,}/{len(catalog.raw_ids):,} videos", flush=True)
    mmap.flush()
    np.save(temporary_dir / "video_ids.npy", catalog.raw_ids, allow_pickle=False)

    index_metadata = {
        "dataset": metadata.get("dataset"),
        "data_fingerprint": schema_fingerprint,
        "catalog_maps_sha256": catalog_maps_sha256,
        "catalogue_type": "metadata_backed_with_oov_video_id_embeddings" if catalog_maps_sha256 else "training_video_ids_only",
        "cold_start_item_count": int(np.count_nonzero(catalog.categorical[1:, 0] == 0)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
        "video_count": int(len(catalog.raw_ids)),
        "embedding_dim": 128,
        "dtype": "float16",
        "normalization": "L2-normalized by video tower",
        "checkpoint_seed": checkpoint.get("seed"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "best_validation_metrics": checkpoint.get("best_validation_metrics"),
        "files": {"embeddings": embedding_path.name, "video_ids": "video_ids.npy"},
    }
    (temporary_dir / "index_metadata.json").write_text(json.dumps(index_metadata, indent=2), encoding="utf-8")
    del mmap
    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not args.overwrite:
                raise FileExistsError(f"Index directory appeared during build: {output_dir}")
            backup_dir = output_dir.with_name(output_dir.name + ".backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            os.replace(output_dir, backup_dir)
            print(f"Previous index preserved at: {backup_dir}")
        else:
            output_dir.rmdir()
    os.replace(temporary_dir, output_dir)
    final_embedding_path = output_dir / "video_embeddings.npy"
    print(f"Index created: {output_dir.resolve()}")
    print(f"Embeddings: {final_embedding_path.stat().st_size / (1024 ** 3):.2f} GiB (float16)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
