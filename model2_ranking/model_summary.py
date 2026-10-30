"""Print the exact V2 parameter count and approximate model-state memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model2_ranking.model_v2 import ReelsRankerV2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Prepared dataset directory containing metadata.json")
    parser.add_argument("--embedding-dim", type=int, default=32)
    args = parser.parse_args()

    metadata_path = Path(args.data_dir) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    model = ReelsRankerV2(
        metadata["categorical_sizes"],
        len(metadata["numeric_features"]),
        num_targets=len(metadata["binary_targets"]),
        embedding_dim=args.embedding_dim,
    )

    print(f"Dataset: {metadata.get('dataset', 'unknown')}")
    print(f"Architecture: input -> 512 -> 256 -> 128 -> (10 action outputs, 1 watch-time output)")
    print(f"Categorical fields: {len(metadata['categorical_sizes'])}")
    print(f"Numeric fields: {len(metadata['numeric_features'])}")
    print(f"Embedding dimension setting: {args.embedding_dim}")
    print("\nParameters by component:")
    totals = {}
    for name, parameter in model.named_parameters():
        group = "embeddings" if name.startswith("embeddings.") else (
            "hidden_layers" if name.startswith("shared.") else "output_heads"
        )
        totals[group] = totals.get(group, 0) + parameter.numel()
    for group, count in totals.items():
        print(f"  {group:16s}: {count:>15,}")

    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"  {'TOTAL':16s}: {total:>15,}")
    print(f"\nFloat32 weights only: {total * 4 / (1024 ** 3):.2f} GiB")
    print(
        "Approx. FP32 AdamW model state (weights + gradients + two optimizer states): "
        f"{total * 16 / (1024 ** 3):.2f} GiB"
    )
    print("Note: excludes dataset arrays, activations, temporary buffers, and macOS/MPS overhead.")


if __name__ == "__main__":
    main()
