"""Feature-aware multi-task neural ranker for short-video/Reels recommendations."""
from __future__ import annotations
import torch
from torch import nn
from model_common import BINARY_TARGETS

class ReelsRankerV2(nn.Module):
    """Consumes multiple categorical embeddings and standardized numeric features."""
    def __init__(self, categorical_sizes: list[int], num_numeric_features: int,
                 num_targets: int = len(BINARY_TARGETS), embedding_dim: int = 32):
        super().__init__()
        if not categorical_sizes or any(int(s) < 1 for s in categorical_sizes):
            raise ValueError("categorical_sizes must contain positive embedding table sizes")
        self.categorical_sizes = [int(x) for x in categorical_sizes]
        self.num_numeric_features = int(num_numeric_features)
        dims = []
        for i, size in enumerate(self.categorical_sizes):
            if i in (0, 1):
                d = embedding_dim  # user ID, video ID
            elif i == 2:
                d = 8  # tab/scenario
            else:
                d = min(16, max(4, int(round(size ** 0.25 * 2))))
            dims.append(d)
        self.embedding_dims = dims
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size, dim in zip(self.categorical_sizes, dims)
        ])
        input_dim = sum(dims) + self.num_numeric_features
        # Three hidden layers: 512 -> 256 -> 128.
        # Keep the final shared representation at 128 so both prediction heads
        # retain their existing output interface.
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.10),
        )
        self.action_head = nn.Linear(128, num_targets)
        self.watch_time_head = nn.Linear(128, 1)
        with torch.no_grad():
            for emb in self.embeddings:
                emb.weight[0].zero_()

    def forward(self, categorical_features: torch.Tensor, numeric_features: torch.Tensor):
        x_parts = [emb(categorical_features[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_parts.append(numeric_features.float())
        hidden = self.shared(torch.cat(x_parts, dim=1))
        return self.action_head(hidden), self.watch_time_head(hidden).squeeze(1)
