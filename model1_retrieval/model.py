"""Two-tower retrieval model for the KuaiRand short-video recommendation project."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class FeatureTower(nn.Module):
    def __init__(self, categorical_sizes: list[int], numeric_features: int,
                 embedding_dims: list[int], tower_name: str):
        super().__init__()
        if len(categorical_sizes) != len(embedding_dims):
            raise ValueError(f"{tower_name}: categorical sizes and embedding dimensions differ")
        if any(int(size) < 1 for size in categorical_sizes):
            raise ValueError(f"{tower_name}: embedding table sizes must be positive")
        if any(int(dim) < 1 for dim in embedding_dims):
            raise ValueError(f"{tower_name}: embedding dimensions must be positive")
        self.categorical_sizes = [int(x) for x in categorical_sizes]
        self.numeric_features = int(numeric_features)
        self.embedding_dims = [int(x) for x in embedding_dims]
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0)
            for size, dim in zip(self.categorical_sizes, self.embedding_dims)
        ])
        input_dim = sum(self.embedding_dims) + self.numeric_features
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.10),
        )
        with torch.no_grad():
            for embedding in self.embeddings:
                embedding.weight[0].zero_()

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        pieces = [embedding(categorical[:, i].long()) for i, embedding in enumerate(self.embeddings)]
        pieces.append(numeric.float())
        representation = self.network(torch.cat(pieces, dim=1))
        return F.normalize(representation, p=2, dim=1, eps=1e-8)


def make_embedding_dims(metadata: dict) -> tuple[list[int], list[int]]:
    """Choose modest side-feature embeddings and explicit entity-ID embeddings.

    The video ID table is intentionally dimension 16 rather than 32: KuaiRand-1K
    has nearly four million videos, so this keeps the trainable retrieval state
    substantially smaller while the tower still emits 128-dimensional vectors.
    """
    sizes = [int(x) for x in metadata["categorical_sizes"]]
    user_cols = [0, 2, *range(3, 29)]
    video_cols = [1, *range(29, 36)]
    user_sizes = [sizes[i] for i in user_cols]
    video_sizes = [sizes[i] for i in video_cols]

    user_dims: list[int] = []
    for i, size in enumerate(user_sizes):
        if i == 0:       # user ID
            user_dims.append(32)
        elif i == 1:     # feed tab / scenario
            user_dims.append(8)
        else:
            user_dims.append(min(8, max(4, int(round(math.sqrt(math.sqrt(size)) * 2)))))

    video_dims: list[int] = []
    for i, size in enumerate(video_sizes):
        if i == 0:       # video ID; exact embedding table
            video_dims.append(16)
        elif i in (1, 5, 7):  # author ID, music ID, tag
            video_dims.append(8)
        else:
            video_dims.append(min(8, max(4, int(round(math.sqrt(math.sqrt(size)) * 2)))))
    return user_dims, video_dims


class TwoTowerRetrieval(nn.Module):
    """User and video towers with independent 512 -> 256 -> 128 MLPs."""
    USER_COLUMNS = [0, 2, *range(3, 29)]
    VIDEO_COLUMNS = [1, *range(29, 36)]
    VIDEO_CATALOG_COLUMNS = list(range(8))
    VIDEO_CATALOG_NUMERIC_COLUMNS = list(range(3))
    USER_NUMERIC_COLUMNS = [1, 2, 3, 4, 5, 6]
    # Video age is deliberately excluded: it is time-dependent and belongs in
    # ranking/freshness logic, not a static, cacheable retrieval item vector.
    VIDEO_NUMERIC_COLUMNS = [0, 8, 9]

    def __init__(self, metadata: dict):
        super().__init__()
        self.metadata_dataset = str(metadata.get("dataset", "unknown"))
        sizes = [int(x) for x in metadata["categorical_sizes"]]
        user_dims, video_dims = make_embedding_dims(metadata)
        user_sizes = [sizes[i] for i in self.USER_COLUMNS]
        video_sizes = [sizes[i] for i in self.VIDEO_COLUMNS]
        self.user_tower = FeatureTower(user_sizes, len(self.USER_NUMERIC_COLUMNS), user_dims, "user")
        self.video_tower = FeatureTower(video_sizes, len(self.VIDEO_NUMERIC_COLUMNS), video_dims, "video")
        self.user_categorical_sizes = user_sizes
        self.video_categorical_sizes = video_sizes
        self.user_embedding_dims = user_dims
        self.video_embedding_dims = video_dims
        self.embedding_dim = 128

    @staticmethod
    def split_user_features(categorical: torch.Tensor, numeric: torch.Tensor):
        return categorical[:, TwoTowerRetrieval.USER_COLUMNS], numeric[:, TwoTowerRetrieval.USER_NUMERIC_COLUMNS]

    @staticmethod
    def split_video_features(categorical: torch.Tensor, numeric: torch.Tensor):
        return categorical[:, TwoTowerRetrieval.VIDEO_COLUMNS], numeric[:, TwoTowerRetrieval.VIDEO_NUMERIC_COLUMNS]

    def encode_user(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        user_cat, user_num = self.split_user_features(categorical, numeric)
        return self.user_tower(user_cat, user_num)

    def encode_video(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        video_cat, video_num = self.split_video_features(categorical, numeric)
        return self.video_tower(video_cat, video_num)

    def forward(self, user_categorical: torch.Tensor, user_numeric: torch.Tensor,
                video_categorical: torch.Tensor, video_numeric: torch.Tensor):
        return self.user_tower(user_categorical, user_numeric), self.video_tower(video_categorical, video_numeric)
