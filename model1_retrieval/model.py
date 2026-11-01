"""Two-tower retrieval model for the KuaiRand short-video recommendation project."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class FeatureTower(nn.Module):
    """Embeds categorical fields, concatenates numeric fields and applies an MLP.

    ``gated_id_column`` marks one categorical column as a high-cardinality entity ID (the
    video ID). Two mechanisms stop the tower from memorising IDs of rarely seen entities:

    * ``id_keep`` (a saved buffer): IDs whose flag is False are replaced by the OOV index 0
      every time, in training *and* serving. The trainer sets it from training frequency,
      so rare and brand-new videos look the same to the model.
    * ``id_dropout`` (training only): a random fraction of rows also loses the ID, which
      teaches the tower to rank from content features alone.

    With the defaults (all IDs kept, no dropout, legacy output activation) the tower behaves
    exactly like before, and checkpoints saved without ``id_keep`` still load.

    ``linear_output`` removes the final ReLU and Dropout. This lets the normalized vector use
    negative as well as positive coordinates. Its flag is saved as a buffer so existing model
    loaders recover the architecture from the state dict; pre-flag checkpoints load as legacy.

    ``linear_output`` removes the final ReLU and dropout, so the 128-d vector can use every
    direction of the space before normalisation. With the legacy ReLU every embedding lies in the
    non-negative orthant (all cosines >= 0). The flag is stored in the checkpoint as a buffer,
    so every loader (build_index, evaluators, retrieve) picks it up with no code change, and
    checkpoints saved before this option existed load as ``linear_output=False``.
    """

    def __init__(self, categorical_sizes: list[int], numeric_features: int,
                 embedding_dims: list[int], tower_name: str, gated_id_column: int | None = None,
                 linear_output: bool = False):
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
        if gated_id_column is not None and not 0 <= int(gated_id_column) < len(self.categorical_sizes):
            raise ValueError(f"{tower_name}: gated_id_column {gated_id_column} is outside the categorical fields")
        self.gated_id_column = None if gated_id_column is None else int(gated_id_column)
        self.linear_output = bool(linear_output)
        self.register_buffer("linear_output_flag", torch.tensor(self.linear_output, dtype=torch.bool))
        self.id_dropout = 0.0
        if self.gated_id_column is not None:
            self.register_buffer("id_keep", torch.ones(self.categorical_sizes[self.gated_id_column], dtype=torch.bool))

    def set_id_keep(self, keep) -> None:
        """Choose which entity IDs keep their own embedding (True) or are treated as OOV (False)."""
        if self.gated_id_column is None:
            raise ValueError("This tower has no gated ID column")
        keep = torch.as_tensor(keep, dtype=torch.bool)
        if tuple(keep.shape) != tuple(self.id_keep.shape):
            raise ValueError(f"id_keep must have shape {tuple(self.id_keep.shape)}, got {tuple(keep.shape)}")
        keep = keep.clone()
        keep[0] = False  # index 0 is the OOV row itself
        self.id_keep.copy_(keep.to(self.id_keep.device))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        # Checkpoints written before ID gating existed have no id_keep entry: keep every ID.
        key = prefix + "id_keep"
        if self.gated_id_column is not None and key not in state_dict:
            state_dict[key] = self.id_keep.detach().clone()
        # Checkpoints written before linear_output existed used the final ReLU.
        flag_key = prefix + "linear_output_flag"
        if flag_key not in state_dict:
            state_dict[flag_key] = torch.tensor(False, dtype=torch.bool)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)
        self.linear_output = bool(self.linear_output_flag.item())

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        columns = [categorical[:, i].long() for i in range(len(self.embeddings))]
        if self.gated_id_column is not None:
            ids = columns[self.gated_id_column]
            ids = torch.where(self.id_keep[ids], ids, torch.zeros_like(ids))
            if self.training and self.id_dropout > 0.0:
                dropped = torch.rand(ids.shape, device=ids.device) < self.id_dropout
                ids = ids.masked_fill(dropped, 0)
            columns[self.gated_id_column] = ids
        pieces = [embedding(column) for embedding, column in zip(self.embeddings, columns)]
        pieces.append(numeric.float())
        features = torch.cat(pieces, dim=1)
        # network[:7] ends at the last Linear, skipping its ReLU and Dropout.
        representation = self.network[:7](features) if self.linear_output else self.network(features)
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

    def __init__(self, metadata: dict, linear_output: bool = False):
        super().__init__()
        self.metadata_dataset = str(metadata.get("dataset", "unknown"))
        sizes = [int(x) for x in metadata["categorical_sizes"]]
        user_dims, video_dims = make_embedding_dims(metadata)
        user_sizes = [sizes[i] for i in self.USER_COLUMNS]
        video_sizes = [sizes[i] for i in self.VIDEO_COLUMNS]
        self.user_tower = FeatureTower(user_sizes, len(self.USER_NUMERIC_COLUMNS), user_dims, "user",
                                       linear_output=linear_output)
        self.video_tower = FeatureTower(video_sizes, len(self.VIDEO_NUMERIC_COLUMNS), video_dims, "video",
                                        gated_id_column=0, linear_output=linear_output)
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
