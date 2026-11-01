"""Losses for the pool-softmax retrieval trainer (kept tiny so it can be checked against NumPy)."""
from __future__ import annotations

import torch


def pool_softmax_loss(user_vec: torch.Tensor, cand_vec: torch.Tensor, own_cols: torch.Tensor,
                      exclude: torch.Tensor, logq: torch.Tensor, temperature: float,
                      row_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled softmax over a shared candidate set with logQ correction.

    user_vec  [B, D]  normalised user vectors
    cand_vec  [C, D]  normalised candidate (video) vectors, shared by every row
    own_cols  [B]     int64 column of each row's own positive in the candidate set
    exclude   [B, C]  bool, True for the user's *other* known positives (neither target nor negative)
    logq      [C]     log expected sampling count of each candidate, subtracted from the logits
    row_weights [B]   per-row weights (event strength)

    per_row = logsumexp(non-excluded corrected logits) - corrected logit of the own positive.
    Mirrors common.pools.reference_pool_softmax_loss.
    """
    logits = torch.matmul(user_vec, cand_vec.transpose(0, 1)) / temperature
    logits = logits - logq.unsqueeze(0)
    own_logit = logits.gather(1, own_cols.unsqueeze(1)).squeeze(1)
    logits = logits.masked_fill(exclude, float("-inf"))
    per_row = torch.logsumexp(logits, dim=1) - own_logit
    loss = torch.sum(per_row * row_weights) / row_weights.sum().clamp(min=1e-6)
    return loss, per_row
