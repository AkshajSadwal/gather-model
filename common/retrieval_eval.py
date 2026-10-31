"""Exact full-catalogue ranking of held-out positives (pure NumPy, no torch)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def rank_against_catalog(
    user_vector_sets: dict[str, np.ndarray],
    positive_rows: np.ndarray,
    catalog,
    chunk_size: int = 131072,
    query_block: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """1-based rank of each query's positive item among the *whole* catalogue.

    Parameters
    ----------
    user_vector_sets: name -> float32 array [Q, D]. Every set is ranked against the same
        positives, so a shuffled-user set measures how much of the score is personalisation.
    positive_rows: 0-based catalogue row of each query's positive item, shape [Q].
    catalog: array-like [N, D] (a float16 np.memmap is fine). It is read chunk by chunk.

    The positive item itself is excluded from its own comparison. Exact ties with the
    positive's score are not counted as being ranked above it.
    """
    positive_rows = np.asarray(positive_rows, dtype=np.int64)
    q = positive_rows.shape[0]
    n = int(catalog.shape[0])
    positive_vectors = np.asarray(catalog[positive_rows], dtype=np.float32)
    sets = {name: np.ascontiguousarray(v, dtype=np.float32) for name, v in user_vector_sets.items()}
    for name, vectors in sets.items():
        if vectors.shape[0] != q:
            raise ValueError(f"user vector set '{name}' has {vectors.shape[0]} rows for {q} queries")
    positive_scores = {name: np.einsum("qd,qd->q", v, positive_vectors) for name, v in sets.items()}
    greater = {name: np.zeros(q, dtype=np.int64) for name in sets}

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = np.asarray(catalog[start:stop], dtype=np.float32)
        for a in range(0, q, query_block):
            b = min(a + query_block, q)
            rows = positive_rows[a:b]
            inside = np.flatnonzero((rows >= start) & (rows < stop))
            for name, vectors in sets.items():
                scores = vectors[a:b] @ block.T
                if inside.size:
                    scores[inside, rows[inside] - start] = -np.inf
                greater[name][a:b] += (scores > positive_scores[name][a:b, None]).sum(axis=1)
        if progress is not None:
            progress(stop, n)
    return {name: g + 1 for name, g in greater.items()}
