"""PureSVD latent-factor recommender (truncated SVD of the user-item matrix).

Items are represented by the top singular vectors of the (optionally damped and recency
weighted) matrix. A user's scores are the projection of their history onto the latent
space and back: scores = (x V) V^T. Fitting once at ``max_factors`` provides every smaller
dimension for free, because singular vectors are nested.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


@dataclass(frozen=True)
class LatentFactors:
    item_factors: np.ndarray    # (n_items, k_max) float32, ordered by singular value (largest first)
    singular_values: np.ndarray
    column_scale: np.ndarray    # (n_items,) factor applied to each history column before projecting


def fit_puresvd(history: sparse.csr_matrix, max_factors: int = 128, item_damping: float = 0.0,
                seed: int = 42) -> LatentFactors:
    """Truncated SVD of ``history`` with columns scaled by (buyer count) ** -item_damping."""
    counts = np.asarray((history != 0).sum(axis=0)).ravel().astype(np.float64)
    scale = np.power(np.maximum(counts, 1.0), -item_damping)
    scaled = (history @ sparse.diags(scale)).astype(np.float64)
    k = min(max_factors, min(scaled.shape) - 1)
    v0 = np.random.default_rng(seed).standard_normal(min(scaled.shape))
    _, s, vt = svds(scaled, k=k, v0=v0)
    order = np.argsort(-s)
    return LatentFactors(vt[order].T.astype(np.float32), s[order], scale.astype(np.float32))


class LatentFactorRecommender:
    def __init__(self, factors: LatentFactors, n_factors: int):
        self.item_factors = factors.item_factors[:, :n_factors]
        self.column_scale = factors.column_scale
        self.n_factors = n_factors

    def recommend(self, history: sparse.csr_matrix, user_positions: np.ndarray, n: int,
                  exclude_seen: bool = True, chunk_size: int = 1000) -> np.ndarray:
        """Top-n item columns per user, processed in chunks to bound memory."""
        out = np.empty((len(user_positions), n), dtype=np.int64)
        for start in range(0, len(user_positions), chunk_size):
            positions = user_positions[start:start + chunk_size]
            rows = history[positions]
            projected = (rows @ sparse.diags(self.column_scale)) @ self.item_factors
            scores = np.asarray(projected @ self.item_factors.T, dtype=np.float32)
            if exclude_seen:
                coo = rows.tocoo()
                scores[coo.row, coo.col] = -np.inf
            top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
            out[start:start + len(positions)] = np.take_along_axis(top, order, axis=1)
        return out
