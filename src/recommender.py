"""Item-based neighbourhood collaborative filtering (cosine similarity).

Score of item j for user u = sum over items i in u's history of w(u, i) * sim(i, j),
where sim keeps only the k most similar items of each i and w is 1 (binary) or a
recency weight. Items the user already bought are excluded by default, and users with
too few candidates are topped up from a popularity ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class NeighbourTable:
    """For every item, its ``k_max`` most similar items (most similar first)."""
    indices: np.ndarray     # (n_items, k_max) int32 item columns
    similarity: np.ndarray  # (n_items, k_max) float32

    def top_k(self, k: int) -> sparse.csr_matrix:
        """Sparse similarity matrix keeping each item's ``k`` nearest neighbours (rows = source item)."""
        k = min(k, self.indices.shape[1])
        n = self.indices.shape[0]
        data = self.similarity[:, :k].ravel()
        keep = data > 0
        rows = np.repeat(np.arange(n), k)[keep]
        return sparse.csr_matrix((data[keep], (rows, self.indices[:, :k].ravel()[keep])), shape=(n, n))


def build_neighbour_table(train_matrix: sparse.csr_matrix, k_max: int = 200, shrink: float = 0.0,
                          block_size: int = 1000) -> NeighbourTable:
    """Cosine similarity between item columns of a binary user-item matrix.

    similarity(i, j) = co-buyers(i, j) / (sqrt(n_i) * sqrt(n_j) + shrink); the shrink term
    lowers similarities that rest on few co-buyers. Computed block by block so the full
    item x item matrix is never held in memory.
    """
    X = train_matrix.tocsc().astype(np.float32)
    n_items = X.shape[1]
    k_max = min(k_max, n_items - 1)
    norms = np.sqrt(np.asarray(X.sum(axis=0)).ravel())  # binary matrix: sqrt of buyer count
    safe = np.where(norms == 0, 1.0, norms).astype(np.float32)
    indices = np.empty((n_items, k_max), dtype=np.int32)
    similarity = np.empty((n_items, k_max), dtype=np.float32)
    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        co = (X[:, start:stop].T @ X).toarray()
        sim = co / (np.outer(safe[start:stop], safe) + shrink)
        sim[np.arange(stop - start), np.arange(start, stop)] = 0.0   # no self-similarity
        top = np.argpartition(-sim, k_max - 1, axis=1)[:, :k_max]
        top_sim = np.take_along_axis(sim, top, axis=1)
        order = np.argsort(-top_sim, axis=1, kind="stable")
        indices[start:stop] = np.take_along_axis(top, order, axis=1)
        similarity[start:stop] = np.take_along_axis(top_sim, order, axis=1)
    return NeighbourTable(indices, similarity)


class ItemKNN:
    """Item-item collaborative filtering on top of a neighbour table."""

    def __init__(self, table: NeighbourTable, k: int):
        self.k = k
        self.similarity = table.top_k(k)
        self._similarity_csc: sparse.csc_matrix | None = None

    def score(self, history: sparse.csr_matrix) -> sparse.csr_matrix:
        """Sparse (n_users, n_items) scores for the rows of ``history``."""
        return (history @ self.similarity).tocsr()

    def recommend(self, history: sparse.csr_matrix, user_positions: np.ndarray, n: int,
                  fallback_ranking: np.ndarray, exclude_seen: bool = True) -> np.ndarray:
        """Top-n item columns per user; ``fallback_ranking`` (best first) fills short lists."""
        scores = self.score(history[user_positions])
        out = np.empty((len(user_positions), n), dtype=np.int64)
        for r, u in enumerate(user_positions):
            lo, hi = scores.indptr[r], scores.indptr[r + 1]
            cand, val = scores.indices[lo:hi], scores.data[lo:hi]
            seen = history.indices[history.indptr[u]:history.indptr[u + 1]]
            if exclude_seen:
                keep = ~np.isin(cand, seen)
                cand, val = cand[keep], val[keep]
            if len(cand) > n:
                top = np.argpartition(-val, n - 1)[:n]
            else:
                top = np.arange(len(cand))
            picks = [int(x) for x in cand[top[np.argsort(-val[top], kind="stable")]]]
            if len(picks) < n:
                taken = set(picks) | (set(seen.tolist()) if exclude_seen else set())
                for item in fallback_ranking:
                    if item not in taken:
                        picks.append(int(item))
                        taken.add(item)
                        if len(picks) == n:
                            break
            out[r] = picks
        return out

    def explain(self, history: sparse.csr_matrix, user_position: int, item: int,
                top: int = 2) -> list[tuple[int, float]]:
        """Items in the user's history that contributed most to recommending ``item``."""
        if self._similarity_csc is None:
            self._similarity_csc = self.similarity.tocsc()
        col = self._similarity_csc[:, item]
        row = history[user_position]
        common, a, b = np.intersect1d(col.indices, row.indices, return_indices=True)
        contribution = col.data[a] * row.data[b]
        order = np.argsort(-contribution, kind="stable")[:top]
        return [(int(common[o]), float(contribution[o])) for o in order]


def diversify_by_product(candidates: np.ndarray, product_of_item: np.ndarray, n: int) -> np.ndarray:
    """First n candidates per user with at most one article per product (colour variants collapsed).

    If a list has fewer than n distinct products, it is topped up with the skipped
    variants in their original order so the output keeps its shape.
    """
    out = np.empty((len(candidates), n), dtype=np.int64)
    for r, row in enumerate(candidates):
        picked, skipped, seen = [], [], set()
        for item in row:
            product = product_of_item[item]
            if product in seen:
                skipped.append(int(item))
            else:
                seen.add(product)
                picked.append(int(item))
            if len(picked) == n:
                break
        out[r] = (picked + skipped)[:n]
    return out
