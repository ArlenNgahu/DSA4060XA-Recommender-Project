"""Top-K ranking metrics and beyond-accuracy measures for recommender evaluation.

All functions take recommendations as an (n_users, k) array of item column indices and
the held-out interactions as a binary CSR matrix over the same user/item index space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _hits(recs: np.ndarray, truth_rows: sparse.csr_matrix) -> np.ndarray:
    n, k = recs.shape
    flat = truth_rows[np.repeat(np.arange(n), k), recs.ravel()]
    return np.asarray(flat).reshape(n, k) > 0


def precision_recall_ndcg(recs: np.ndarray, truth_rows: sparse.csr_matrix) -> pd.DataFrame:
    """Per-user Precision@k, Recall@k and NDCG@k (binary relevance)."""
    n, k = recs.shape
    hits = _hits(recs, truth_rows)
    n_rel = truth_rows.getnnz(axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.array([discounts[: min(k, r)].sum() for r in n_rel])
    return pd.DataFrame({
        "precision": hits.sum(axis=1) / k,
        "recall": hits.sum(axis=1) / np.maximum(n_rel, 1),
        "ndcg": np.divide(dcg, ideal, out=np.zeros(n), where=ideal > 0),
    })


def catalog_coverage(recs: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one recommendation list."""
    return np.unique(recs).size / n_items


def novelty(recs: np.ndarray, item_user_share: np.ndarray) -> float:
    """Mean self-information (bits) of recommended items; higher = less popular items."""
    share = np.clip(item_user_share[recs], 1e-12, 1.0)
    return float((-np.log2(share)).mean())


def category_diversity(recs: np.ndarray, category_of_item: np.ndarray) -> float:
    """Mean share of distinct categories in a list (1.0 = every item from a different category)."""
    ordered = np.sort(category_of_item[recs], axis=1)
    distinct = 1 + (np.diff(ordered, axis=1) != 0).sum(axis=1)
    return float((distinct / recs.shape[1]).mean())


def mean_popularity(recs: np.ndarray, item_user_share: np.ndarray) -> float:
    """Average share of training users who bought the recommended items (popularity bias)."""
    return float(item_user_share[recs].mean())


def evaluate_topk(recs: np.ndarray, truth: sparse.csr_matrix, user_positions: np.ndarray,
                  item_user_share: np.ndarray, n_items: int,
                  category_of_item: np.ndarray | None = None) -> pd.Series:
    """Mean metrics over ``user_positions`` (rows of ``truth``) with at least one relevant item."""
    k = recs.shape[1]
    per_user = precision_recall_ndcg(recs, truth[user_positions])
    out = {
        f"Precision@{k}": per_user["precision"].mean(),
        f"Recall@{k}": per_user["recall"].mean(),
        f"NDCG@{k}": per_user["ndcg"].mean(),
        "Coverage": catalog_coverage(recs, n_items),
        "Novelty (bits)": novelty(recs, item_user_share),
        "Mean popularity": mean_popularity(recs, item_user_share),
    }
    if category_of_item is not None:
        out["Category diversity"] = category_diversity(recs, category_of_item)
    out["users evaluated"] = len(user_positions)
    return pd.Series(out)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """Mean difference a - b of per-user scores with a bootstrap 95% interval over users."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean_difference": float(diff.mean()), "ci_low": float(low), "ci_high": float(high),
            "significant": bool(low > 0 or high < 0)}
