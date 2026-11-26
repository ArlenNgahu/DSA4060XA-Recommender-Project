"""Outfit completion: given one anchor item, suggest items bought together with it.

Items that appear in the same customer-day basket are treated as an outfit. Item
similarity is cosine similarity across baskets (two items are close if they are bought
in the same baskets), computed with the same blockwise routine as item-item CF. At
recommendation time candidates of the anchor's own product type and product are removed,
because a complement is a *different* kind of item.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src import config
from src.recommender import NeighbourTable, build_neighbour_table


def basket_matrix(df: pd.DataFrame, users: pd.Index, items: pd.Index,
                  min_size: int = config.OUTFIT_BASKET_SIZES[0],
                  max_size: int = config.OUTFIT_BASKET_SIZES[1]) -> sparse.csr_matrix:
    """(n_baskets, n_items) binary matrix of same-customer same-day baskets with min..max items."""
    cust = df["customer_id"].cat.codes.to_numpy(np.int64)
    day = (df["t_dat"] - df["t_dat"].min()).dt.days.to_numpy(np.int64)
    cols = items.get_indexer(df["article_id"])
    ok = cols >= 0
    key = (cust * 100_000 + day)[ok]
    cols = cols[ok]
    _, basket = np.unique(key, return_inverse=True)
    size = np.bincount(basket)
    keep = (size[basket] >= min_size) & (size[basket] <= max_size)
    _, compact = np.unique(basket[keep], return_inverse=True)
    n_baskets = int(compact.max()) + 1 if keep.any() else 0
    return sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32), (compact, cols[keep])),
                             shape=(n_baskets, len(items)))


def fit_outfit_table(baskets: sparse.csr_matrix, k_max: int = config.OUTFIT_NEIGHBOURS,
                     shrink: float = 0.0) -> NeighbourTable:
    return build_neighbour_table(baskets, k_max=k_max, shrink=shrink)


def complement_candidates(table: NeighbourTable, anchors: np.ndarray, category_of_item: np.ndarray,
                          product_of_item: np.ndarray, n: int, fallback_ranking: np.ndarray) -> np.ndarray:
    """Top-n complements per anchor, never sharing the anchor's product type or product.

    Short lists are completed from ``fallback_ranking`` (e.g. trending items) under the same rule.
    """
    out = np.empty((len(anchors), n), dtype=np.int64)
    for r, a in enumerate(anchors):
        cand, sim = table.indices[a], table.similarity[a]
        ok = (sim > 0) & (category_of_item[cand] != category_of_item[a]) & (product_of_item[cand] != product_of_item[a])
        picks = [int(x) for x in cand[ok][:n]]
        if len(picks) < n:
            taken = set(picks)
            for item in fallback_ranking:
                if (item not in taken and category_of_item[item] != category_of_item[a]
                        and product_of_item[item] != product_of_item[a]):
                    picks.append(int(item))
                    taken.add(item)
                    if len(picks) == n:
                        break
        out[r] = picks
    return out


def anchor_targets(baskets: sparse.csr_matrix, category_of_item: np.ndarray, max_pairs: int,
                   seed: int = config.RANDOM_SEED) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Sample (basket, anchor) pairs; the targets are the other items in that basket of a different type.

    Returns (anchor item per pair, binary (n_pairs, n_items) matrix of target items). Pairs without
    any different-type target are dropped.
    """
    pairs = []
    for b in range(baskets.shape[0]):
        members = baskets.indices[baskets.indptr[b]:baskets.indptr[b + 1]]
        for a in members:
            targets = members[category_of_item[members] != category_of_item[a]]
            if len(targets):
                pairs.append((a, targets))
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    anchors = np.array([a for a, _ in pairs], dtype=np.int64)
    rows = np.concatenate([np.full(len(t), i) for i, (_, t) in enumerate(pairs)]) if pairs else np.array([], dtype=int)
    cols = np.concatenate([t for _, t in pairs]) if pairs else np.array([], dtype=int)
    truth = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(len(pairs), baskets.shape[1]))
    return anchors, truth
