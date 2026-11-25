"""Content features, repurchase ranking, outfit affinity and weighted rank fusion.

A hybrid recommendation is built in two steps:
1. every component (popularity, item-item CF, latent factors, content similarity,
   repurchase) produces its own top-N list per user, converted into reciprocal-rank
   scores 1 / (c + rank);
2. the component scores are summed with weights tuned on validation, optionally
   re-ranked by an outfit affinity multiplier, and the top K items are returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize

from src import config, data_audit as da, prepare as prep
from src.latent import LatentFactorRecommender, LatentFactors, fit_puresvd
from src.recommender import ItemKNN, NeighbourTable, build_neighbour_table

CATEGORICAL_COLUMNS = ["product_type_name", "product_group_name", "graphical_appearance_name",
                       "colour_group_name", "perceived_colour_master_name", "department_name",
                       "index_name", "section_name", "garment_group_name"]


# ---------------------------------------------------------------- content features
def build_item_features(articles: pd.DataFrame, categorical_weight: float = 1.0,
                        max_text_features: int = 5000) -> sparse.csr_matrix:
    """L2-normalised item vectors: TF-IDF of name and description plus weighted one-hot attributes.

    Rows follow the row order of ``articles``.
    """
    text = (articles["prod_name"].fillna("") + " " + articles["detail_desc"].fillna("")).str.lower()
    tfidf = TfidfVectorizer(max_features=max_text_features, ngram_range=(1, 2), sublinear_tf=True,
                            min_df=2, stop_words="english").fit_transform(text)
    categorical = OneHotEncoder(handle_unknown="ignore").fit_transform(articles[CATEGORICAL_COLUMNS].fillna("unknown"))
    tfidf = normalize(tfidf)
    categorical = normalize(categorical) * categorical_weight
    return normalize(sparse.hstack([tfidf, categorical]).tocsr()).astype(np.float32)


def feature_neighbour_table(features: sparse.csr_matrix, k_max: int = 50, block_size: int = 1000) -> NeighbourTable:
    """Cosine nearest neighbours between items using content features (rows are L2-normalised)."""
    n = features.shape[0]
    k_max = min(k_max, n - 1)
    transposed = features.T.tocsc()
    indices = np.empty((n, k_max), dtype=np.int32)
    similarity = np.empty((n, k_max), dtype=np.float32)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sim = (features[start:stop] @ transposed).toarray()
        sim[np.arange(stop - start), np.arange(start, stop)] = 0.0
        top = np.argpartition(-sim, k_max - 1, axis=1)[:, :k_max]
        top_sim = np.take_along_axis(sim, top, axis=1)
        order = np.argsort(-top_sim, axis=1, kind="stable")
        indices[start:stop] = np.take_along_axis(top, order, axis=1)
        similarity[start:stop] = np.take_along_axis(top_sim, order, axis=1)
    return NeighbourTable(indices, similarity)


# ------------------------------------------------------------------- repurchase
def repurchase_ranking(history: sparse.csr_matrix, user_positions: np.ndarray, n: int) -> np.ndarray:
    """A user's own past items, highest (recency-weighted) score first; padded with -1."""
    out = np.full((len(user_positions), n), -1, dtype=np.int64)
    for r, u in enumerate(user_positions):
        lo, hi = history.indptr[u], history.indptr[u + 1]
        items, weights = history.indices[lo:hi], history.data[lo:hi]
        order = np.argsort(-weights, kind="stable")[:n]
        out[r, :len(order)] = items[order]
    return out


# ----------------------------------------------------------------- rank fusion
def rrf_matrix(ranked: np.ndarray, n_items: int, c: float = config.RRF_C) -> sparse.csr_matrix:
    """Sparse (n_users, n_items) matrix of reciprocal-rank scores; -1 entries are ignored."""
    n_users, depth = ranked.shape
    rows = np.repeat(np.arange(n_users), depth)
    cols = ranked.ravel()
    values = np.tile(1.0 / (c + np.arange(1, depth + 1)), n_users)
    keep = cols >= 0
    return sparse.csr_matrix((values[keep].astype(np.float32), (rows[keep], cols[keep])),
                             shape=(n_users, n_items))


def fuse(parts: dict[str, sparse.csr_matrix], weights: dict[str, float]) -> sparse.csr_matrix:
    """Weighted sum of component score matrices (components with weight 0 are skipped)."""
    active = [(weights[name] * parts[name]) for name in parts if weights.get(name, 0) > 0]
    if not active:
        raise ValueError("at least one component needs a positive weight")
    total = active[0]
    for part in active[1:]:
        total = total + part
    return total.tocsr()


def select_top_k(fused: sparse.csr_matrix, k: int, fallback_ranking: np.ndarray,
                 seen: Optional[sparse.csr_matrix] = None, user_positions: Optional[np.ndarray] = None) -> np.ndarray:
    """Top-k columns of each row; short rows are filled from ``fallback_ranking`` (unseen items)."""
    out = np.empty((fused.shape[0], k), dtype=np.int64)
    for r in range(fused.shape[0]):
        lo, hi = fused.indptr[r], fused.indptr[r + 1]
        cols, vals = fused.indices[lo:hi], fused.data[lo:hi]
        if len(cols) > k:
            top = np.argpartition(-vals, k - 1)[:k]
        else:
            top = np.arange(len(cols))
        picks = [int(x) for x in cols[top[np.argsort(-vals[top], kind="stable")]]]
        if len(picks) < k:
            taken = set(picks)
            if seen is not None and user_positions is not None:
                u = user_positions[r]
                taken |= set(seen.indices[seen.indptr[u]:seen.indptr[u + 1]].tolist())
            for item in fallback_ranking:
                if item not in taken:
                    picks.append(int(item))
                    taken.add(item)
                    if len(picks) == k:
                        break
        out[r] = picks
    return out


# ---------------------------------------------------------------- outfit affinity
def log_lift_matrix(train_df: pd.DataFrame, articles: pd.DataFrame, category_names: np.ndarray,
                    min_size: int = config.LIFT_BASKET_SIZES[0], max_size: int = config.LIFT_BASKET_SIZES[1],
                    min_support: int = config.OUTFIT_MIN_SUPPORT) -> np.ndarray:
    """log(lift) between product types in small same-day baskets, in the order of ``category_names``.

    Pairs with too little support, and the diagonal, get 0 (no evidence), and values are
    clipped to [-2, 2] so a few extreme pairs cannot dominate.
    """
    baskets, cooc = da.basket_analysis(train_df, articles, "product_type_name", min_size=min_size, max_size=max_size)
    n_used = int(((baskets["basket_size"] >= min_size) & (baskets["basket_size"] <= max_size)).sum())
    lift = da.lift_matrix(cooc, max(n_used, 1), min_support=min_support).reindex(index=category_names, columns=category_names)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lift = np.log(lift.to_numpy().astype(float))
    return np.nan_to_num(np.clip(log_lift, -2, 2), nan=0.0, posinf=0.0, neginf=0.0)


def last_basket_profile(train_df: pd.DataFrame, users: pd.Index, items: pd.Index,
                        category_of_item: np.ndarray, n_categories: int) -> sparse.csr_matrix:
    """(n_users, n_categories) share of each product type in the user's most recent purchase day."""
    last_day = train_df.groupby("customer_id", observed=True)["t_dat"].transform("max")
    last = train_df[train_df["t_dat"] == last_day]
    rows = users.get_indexer(last["customer_id"].astype(object))
    cols = category_of_item[items.get_indexer(last["article_id"])]
    ok = (rows >= 0) & (cols >= 0)
    counts = sparse.coo_matrix((np.ones(int(ok.sum()), dtype=np.float32), (rows[ok], cols[ok])),
                               shape=(len(users), n_categories)).tocsr()
    return normalize(counts, norm="l1")


def apply_outfit_affinity(fused: sparse.csr_matrix, profile: sparse.csr_matrix, log_lift: np.ndarray,
                          category_of_item: np.ndarray, user_positions: np.ndarray, gamma: float) -> sparse.csr_matrix:
    """Multiply each score by exp(gamma * affinity), where affinity is the mean log-lift between
    the item's product type and the product types of the user's latest basket."""
    if gamma == 0:
        return fused
    affinity = np.asarray((profile[user_positions] @ log_lift))
    rows = np.repeat(np.arange(fused.shape[0]), np.diff(fused.indptr))
    out = fused.copy()
    out.data = out.data * np.exp(gamma * affinity[rows, category_of_item[out.indices]]).astype(np.float32)
    return out


# ------------------------------------------------------------------ orchestration
@dataclass(frozen=True)
class HybridParams:
    pop_half_life: Optional[float]
    knn_k: int
    knn_shrink: float
    knn_history_half_life: Optional[float]
    svd_factors: int
    svd_damping: float
    svd_history_half_life: Optional[float]
    content_k: int
    repurchase_half_life: float
    n_candidates: int = 100
    rrf_c: float = config.RRF_C


class HybridRecommender:
    """Fits every component on one training table and serves component rank matrices."""

    def __init__(self, params: HybridParams, users: pd.Index, items: pd.Index,
                 category_of_item: np.ndarray, category_names: np.ndarray,
                 content_neighbours: NeighbourTable, log_lift: np.ndarray):
        self.params, self.users, self.items = params, users, items
        self.category_of_item, self.category_names = category_of_item, category_names
        self._content_table, self.log_lift = content_neighbours, log_lift

    def _history(self, df: pd.DataFrame, half_life: Optional[float]) -> sparse.csr_matrix:
        if half_life is None:
            return self.seen
        return prep.build_interaction_matrix(df, self.users, self.items, "recency",
                                             half_life_days=half_life, ref_date=self._ref_date)

    def fit(self, train_df: pd.DataFrame) -> "HybridRecommender":
        p = self.params
        self._ref_date = train_df["t_dat"].max()
        self.seen = prep.build_interaction_matrix(train_df, self.users, self.items, "binary")
        pop_scores = prep.popularity_scores(train_df, self.items, p.pop_half_life)
        self.pop_scores = pop_scores
        self.pop_ranking = np.argsort(-pop_scores, kind="stable")
        self.knn_history = self._history(train_df, p.knn_history_half_life)
        self.knn = ItemKNN(build_neighbour_table(self.seen, k_max=p.knn_k, shrink=p.knn_shrink), p.knn_k)
        self.svd_history = self._history(train_df, p.svd_history_half_life)
        self.svd = LatentFactorRecommender(fit_puresvd(self.svd_history, p.svd_factors, p.svd_damping), p.svd_factors)
        self.content = ItemKNN(self._content_table, p.content_k)
        self.content_history = self._history(train_df, 28)
        self.repurchase_history = self._history(train_df, p.repurchase_half_life)
        self.profile = last_basket_profile(train_df, self.users, self.items, self.category_of_item, len(self.category_names))
        return self

    def component_ranks(self, user_positions: np.ndarray) -> dict[str, sparse.csr_matrix]:
        """Reciprocal-rank score matrices (one row per user in ``user_positions``) for every component."""
        p, n_items = self.params, len(self.items)
        n = p.n_candidates
        lists = {
            "popularity": prep.recommend_popular(self.pop_scores, self.seen, user_positions, n),
            "popularity_incl_seen": prep.recommend_popular(self.pop_scores, self.seen, user_positions, n, exclude_seen=False),
            "item_knn": self.knn.recommend(self.knn_history, user_positions, n, self.pop_ranking),
            "svd": self.svd.recommend(self.svd_history, user_positions, n),
            "content": self.content.recommend(self.content_history, user_positions, n, self.pop_ranking),
            "repurchase": repurchase_ranking(self.repurchase_history, user_positions, n),
        }
        return {name: rrf_matrix(ranked, n_items, p.rrf_c) for name, ranked in lists.items()}

    def recommend(self, ranks: dict[str, sparse.csr_matrix], weights: dict[str, float], user_positions: np.ndarray,
                  k: int, gamma: float = 0.0) -> np.ndarray:
        fused = fuse(ranks, weights)
        fused = apply_outfit_affinity(fused, self.profile, self.log_lift, self.category_of_item, user_positions, gamma)
        return select_top_k(fused, k, self.pop_ranking, self.seen, user_positions)


COMPONENT_LABELS = {
    "popularity": "trending right now",
    "item_knn": "customers who bought your items also bought this",
    "svd": "matches your taste profile",
    "content": "similar in style to what you bought",
    "repurchase": "you bought this before",
}


def component_contributions(ranks: dict[str, sparse.csr_matrix], weights: dict[str, float],
                            row: int, item: int) -> dict[str, float]:
    """Weighted reciprocal-rank score each component contributed to ``item`` for one user (row)."""
    return {name: float(weights.get(name, 0.0) * ranks[name][row, item])
            for name in ranks if weights.get(name, 0.0) > 0 and ranks[name][row, item] > 0}
