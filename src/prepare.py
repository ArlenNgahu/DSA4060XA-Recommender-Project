"""Week 2 data preparation: collapsing, filtering, sampling, splitting, matrices, baseline.

Pipeline (all steps are deterministic given the seed):
    raw transactions -> collapse same-day repeats -> time window -> eligible users ->
    seeded user sample -> k-core filter (min interactions per user and per item) ->
    cleaned interactions -> time-based train/validation/test split -> sparse matrices.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from src import config

_ARTICLE_BITS, _DAY_BITS = 30, 10


# ------------------------------------------------------------------ specification
@dataclass(frozen=True)
class PrepSpec:
    """Filtering choices. All thresholds are measured on the TRAINING period only."""
    window_days: Optional[int] = config.WINDOW_DAYS
    min_user_interactions: int = config.MIN_USER_INTERACTIONS   # distinct items per user
    min_item_interactions: int = config.MIN_ITEM_INTERACTIONS   # distinct buyers per item
    sample_users: Optional[int] = config.SAMPLE_USERS
    seed: int = config.RANDOM_SEED
    val_days: int = config.VAL_DAYS
    test_days: int = config.TEST_DAYS


def split_dates(last_day: pd.Timestamp, val_days: int, test_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(validation start, test start); train < val_start <= val < test_start <= test."""
    test_start = last_day.normalize() - pd.Timedelta(days=test_days - 1)
    return test_start - pd.Timedelta(days=val_days), test_start


# ------------------------------------------------------------------- collapsing
def collapse_same_day(df: pd.DataFrame) -> pd.DataFrame:
    """Merge rows with the same customer, article and date into one interaction.

    Adds ``n_units`` (how many rows were merged). Keeps the first row's price and channel.
    Chronological order is preserved.
    """
    df = df[df["customer_id"].notna()]
    users = df["customer_id"].cat.codes.to_numpy(np.int64)
    articles = df["article_id"].to_numpy(np.int64)
    days = (df["t_dat"] - df["t_dat"].min()).dt.days.to_numpy(np.int64)
    if articles.max() >= 2 ** _ARTICLE_BITS or days.max() >= 2 ** _DAY_BITS:
        raise ValueError("article ids or date range exceed the packed-key capacity")
    key = (users << (_ARTICLE_BITS + _DAY_BITS)) | (days << _ARTICLE_BITS) | articles
    _, first, counts = np.unique(key, return_index=True, return_counts=True)
    order = np.argsort(first)
    out = df.iloc[first[order]].copy()
    out["n_units"] = counts[order].astype(np.int16)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------- k-core
def kcore_mask(users: np.ndarray, items: np.ndarray, min_user: int, min_item: int,
               n_users: int, n_items: int, max_iter: int = 100) -> np.ndarray:
    """Boolean mask of rows that survive iterated user/item minimum-count filtering."""
    keep = np.ones(len(users), dtype=bool)
    for _ in range(max_iter):
        user_counts = np.bincount(users[keep], minlength=n_users)
        item_counts = np.bincount(items[keep], minlength=n_items)
        new = keep & (user_counts[users] >= min_user) & (item_counts[items] >= min_item)
        if new.sum() == keep.sum():
            return new
        keep = new
    return keep


class _Arrays:
    """Integer views of a collapsed interaction table, built once and reused."""

    def __init__(self, df: pd.DataFrame):
        self.t0 = df["t_dat"].min()
        self.last_day = df["t_dat"].max().normalize()
        self.users = df["customer_id"].cat.codes.to_numpy(np.int64)
        items, self.item_values = pd.factorize(df["article_id"].to_numpy())
        self.items = items.astype(np.int64)
        self.days = (df["t_dat"] - self.t0).dt.days.to_numpy(np.int64)
        self.n_users = int(self.users.max()) + 1
        self.n_items = len(self.item_values)

    def day_index(self, timestamp: pd.Timestamp) -> int:
        return int((timestamp - self.t0).days)

    def train_pairs(self, window_start_day: int, val_start_day: int) -> tuple[np.ndarray, np.ndarray]:
        """Distinct (user, item) pairs seen in the training period of the window."""
        rows = np.flatnonzero((self.days >= window_start_day) & (self.days < val_start_day))
        u, i = self.users[rows], self.items[rows]
        _, first = np.unique(u * (2 ** 31) + i, return_index=True)
        return u[first], i[first]

    def universe(self, pairs: tuple[np.ndarray, np.ndarray], min_user: int, min_item: int,
                 sample_users: Optional[int], seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """Users and items that survive sampling and k-core filtering on distinct training pairs."""
        pu, pi = pairs
        counts = np.bincount(pu, minlength=self.n_users)
        eligible = np.flatnonzero(counts >= min_user)
        if sample_users is not None and len(eligible) > sample_users:
            chosen = np.random.default_rng(seed).choice(eligible, sample_users, replace=False)
        else:
            chosen = eligible
        lookup = np.zeros(self.n_users, dtype=bool)
        lookup[chosen] = True
        sel = lookup[pu]
        pu, pi = pu[sel], pi[sel]
        keep = kcore_mask(pu, pi, min_user, min_item, self.n_users, self.n_items)
        user_mask = np.zeros(self.n_users, dtype=bool)
        item_mask = np.zeros(self.n_items, dtype=bool)
        user_mask[pu[keep]] = True
        item_mask[pi[keep]] = True
        info = {"users_in_train": int((counts > 0).sum()), "users_eligible": int(len(eligible)),
                "users_sampled": int(len(chosen))}
        return user_mask, item_mask, info


def _unique_pairs(users: np.ndarray, items: np.ndarray) -> int:
    return int(np.unique(users * (2 ** 31) + items).size)


@dataclass
class PrepResult:
    cleaned: pd.DataFrame      # all periods; users and items from the training-period universe
    cold_items: pd.DataFrame   # validation/test rows of universe users on items outside the universe
    report: pd.DataFrame
    meta: dict


def prepare_working_set(df: pd.DataFrame, spec: PrepSpec = PrepSpec()) -> PrepResult:
    """Leakage-safe filtering.

    Thresholds (distinct items per user, distinct buyers per item) are measured on the
    training period only. The surviving users and items form a fixed universe; every
    window row inside that universe is kept, including validation and test rows, so no
    future information decides which users or items exist. Validation/test rows of
    universe users on items outside the universe are returned separately as ``cold_items``.
    """
    arrays = _Arrays(df)
    val_start, test_start = split_dates(arrays.last_day, spec.val_days, spec.test_days)
    window_start_day = 0 if spec.window_days is None else arrays.day_index(arrays.last_day) - spec.window_days + 1
    val_day = arrays.day_index(val_start)
    pairs = arrays.train_pairs(window_start_day, val_day)
    user_mask, item_mask, info = arrays.universe(
        pairs, spec.min_user_interactions, spec.min_item_interactions, spec.sample_users, spec.seed)

    in_window = arrays.days >= window_start_day
    in_user = user_mask[arrays.users]
    final = np.flatnonzero(in_window & in_user & item_mask[arrays.items])
    cold = np.flatnonzero(in_window & (arrays.days >= val_day) & in_user & ~item_mask[arrays.items])
    train_rows = np.flatnonzero(in_window & (arrays.days < val_day))

    def count(idx):
        return {"interactions": len(idx), "users": int(np.unique(arrays.users[idx]).size),
                "items": int(np.unique(arrays.items[idx]).size)}

    report = pd.DataFrame([
        {"step": "collapsed input", **count(np.arange(len(df)))},
        {"step": f"time window ({spec.window_days or 'all'} days)", **count(np.flatnonzero(in_window))},
        {"step": f"training period of the window (before {val_start.date()})", **count(train_rows)},
        {"step": f"users with >= {spec.min_user_interactions} distinct training items",
         "interactions": np.nan, "users": info["users_eligible"], "items": np.nan},
        {"step": f"universe: sample <= {spec.sample_users} users, k-core "
                 f"(user>={spec.min_user_interactions} items, item>={spec.min_item_interactions} buyers)",
         "interactions": np.nan, "users": int(user_mask.sum()), "items": int(item_mask.sum())},
        {"step": "cleaned interactions (all periods, inside universe)", **count(final)},
        {"step": "cold-item interactions (val/test, items outside universe)", **count(cold)},
    ]).set_index("step")

    def finish(idx):
        out = df.iloc[idx].copy()
        out["customer_id"] = out["customer_id"].cat.remove_unused_categories()
        return out.reset_index(drop=True)

    meta = {"spec": asdict(spec), "last_day": str(arrays.last_day.date()),
            "val_start": str(val_start.date()), "test_start": str(test_start.date())}
    return PrepResult(finish(final), finish(cold), report, meta)


def sensitivity_table(df: pd.DataFrame, windows: dict[str, Optional[int]], min_users=(1, 5, 10),
                      min_items=(1, 20), sample_users: Optional[int] = config.SAMPLE_USERS,
                      seed: int = config.RANDOM_SEED, val_days: int = config.VAL_DAYS,
                      test_days: int = config.TEST_DAYS) -> pd.DataFrame:
    """Size of the resulting dataset for each window / threshold combination (same pipeline)."""
    arrays = _Arrays(df)
    val_start, _ = split_dates(arrays.last_day, val_days, test_days)
    val_day, last_idx = arrays.day_index(val_start), arrays.day_index(arrays.last_day)
    records = []
    for label, days in windows.items():
        start = 0 if days is None else last_idx - days + 1
        pairs = arrays.train_pairs(start, val_day)
        in_window = arrays.days >= start
        for mu in min_users:
            for mi in min_items:
                user_mask, item_mask, info = arrays.universe(pairs, mu, mi, sample_users, seed)
                rows = np.flatnonzero(in_window & user_mask[arrays.users] & item_mask[arrays.items])
                n_u, n_i = int(user_mask.sum()), int(item_mask.sum())
                distinct = _unique_pairs(arrays.users[rows], arrays.items[rows]) if len(rows) else 0
                records.append({
                    "window": label, "min_user": mu, "min_item": mi,
                    "users_in_train": info["users_in_train"], "users_eligible": info["users_eligible"],
                    "users": n_u, "items": n_i, "interactions": len(rows),
                    "interactions_per_user": len(rows) / n_u if n_u else np.nan,
                    "sparsity": 1 - distinct / (n_u * n_i) if n_u and n_i else np.nan,
                })
    return pd.DataFrame(records)


# ------------------------------------------------------------------- persistence
def save_prepared(result: PrepResult, directory=None) -> None:
    directory = directory or config.CLEANED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in (("interactions", result.cleaned), ("interactions_cold_items", result.cold_items)):
        out = frame.copy()
        out["customer_id"] = out["customer_id"].astype(str)
        out.to_parquet(directory / f"{name}.parquet", index=False)
    (directory / "prepared_meta.json").write_text(json.dumps(result.meta, indent=2))


def load_prepared(directory=None) -> PrepResult:
    directory = directory or config.CLEANED_DIR
    frames = []
    for name in ("interactions", "interactions_cold_items"):
        frame = pd.read_parquet(directory / f"{name}.parquet")
        frame["customer_id"] = frame["customer_id"].astype("category")
        frames.append(frame)
    meta = json.loads((directory / "prepared_meta.json").read_text())
    return PrepResult(frames[0], frames[1], pd.DataFrame(), meta)


# --------------------------------------------------------------------- splitting
@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_start: pd.Timestamp
    test_start: pd.Timestamp


def time_split(df: pd.DataFrame, val_start: pd.Timestamp, test_start: pd.Timestamp) -> Split:
    """Chronological split: train < val_start <= val < test_start <= test."""
    return Split(train=df[df["t_dat"] < val_start], val=df[(df["t_dat"] >= val_start) & (df["t_dat"] < test_start)],
                 test=df[df["t_dat"] >= test_start], val_start=val_start, test_start=test_start)


def split_report(split: Split, cold_items: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    train_users, train_items = set(split.train["customer_id"].unique()), set(split.train["article_id"].unique())
    rows = []
    for name, part in (("train", split.train), ("validation", split.val), ("test", split.test)):
        users = part["customer_id"].unique()
        row = {"part": name, "from": part["t_dat"].min(), "to": part["t_dat"].max(),
               "interactions": len(part), "users": len(users), "items": part["article_id"].nunique(),
               "users also in train": np.mean([u in train_users for u in users]) if name != "train" else np.nan,
               "interactions on items not in train":
                   (~part["article_id"].isin(train_items)).mean() if name != "train" else np.nan}
        if cold_items is not None and name != "train":
            cold = cold_items[(cold_items["t_dat"] >= part["t_dat"].min()) & (cold_items["t_dat"] <= part["t_dat"].max())]
            row["cold-item interactions excluded"] = len(cold)
        rows.append(row)
    return pd.DataFrame(rows).set_index("part")


# -------------------------------------------------------------------- matrices
def make_index(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Fixed user and item index spaces shared by every matrix in the project."""
    users = pd.Index(df["customer_id"].cat.remove_unused_categories().cat.categories)
    items = pd.Index(np.sort(df["article_id"].unique()))
    return users, items


def build_interaction_matrix(df: pd.DataFrame, users: pd.Index, items: pd.Index,
                             weighting: str = "binary", half_life_days: Optional[float] = None,
                             ref_date: Optional[pd.Timestamp] = None) -> sparse.csr_matrix:
    """User x item sparse matrix.

    weighting: 'binary'; 'count' (number of interactions); 'units' (sum of n_units); or
    'recency' (each interaction weighted 0.5 ** (age / half_life_days), summed per pair,
    where age is measured from ``ref_date``, default the latest date in ``df``).
    """
    rows = users.get_indexer(df["customer_id"].cat.categories)[df["customer_id"].cat.codes.to_numpy()]
    cols = items.get_indexer(df["article_id"])
    ok = (rows >= 0) & (cols >= 0)
    if weighting == "units":
        data = df["n_units"].to_numpy()[ok].astype(np.float32)
    elif weighting == "recency":
        if not half_life_days:
            raise ValueError("weighting='recency' needs half_life_days")
        ref = ref_date if ref_date is not None else df["t_dat"].max()
        age = (ref - df["t_dat"]).dt.days.to_numpy()[ok]
        data = (0.5 ** (age / half_life_days)).astype(np.float32)
    elif weighting in ("binary", "count"):
        data = np.ones(int(ok.sum()), dtype=np.float32)
    else:
        raise ValueError("weighting must be 'binary', 'count', 'units' or 'recency'")
    matrix = sparse.coo_matrix((data, (rows[ok], cols[ok])), shape=(len(users), len(items))).tocsr()
    if weighting == "binary":
        matrix.data[:] = 1.0
    return matrix


def matrix_sparsity(matrix: sparse.spmatrix) -> float:
    return 1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])


# ---------------------------------------------------------- popularity baseline
def popularity_scores(train: pd.DataFrame, items: pd.Index, half_life_days: Optional[float] = None) -> np.ndarray:
    """Item scores = interaction count, optionally exponentially down-weighted by age."""
    pos = items.get_indexer(train["article_id"])
    ok = pos >= 0
    if half_life_days is None:
        weights = np.ones(int(ok.sum()))
    else:
        age = (train["t_dat"].max() - train["t_dat"]).dt.days.to_numpy()[ok]
        weights = 0.5 ** (age / half_life_days)
    return np.bincount(pos[ok], weights=weights, minlength=len(items))


def recommend_popular(scores: np.ndarray, train_matrix: sparse.csr_matrix, user_positions: np.ndarray,
                      n: int = config.TOP_K, exclude_seen: bool = True) -> np.ndarray:
    """Top-n most popular items for each user, optionally skipping items already bought."""
    ranking = np.argsort(-scores, kind="stable")
    out = np.empty((len(user_positions), n), dtype=np.int64)
    for r, u in enumerate(user_positions):
        if not exclude_seen:
            out[r] = ranking[:n]
            continue
        seen = set(train_matrix.indices[train_matrix.indptr[u]:train_matrix.indptr[u + 1]])
        picked = []
        for item in ranking:
            if item not in seen:
                picked.append(item)
                if len(picked) == n:
                    break
        out[r] = picked
    return out


# ------------------------------------------------------------------ experiment bundle
@dataclass
class Experiment:
    """Everything the modelling notebooks share, built once from the prepared data."""
    cleaned: pd.DataFrame
    split: Split
    users: pd.Index
    items: pd.Index
    train_matrix: sparse.csr_matrix       # binary, training period
    val_matrix: sparse.csr_matrix         # binary, validation period
    test_matrix: sparse.csr_matrix        # binary, test period (do not use before Week 4's final run)
    item_user_share: np.ndarray           # share of training users who bought each item
    product_of_item: np.ndarray           # product_code per item column (colour variants share one)
    category_of_item: np.ndarray          # integer code of product_type_name per item column
    item_names: np.ndarray                # prod_name per item column


def build_experiment(prepared: PrepResult, articles: pd.DataFrame) -> Experiment:
    meta = prepared.meta
    split = time_split(prepared.cleaned, pd.Timestamp(meta["val_start"]), pd.Timestamp(meta["test_start"]))
    users, items = make_index(prepared.cleaned)
    by_id = articles.set_index("article_id").loc[items]
    train = build_interaction_matrix(split.train, users, items, "binary")
    return Experiment(
        cleaned=prepared.cleaned, split=split, users=users, items=items, train_matrix=train,
        val_matrix=build_interaction_matrix(split.val, users, items, "binary"),
        test_matrix=build_interaction_matrix(split.test, users, items, "binary"),
        item_user_share=train.getnnz(axis=0) / train.shape[0],
        product_of_item=by_id["product_code"].to_numpy(),
        category_of_item=pd.factorize(by_id["product_type_name"])[0],
        item_names=by_id["prod_name"].to_numpy(),
    )


def users_with_interactions(matrix: sparse.csr_matrix) -> np.ndarray:
    """Row positions of users with at least one interaction in ``matrix``."""
    return np.flatnonzero(matrix.getnnz(axis=1) > 0)
