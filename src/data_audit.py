"""Data loading and audit helpers for the H&M Personalized Fashion dataset.

Design notes
------------
* Every function reports what is in the data; no numbers are hard-coded.
* ``stream_audit`` scans the FULL transactions file in chunks with bounded memory.
  It assumes the file is sorted by date (true for the Kaggle file) and verifies that
  assumption; duplicate counts are exact only if the check passes.
* ``load_transactions`` builds a compact in-memory working set (categorical customer
  ids taken from customers.csv, int32 article ids) so that the 31.8M-row file fits in
  a few GB of RAM.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from src import config

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "transactions": ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
    "customers": [
        "customer_id", "FN", "Active", "club_member_status",
        "fashion_news_frequency", "age", "postal_code",
    ],
    "articles": [
        "article_id", "product_code", "prod_name", "product_type_no", "product_type_name",
        "product_group_name", "graphical_appearance_no", "graphical_appearance_name",
        "colour_group_code", "colour_group_name", "perceived_colour_value_id",
        "perceived_colour_value_name", "perceived_colour_master_id",
        "perceived_colour_master_name", "department_no", "department_name", "index_code",
        "index_name", "index_group_no", "index_group_name", "section_no", "section_name",
        "garment_group_no", "garment_group_name", "detail_desc",
    ],
}
_TX_DTYPES = {"customer_id": "string", "article_id": "int32", "price": "float32",
              "sales_channel_id": "int8"}


# --------------------------------------------------------------------------- I/O
def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Follow the dataset instructions in README.md "
            "to download the H&M files into data/raw/."
        )
    return path


def validate_schema(path: Path, table: str) -> list[str]:
    """Raise if expected columns are missing; return any unexpected extra columns."""
    columns = list(pd.read_csv(_require(path), nrows=0).columns)
    missing = [c for c in EXPECTED_COLUMNS[table] if c not in columns]
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {missing}")
    return [c for c in columns if c not in EXPECTED_COLUMNS[table]]


def load_customers(path: Path = config.CUSTOMERS_FILE) -> pd.DataFrame:
    validate_schema(path, "customers")
    return pd.read_csv(path, dtype={"customer_id": "string"})


def load_articles(path: Path = config.ARTICLES_FILE) -> pd.DataFrame:
    validate_schema(path, "articles")
    return pd.read_csv(path, dtype={"article_id": "int32"})


def load_transactions(
    path: Path = config.TRANSACTIONS_FILE,
    customer_ids: Optional[pd.Series] = None,
    start_date: Optional[str] = config.START_DATE,
    max_rows: Optional[int] = config.MAX_ROWS,
    chunk_size: int = config.CHUNK_SIZE,
) -> pd.DataFrame:
    """Load transactions into a compact DataFrame.

    If ``customer_ids`` (e.g. customers.csv['customer_id']) is given, ``customer_id`` is
    stored as a categorical over that fixed list, which keeps memory low. Ids not found
    in the list become NaN and are reported (they indicate a referential-integrity issue).
    """
    validate_schema(path, "transactions")
    start = pd.Timestamp(start_date) if start_date else None
    dtype = pd.CategoricalDtype(pd.Index(customer_ids)) if customer_ids is not None else None
    kept, rows_read = [], 0
    for chunk in pd.read_csv(path, usecols=EXPECTED_COLUMNS["transactions"], dtype=_TX_DTYPES,
                             parse_dates=["t_dat"], chunksize=chunk_size):
        if max_rows is not None and rows_read + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - rows_read]
        rows_read += len(chunk)
        if start is not None:
            chunk = chunk[chunk["t_dat"] >= start]
        if dtype is not None:
            chunk = chunk.assign(customer_id=chunk["customer_id"].astype(dtype))
        kept.append(chunk)
        if max_rows is not None and rows_read >= max_rows:
            break
    df = pd.concat(kept, ignore_index=True)
    if dtype is None:
        df["customer_id"] = df["customer_id"].astype("category")
    orphans = int(df["customer_id"].isna().sum())
    if orphans:
        logger.warning("%d transactions reference customers missing from customers.csv", orphans)
    logger.info("Kept %d of %d rows read", len(df), rows_read)
    return df


# ------------------------------------------------------- full-file streaming audit
def stream_audit(path: Path = config.TRANSACTIONS_FILE, chunk_size: int = config.CHUNK_SIZE) -> dict:
    """One pass over the whole transactions file with bounded memory.

    Rows of the last date in each chunk are carried into the next chunk, so exact
    duplicate rows (which share a date) are never split across blocks, provided the file
    is sorted by date. ``is_sorted_by_date`` records whether that held.
    """
    validate_schema(path, "transactions")
    acc = {"rows": 0, "missing": None, "duplicates": 0, "users": None, "items": None,
           "channels": None, "daily": [], "price_sum": 0.0, "price_min": np.inf,
           "price_max": -np.inf, "price_nonpositive": 0}

    def consume(block: pd.DataFrame) -> None:
        if block.empty:
            return
        acc["rows"] += len(block)
        miss = block.isna().sum()
        acc["missing"] = miss if acc["missing"] is None else acc["missing"].add(miss)
        acc["duplicates"] += int(block.duplicated().sum())
        for key, col in (("users", "customer_id"), ("items", "article_id"),
                         ("channels", "sales_channel_id")):
            vc = block[col].value_counts()
            acc[key] = vc if acc[key] is None else acc[key].add(vc, fill_value=0)
        acc["daily"].append(block.groupby("t_dat").size())
        price = block["price"].astype("float64")
        acc["price_sum"] += float(price.sum())
        acc["price_min"] = min(acc["price_min"], float(price.min()))
        acc["price_max"] = max(acc["price_max"], float(price.max()))
        acc["price_nonpositive"] += int((price <= 0).sum())

    is_sorted, last_max, carry = True, None, None
    reader = pd.read_csv(path, usecols=EXPECTED_COLUMNS["transactions"], dtype=_TX_DTYPES,
                         parse_dates=["t_dat"], chunksize=chunk_size)
    for chunk in reader:
        if not chunk["t_dat"].is_monotonic_increasing or (
                last_max is not None and chunk["t_dat"].min() < last_max):
            is_sorted = False
        last_max = chunk["t_dat"].max() if last_max is None else max(last_max, chunk["t_dat"].max())
        data = chunk if carry is None else pd.concat([carry, chunk], ignore_index=True)
        cutoff = data["t_dat"].max()
        carry = data[data["t_dat"] == cutoff]
        consume(data[data["t_dat"] < cutoff])
    if carry is not None:
        consume(carry)

    daily = pd.concat(acc["daily"]).groupby(level=0).sum()
    return {
        "rows": acc["rows"],
        "missing": acc["missing"].astype("int64"),
        "exact_duplicate_rows": acc["duplicates"],
        "is_sorted_by_date": is_sorted,
        "date_min": daily.index.min(),
        "date_max": daily.index.max(),
        "n_days_with_sales": int(len(daily)),
        "user_counts": acc["users"].astype("int64").sort_values(ascending=False),
        "item_counts": acc["items"].astype("int64").sort_values(ascending=False),
        "channel_counts": acc["channels"].astype("int64").sort_index(),
        "daily_counts": daily,
        "price_mean": acc["price_sum"] / acc["rows"],
        "price_min": acc["price_min"],
        "price_max": acc["price_max"],
        "price_nonpositive": acc["price_nonpositive"],
    }


def stream_summary(audit: dict) -> pd.Series:
    users, items = audit["user_counts"], audit["item_counts"]
    return pd.Series({
        "interactions (rows)": audit["rows"],
        "users": len(users),
        "items": len(items),
        "date_min": audit["date_min"],
        "date_max": audit["date_max"],
        "days with sales": audit["n_days_with_sales"],
        "exact duplicate rows": audit["exact_duplicate_rows"],
        "file sorted by date (duplicate count exact)": audit["is_sorted_by_date"],
        "mean price (scaled units)": audit["price_mean"],
    })


# ---------------------------------------------------------------- derived measures
def brief_requirements_check(n_interactions: int, n_users: int, n_items: int) -> pd.DataFrame:
    rows = [("interactions", n_interactions, config.BRIEF_MIN_INTERACTIONS),
            ("users", n_users, config.BRIEF_MIN_USERS),
            ("items", n_items, config.BRIEF_MIN_ITEMS)]
    return pd.DataFrame(
        [{"measure": m, "observed": o, "brief minimum": r, "meets minimum": o >= r}
         for m, o, r in rows]).set_index("measure")


def sparsity(n_users: int, n_items: int, n_unique_pairs: int) -> float:
    """Fraction of the user-item matrix that is empty."""
    return 1.0 - n_unique_pairs / (n_users * n_items)


def describe_activity(counts: pd.Series) -> pd.Series:
    """Summary statistics of interactions per entity (user or item)."""
    return counts.describe(percentiles=[0.5, 0.75, 0.9, 0.99])


def gini(counts: pd.Series) -> float:
    """Gini coefficient of interaction counts (0 = equal, 1 = all on one entity)."""
    x = np.sort(counts.to_numpy(dtype="float64"))
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def concentration_table(counts: pd.Series, top_fractions=(0.01, 0.05, 0.10, 0.20)) -> pd.Series:
    """Share of all interactions held by the top x% of entities."""
    ordered = counts.sort_values(ascending=False).to_numpy()
    total = ordered.sum()
    return pd.Series({f"top {f:.0%}": ordered[: max(1, int(len(ordered) * f))].sum() / total
                      for f in top_fractions})


def cold_start_table(user_counts: pd.Series, thresholds=(1, 2, 3, 5, 10, 20)) -> pd.Series:
    """Share of users with at most t interactions."""
    return pd.Series({f"<= {t} purchases": float((user_counts <= t).mean()) for t in thresholds})


def referential_integrity(audit: dict, customers: pd.DataFrame, articles: pd.DataFrame) -> pd.Series:
    users, items = audit["user_counts"].index, audit["item_counts"].index
    return pd.Series({
        "customer ids duplicated in customers.csv": int(customers["customer_id"].duplicated().sum()),
        "article ids duplicated in articles.csv": int(articles["article_id"].duplicated().sum()),
        "transaction customers missing from customers.csv": int((~users.isin(customers["customer_id"])).sum()),
        "transaction articles missing from articles.csv": int((~items.isin(articles["article_id"])).sum()),
        "customers with no transactions": int((~customers["customer_id"].isin(users)).sum()),
        "articles with no transactions": int((~articles["article_id"].isin(items)).sum()),
    })


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"missing": df.isna().sum(), "missing_pct": df.isna().mean() * 100,
                        "n_unique": df.nunique()})
    return out.sort_values("missing_pct", ascending=False)


def image_path(article_id: int, images_dir: Path = config.ARTICLE_IMAGES_DIR) -> Path:
    """H&M photos are stored as images/<first 3 digits of the 10-digit id>/<id>.jpg."""
    padded = f"{int(article_id):010d}"
    return images_dir / padded[:3] / f"{padded}.jpg"


def image_coverage(articles: pd.DataFrame, images_dir: Path = config.ARTICLE_IMAGES_DIR) -> Optional[float]:
    """Share of articles that have a photo on disk, or None if no image folder exists."""
    if not images_dir.exists():
        return None
    return float(np.mean([image_path(a, images_dir).exists() for a in articles["article_id"]]))


# ----------------------------------------------------------- working-set measures
def _customer_codes(series: pd.Series) -> np.ndarray:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy(dtype=np.int64)
    return pd.factorize(series)[0].astype(np.int64)


def unique_pair_count(df: pd.DataFrame) -> int:
    """Number of distinct (customer, article) pairs, i.e. non-empty matrix cells."""
    keys = _customer_codes(df["customer_id"]) * (2 ** 31) + df["article_id"].to_numpy(np.int64)
    return int(np.unique(keys).size)


def repeat_purchase_distribution(df: pd.DataFrame) -> pd.Series:
    """How many times each purchased (customer, article) pair occurs."""
    keys = _customer_codes(df["customer_id"]) * (2 ** 31) + df["article_id"].to_numpy(np.int64)
    _, counts = np.unique(keys, return_counts=True)
    return pd.Series(counts).value_counts().sort_index().rename("pairs")


def working_set_summary(df: pd.DataFrame) -> pd.Series:
    n_users, n_items = df["customer_id"].nunique(), df["article_id"].nunique()
    pairs = unique_pair_count(df)
    return pd.Series({
        "interactions (rows)": len(df), "users": n_users, "items": n_items,
        "unique user-item pairs": pairs,
        "repeat purchases (rows - unique pairs)": len(df) - pairs,
        "sparsity (share of empty matrix cells)": sparsity(n_users, n_items, pairs),
        "date_min": df["t_dat"].min(), "date_max": df["t_dat"].max(),
    })


def basket_analysis(df: pd.DataFrame, articles: pd.DataFrame,
                    group_col: str = "product_group_name", min_size: int = 1,
                    max_size: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treat all items one customer bought on one day as an outfit-basket proxy.

    Returns (basket summary with size and distinct groups per basket for ALL baskets,
             group x group co-occurrence matrix counted only in baskets whose size lies in
             [min_size, max_size]; the diagonal is the number of such baskets containing
             that group).
    Restricting basket size matters: very large baskets contain almost every group and
    push every pairwise lift towards 1.
    """
    df = df[df["customer_id"].notna()]
    groups = df["article_id"].map(articles.set_index("article_id")[group_col])
    g_codes, g_names = pd.factorize(groups)
    day = (df["t_dat"] - df["t_dat"].min()).dt.days.to_numpy(np.int64)
    key = _customer_codes(df["customer_id"]) * 100_000 + day
    basket_ids, inverse = np.unique(key, return_inverse=True)
    size = np.bincount(inverse, minlength=len(basket_ids))

    in_range = size >= min_size
    if max_size is not None:
        in_range &= size <= max_size
    row_ok = (g_codes >= 0) & in_range[inverse]

    n_groups = len(g_names)
    pair_keys = np.unique(inverse[row_ok].astype(np.int64) * n_groups + g_codes[row_ok])
    b, g = pair_keys // n_groups, pair_keys % n_groups
    all_pairs = np.unique(inverse[g_codes >= 0].astype(np.int64) * n_groups + g_codes[g_codes >= 0])
    distinct = np.bincount(all_pairs // n_groups, minlength=len(basket_ids))
    membership = sparse.csr_matrix((np.ones(len(b), dtype=np.int32), (b, g)),
                                   shape=(len(basket_ids), n_groups))
    cooc = pd.DataFrame((membership.T @ membership).toarray(), index=g_names, columns=g_names)
    summary = pd.DataFrame({"basket_size": size, "distinct_product_groups": distinct})
    return summary, cooc


def lift_matrix(cooc: pd.DataFrame, n_baskets: int, min_support: int = 1) -> pd.DataFrame:
    """Lift = P(a and b) / (P(a) P(b)) over ``n_baskets`` baskets.

    Pairs seen in fewer than ``min_support`` baskets and the diagonal are set to NaN, so
    tiny groups cannot produce huge, meaningless lifts.
    """
    values = cooc.to_numpy().astype(float)
    diag = np.diag(values).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = values * n_baskets / np.outer(diag, diag)
    lift[values < min_support] = np.nan
    np.fill_diagonal(lift, np.nan)
    return pd.DataFrame(lift, index=cooc.index, columns=cooc.columns)


def export_summary(summary: dict, path: Path) -> None:
    """Write a JSON snapshot of key audit numbers for the report."""
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=convert))
