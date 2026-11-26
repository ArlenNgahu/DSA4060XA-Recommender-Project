"""Serving layer: turns the fitted hybrid into customer-facing shelves with reasons.

Shelves
-------
* Buy again      : the customer's own past products, most recent first (repurchase component).
* Discover       : the discovery hybrid (items the customer has not bought), at most one colour
                   variant per product, each with the component that contributed most.
* Trending       : the recency-weighted popularity list, for customers without history.
* Complete look  : items bought on the same day as a chosen item, from other product types.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src import config, hybrid as hy, outfit as of
from src.recommender import NeighbourTable, diversify_by_product


def _first_per_product(items: list[int], product_of_item: np.ndarray) -> list[int]:
    seen, out = set(), []
    for item in items:
        product = product_of_item[item]
        if product not in seen:
            seen.add(product)
            out.append(int(item))
    return out


class RecommendationService:
    def __init__(self, model: hy.HybridRecommender, settings: dict, articles: pd.DataFrame,
                 history: pd.DataFrame, outfit_table: Optional[NeighbourTable] = None):
        self.model, self.settings, self.outfit_table = model, settings, outfit_table
        self.articles = articles.set_index("article_id")
        self.history = history
        by_id = self.articles.loc[model.items]
        self.product_of_item = by_id["product_code"].to_numpy()
        self.names = by_id["prod_name"].to_numpy()
        self.types = by_id["product_type_name"].to_numpy()
        self.colours = by_id["colour_group_name"].to_numpy()
        self.departments = by_id["department_name"].to_numpy()
        self.k = settings.get("top_k", config.TOP_K)

    # ------------------------------------------------------------------ lookup
    def customers(self) -> list[str]:
        return [str(c) for c in self.model.users]

    def position(self, customer_id: str) -> int:
        return int(self.model.users.get_loc(customer_id))

    def _rows(self, customer_id: str) -> pd.DataFrame:
        return self.history[self.history["customer_id"] == customer_id]

    def profile(self, customer_id: str) -> dict:
        rows = self._rows(customer_id)
        pos = self.position(customer_id)
        return {"distinct items bought": int(self.model.seen[pos].nnz),
                "purchase days": int(rows["t_dat"].nunique()),
                "last purchase": rows["t_dat"].max().date() if len(rows) else None}

    def _describe(self, positions: list[int], reasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "Product": self.names[positions], "Type": self.types[positions],
            "Colour": self.colours[positions], "Department": self.departments[positions],
            "Why": reasons})

    def recent_purchases(self, customer_id: str, n: int = 8) -> pd.DataFrame:
        rows = self._rows(customer_id).sort_values("t_dat", ascending=False).drop_duplicates("article_id").head(n)
        pos = self.model.items.get_indexer(rows["article_id"])
        out = self._describe(list(pos), [""] * len(pos)).drop(columns="Why")
        out.insert(0, "Date", rows["t_dat"].dt.date.to_numpy())
        return out

    # ----------------------------------------------------------------- shelves
    def buy_again(self, customer_id: str, n: int = 5) -> pd.DataFrame:
        pos = self.position(customer_id)
        ranked = hy.repurchase_ranking(self.model.repurchase_history, np.array([pos]), 5 * n)[0]
        items = _first_per_product([int(i) for i in ranked if i >= 0], self.product_of_item)[:n]
        last = (self._rows(customer_id).groupby("article_id")["t_dat"].max())
        reasons = [f"Last bought on {last[self.model.items[i]]:%d %b %Y}" for i in items]
        return self._describe(items, reasons)

    def discover(self, customer_id: str, n: int = 10, hide_owned_products: bool = False) -> pd.DataFrame:
        model, pos = self.model, np.array([self.position(customer_id)])
        weights = self.settings["weights_discovery"]
        gamma = self.settings["outfit_gamma_discovery"]
        ranks = model.component_ranks(pos)
        wide = model.recommend(ranks, weights, pos, 6 * n, gamma=gamma)[0]
        items = _first_per_product([int(i) for i in wide], self.product_of_item)
        if hide_owned_products:
            owned = set(self.product_of_item[model.seen[pos[0]].indices])
            items = [i for i in items if self.product_of_item[i] not in owned]
        items = items[:n]
        reasons = []
        for item in items:
            contrib = hy.component_contributions(ranks, weights, 0, item)
            top = max(contrib, key=contrib.get) if contrib else "popularity"
            reason = hy.COMPONENT_LABELS[top].capitalize()
            if top == "item_knn":
                why = model.knn.explain(model.knn_history, int(pos[0]), item, top=1)
                if why:
                    reason += f" (because you bought {self.names[why[0][0]]})"
            reasons.append(reason)
        return self._describe(items, reasons)

    def trending(self, n: int = 10) -> pd.DataFrame:
        items = _first_per_product([int(i) for i in self.model.pop_ranking[: 6 * n]], self.product_of_item)[:n]
        return self._describe(items, ["Trending right now"] * len(items))

    def complete_the_look(self, article_id: int, n: int = 6) -> pd.DataFrame:
        if self.outfit_table is None:
            raise RuntimeError("outfit model not available; run notebook 05")
        anchor = int(self.model.items.get_loc(article_id))
        cand = of.complement_candidates(self.outfit_table, np.array([anchor]), self.model.category_of_item,
                                        self.product_of_item, 6 * n, self.model.pop_ranking)[0]
        items = _first_per_product([int(i) for i in cand], self.product_of_item)[:n]
        sims = {int(i): float(s) for i, s in zip(self.outfit_table.indices[anchor], self.outfit_table.similarity[anchor])}
        reasons = [f"Often bought on the same day as {self.names[anchor]}" if sims.get(i, 0) > 0
                   else "Trending item from a different type" for i in items]
        return self._describe(items, reasons)

    def search_items(self, text: str, limit: int = 30) -> pd.DataFrame:
        mask = pd.Series(self.names).str.contains(text, case=False, regex=False).to_numpy()
        positions = np.flatnonzero(mask)[:limit]
        out = self._describe(list(positions), [""] * len(positions)).drop(columns="Why")
        out.insert(0, "article_id", self.model.items[positions])
        return out
