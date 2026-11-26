"""Streamlit prototype: personalised shelves with reasons, outfit completion and a cold-start view.

Run from the project root:   streamlit run app/app.py
Requires notebooks 02, 04 and 05 to have been run (they create data/cleaned/*.pkl and the parquet files).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src import config, data_audit as da, prepare as prep, serving

st.set_page_config(page_title="Fashion Outfit Recommender", layout="wide")


@st.cache_resource(show_spinner="Loading the trained model...")
def load_service():
    model_path = config.CLEANED_DIR / "final_hybrid.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as handle:
        bundle = pickle.load(handle)
    prepared = prep.load_prepared()
    test_start = pd.Timestamp(prepared.meta["test_start"])
    history = prepared.cleaned[prepared.cleaned["t_dat"] < test_start]
    outfit_table = None
    outfit_path = config.CLEANED_DIR / "outfit_model.pkl"
    if outfit_path.exists():
        with open(outfit_path, "rb") as handle:
            outfit_table = pickle.load(handle)["table"]
    return serving.RecommendationService(bundle["model"], bundle["settings"], da.load_articles(), history, outfit_table)


def show(table: pd.DataFrame) -> None:
    """Render a table with a 1-based index; plain tables are read well by screen readers."""
    table = table.copy()
    table.index = range(1, len(table) + 1)
    st.table(table)


service = load_service()
st.title("Fashion Outfit Recommender")
st.caption("DSA 4060XA individual project. Demonstration on anonymised H&M purchase data; no real customers are shown.")

if service is None:
    st.error("The trained model was not found. Run notebooks 02, 04 and 05 first, then restart this app.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    n_discover = st.slider("Items in 'Recommended for you'", 5, 20, 10)
    n_again = st.slider("Items in 'Buy it again'", 3, 10, 5)
    hide_owned = st.checkbox("Hide other colours of products the customer already owns", value=False)
    st.caption("Recommendations are computed live from purchase history only. Age and other personal "
               "attributes are not used by the model.")

tab_customer, tab_look, tab_new, tab_about = st.tabs(
    ["For a customer", "Complete the look", "New customer", "About this system"])

with tab_customer:
    rng = np.random.default_rng(config.RANDOM_SEED)
    all_ids = service.customers()
    sample = [all_ids[i] for i in rng.choice(len(all_ids), size=min(200, len(all_ids)), replace=False)]
    customer = st.selectbox("Choose a customer", sample,
                            format_func=lambda c: f"Customer {c[:8]} ({service.profile(c)['distinct items bought']} items bought)")
    info = service.profile(customer)
    cols = st.columns(3)
    cols[0].metric("Distinct items bought", info["distinct items bought"])
    cols[1].metric("Purchase days", info["purchase days"])
    cols[2].metric("Last purchase", str(info["last purchase"]))

    st.subheader("Recently bought")
    show(service.recent_purchases(customer))
    st.subheader("Buy it again")
    st.caption("The customer's own past products, most recent first.")
    show(service.buy_again(customer, n_again))
    st.subheader("Recommended for you")
    st.caption("Items the customer has not bought. Each line says which part of the system suggested it.")
    show(service.discover(customer, n_discover, hide_owned_products=hide_owned))

with tab_look:
    st.subheader("Complete the look")
    st.write("Search for a product, pick one, and see items from other product types that customers "
             "often bought on the same day.")
    query = st.text_input("Product name contains", value="dress")
    matches = service.search_items(query) if query else pd.DataFrame()
    if matches.empty:
        st.info("No product matches that text.")
    elif service.outfit_table is None:
        st.warning("The outfit model was not found. Run notebook 05 to create it.")
    else:
        labels = {int(r.article_id): f"{r.Product} ({r.Type}, {r.Colour})" for r in matches.itertuples()}
        chosen = st.selectbox("Pick a product", list(labels), format_func=labels.get)
        show(service.complete_the_look(int(chosen)))

with tab_new:
    st.subheader("New customer (no purchase history)")
    st.write("Without history the system falls back to what is trending now, one article per product. "
             "It needs no personal data.")
    show(service.trending(n_discover))

with tab_about:
    st.subheader("How it works")
    st.markdown(
        "- **Components:** recency-weighted popularity, item-item collaborative filtering, latent factors (SVD), "
        "content similarity and repurchase, combined by weighted reciprocal-rank fusion with weights tuned on a "
        "validation period.\n"
        "- **Complete the look:** items bought in the same customer-day basket, from a different product type.\n"
        "- **Reasons:** each recommendation shows the component that contributed most to it.")
    results_path = config.DOCS_DIR / "model_comparison_test.csv"
    if results_path.exists():
        st.subheader("Measured results on the held-out test period")
        st.dataframe(pd.read_csv(results_path, index_col=[0, 1]))
    st.subheader("Limitations")
    st.markdown(
        "- Results are on a time-based test block, measured on purchases, not on customer satisfaction.\n"
        "- Many purchases in the test period are of new or rarely bought items that collaborative components cannot "
        "score; only content similarity reaches them.\n"
        "- The system reinforces what is popular and what customers already buy; see the responsible-recommendation "
        "analysis in notebook 05.")
