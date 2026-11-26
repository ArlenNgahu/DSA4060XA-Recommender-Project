# DSA 4060XA Individual Recommender Systems Project
Project option 18: **Fashion Outfit Recommender** (item-level, outfit-aware hybrid).

## Status
- [x] Week 1: problem definition, dataset audit (`notebooks/01_data_understanding.ipynb`)
- [x] Week 2: cleaning, sparsity, popularity baseline (`notebooks/02_eda_and_baseline.ipynb`)
- [x] Week 3: personalized model, item-based collaborative filtering (`notebooks/03_personalized_model.ipynb`)
- [x] Week 4: advanced model, hybrid and final evaluation (`notebooks/04_advanced_model_and_evaluation.ipynb`)
- [x] Week 5: outfit completion, responsible recommendation (`notebooks/05_outfit_and_responsible_recommendation.ipynb`), prototype (`app/app.py`)
- [ ] Final report and presentation (see `docs/report_outline.md` and `docs/presentation_outline.md`)

## Dataset
H&M Personalized Fashion Recommendations (Kaggle competition data).
1. Download the competition files from Kaggle (a Kaggle account and acceptance of the
   competition rules are required).
2. Place `transactions_train.csv`, `articles.csv` and `customers.csv` in `data/raw/`.
   Product photos are optional: if you download them, keep the `images/` folder inside `data/raw/`.
3. Raw data is git-ignored; do not commit it.

Cite the dataset and respect its licence terms in the final report.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_data_understanding.ipynb
```
`src/config.py` controls paths and the optional `START_DATE` / `MAX_ROWS` limits for the in-memory working set.

`01_data_understanding.ipynb` has four parts: (A) a streaming audit of the full transactions file,
(B) article and customer metadata checks, (C) working-set analysis (sparsity, repeats, baskets,
product-group co-purchase) and (D) export of key numbers to `docs/audit_summary.json`.
The first run streams the whole file once and caches the result in `data/cleaned/stream_audit.pkl`.

## Week 2 workflow
Run `01_data_understanding.ipynb` first, then `02_eda_and_baseline.ipynb`. Notebook 02 collapses
same-day repeat purchases, prints a sensitivity table of window / minimum-interaction / sample choices,
applies the choices set in `src/config.py`, saves `data/cleaned/interactions.parquet`, builds the
time-based train/validation/test split and evaluates popularity baselines on validation only.
The test block is not used until Week 4.

Filtering is leakage-safe: thresholds (distinct items per user, distinct buyers per item) are measured on
the training period only, so validation and test data never decide which users or items exist. Validation
and test purchases of items outside that universe are saved as `interactions_cold_items.parquet`.

## Week 3 workflow
Run `03_personalized_model.ipynb` after notebook 02. It tunes item-item cosine collaborative filtering
(shrinkage, neighbours, recency weighting) on validation only, compares it with the popularity baselines,
shows example users with "because you bought X" explanations, and breaks results down by user activity.

## Week 4 workflow
Run `04_advanced_model_and_evaluation.ipynb` after notebook 03. It first measures what customers buy next
(repeats, colour variants, new products), tunes five components on validation (popularity, item-item CF,
latent factors, content similarity, repurchase), fuses them with tuned weights and an optional outfit-affinity
re-rank, then refits on training plus validation and evaluates once on the test block. Two tracks are
reported: all purchases, and discovery of items never bought before. It also covers a bootstrap comparison,
a cold-item slice and example recommendations with reasons, and saves `docs/model_comparison_test.csv`,
`docs/final_model_settings.json` and `data/cleaned/final_hybrid.pkl` for the Week 5 prototype.

The notebook numbering differs slightly from the brief's suggested layout: notebooks 04 (advanced model and
evaluation) here cover the brief's 05 and 06.

## Week 5 workflow
1. Re-run `04_advanced_model_and_evaluation.ipynb` (it saves `data/cleaned/final_hybrid.pkl` and the settings).
2. Run `05_outfit_and_responsible_recommendation.ipynb` (outfit completion, collapse of colour variants,
   popularity bias, fairness by age group, filter bubbles, cold start; saves `data/cleaned/outfit_model.pkl`
   and `docs/responsible_and_outfit_metrics.csv`).
3. Start the prototype from the project root: `streamlit run app/app.py`.
4. Write the report and presentation using `docs/report_outline.md` and `docs/presentation_outline.md`.

## Structure
```
data/raw, data/cleaned   inputs and prepared data
notebooks/               one notebook per pipeline stage
src/                     config.py, data_audit.py (Week 1), prepare.py (leakage-safe filtering, split,
                         matrices, popularity baseline), recommender.py (item-item CF), latent.py (PureSVD),
                         hybrid.py (content features, repurchase, outfit affinity, rank fusion), outfit.py (complete the look),
                         serving.py (shelves with reasons for the app),
                         evaluation.py (ranking and beyond-accuracy metrics, bootstrap)
docs/                    problem statement and modelling plan
images/                  figures used in the report
app/                     Streamlit prototype (app.py)
```

## Acknowledgements
Dataset: H&M Group via Kaggle. Libraries: pandas, NumPy, SciPy, matplotlib, seaborn.
