# Final report outline (8 to 12 pages) and where the evidence lives

Write every section in your own words. The numbers below are the ones that existed when this outline was
written; copy the final values from your last runs (`docs/model_comparison_test.csv`,
`docs/responsible_and_outfit_metrics.csv`, `docs/audit_summary.json`) because the widened tuning grids can
change them.

| # | Section (brief) | Marks | Evidence to use | Points you must be able to defend |
|---|---|---|---|---|
| 1 | Introduction and problem statement | 8 | `docs/01_problem_statement.md` | Why fashion outfit recommendation; user, item, interaction; why the framing is item-level with an outfit-aware layer |
| 2 | Dataset and data preparation | 10 | Notebook 01 parts A and B; notebook 02 sections 1 to 3 | Source and licence; 31.8M transactions, 1.36M customers, 104,547 articles; same-day collapse (about 10% of rows); time window, user sample and k-core thresholds measured on the training period only (no leakage) |
| 3 | Exploratory analysis | 8 | Notebook 01 A4 to A6, B; notebook 02 sections 4 to 6 | Sparsity about 99.87% after filtering; popularity concentration (Gini 0.76 full data, 0.41 after filtering, and why they differ); colour variants share a `product_code`; popularity drifts quickly |
| 4 | Popularity baseline | 6 | Notebook 02 section 8; notebook 04 section 3a | Recency matters: a 3-day half-life beats global popularity; why coverage is tiny |
| 5 | Personalised model | 12 | Notebook 03 | Item-item cosine CF on binary purchases; tuning on validation; it did not beat recency popularity on accuracy but covers far more of the catalogue; example users with reasons |
| 6 | Advanced model and hybrid | 15 | Notebook 04 sections 3 to 5 | PureSVD (what the factors are, damping); content features; repurchase; reciprocal-rank fusion and why weights are tuned on validation; ablation: what each component adds |
| 7 | Evaluation and model comparison | 15 | Notebook 04 sections 6 to 9 | Time-based split; two tracks (all purchases, discovery); Precision/Recall/NDCG@10; coverage, novelty, popularity bias, category diversity; bootstrap intervals; why MAE/RMSE are not used; validation-versus-test gap |
| 8 | Outfit completion | (part of 6/7) | Notebook 05 part A | The honest result of basket co-purchase against trending and content baselines; why the product-type affinity re-rank added little |
| 9 | Responsible recommendation | 6 | Notebook 05 part C | At least three issues, each with a measurement: popularity bias and exposure; fairness across age groups; filter bubbles; cold start; privacy and data minimisation; manipulation and feedback loops |
| 10 | Limitations | | Notebook 04 section 8; notebook 05 C4 | Most test-period purchases are of items outside the item universe; one split; sampled customers; purchases are not preferences; items are scored without price, stock or size |
| 11 | Conclusion and recommendation | | Model comparison table | Final model choice and why; how it would be used (two shelves: "Buy it again", "Recommended for you"); what to do next |

## Checks before you submit
- Every number in the report appears in a saved table or notebook output, and the README tells a reader how to reproduce it.
- No result is described as better than another unless the bootstrap interval supports it.
- The report says what each metric means in business terms ("about 1 in 100 recommended items was bought within two weeks"), not only the value.
- All external sources, libraries and the dataset are acknowledged in the README and the report.
- The GitHub history shows your own work over time (commit after each week's notebook).
