# Presentation and defence outline (about 10 minutes plus questions)

## Slides
1. **Problem.** What is recommended, to whom, and why it matters (one sentence each for user, item, interaction).
2. **Data.** H&M dataset size; the two decisions that shaped it: same-day collapse and leakage-safe filtering.
3. **What the data showed.** Sparsity, popularity concentration, fast drift, colour variants, repeat purchases.
4. **Baselines and the first surprise.** Recency popularity beats collaborative filtering on accuracy; CF wins on coverage.
5. **Models.** Item-item CF, latent factors, content similarity, repurchase: one line each on how they work.
6. **Hybrid.** Rank fusion, tuned weights, ablation chart (`images/hybrid_ablation.png`).
7. **Evaluation design.** Time split, two tracks, metrics, bootstrap intervals.
8. **Results.** The test model-comparison table and the three numbers that matter most.
9. **Outfit completion and the demo.** Live: pick a customer, show the two shelves and the reasons, then "Complete the look".
10. **Responsible recommendation.** Three findings from notebook 05 with numbers.
11. **Limitations and next steps.** The cold-item share, single split, purchases not preferences.
12. **Final choice.** Which model and why.

## Demo script (3 minutes)
1. `streamlit run app/app.py`; choose a customer with 20+ items; read the profile and recent purchases aloud.
2. Show "Buy it again" and explain it comes from the customer's own history.
3. Show "Recommended for you"; point at two reasons ("because you bought ...", "trending right now").
4. Tick "Hide other colours"; explain what changes and why it matters.
5. Open "Complete the look", search a product, show suggestions from other product types.
6. Open "New customer" and explain the cold-start fallback.

## Questions you should be ready to answer
- Why is the filtering done on the training period only? (Otherwise the future decides which items exist.)
- Why does popularity beat collaborative filtering here, and why use CF at all? (Recency dominates fashion; CF adds coverage and novelty.)
- Why reciprocal-rank fusion instead of averaging scores? (Component scores are on different scales; ranks are comparable.)
- How were weights chosen and how do you know they are not overfitted? (Validation only; test used once after refit; validation-versus-test table.)
- What does Precision@10 of about 0.01 mean? (About one recommended item in a hundred was bought in the next two weeks; compare relative values.)
- What would you change with more time? (Price and availability, a learned ranker, evaluating cold items in the main comparison, an online test.)
- What are the privacy risks and how are they limited? (Purchase history is personal data; the model uses no demographics; IDs are anonymised; show only short IDs.)
