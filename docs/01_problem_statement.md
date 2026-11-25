# Problem Statement and Modelling Plan (Week 1)

## Recommendation objective
Recommend individual fashion items to a specific customer so that the items fit
what the customer is likely to buy and work together with items the customer
already owns or is currently considering (outfit-aware recommendation).

## Definitions
| Concept | Definition in this project |
|---|---|
| User | An H&M customer (`customer_id`) |
| Item | An H&M article (`article_id`), with category, colour, pattern and text metadata |
| Interaction | A purchase transaction (implicit feedback, no ratings) |
| Outfit context | Items the same customer bought on the same day (proxy for an outfit basket), plus product-group complementarity |

## Dataset
H&M Personalized Fashion Recommendations (Kaggle): `transactions_train.csv`,
`articles.csv`, `customers.csv`. Source, licence and any filtering/sampling are
documented in README.md and in the Week 2 cleaning notebook.

## Modelling plan
1. Popularity baseline (consider recency weighting).
2. Personalized model: content-based filtering on article attributes OR item-item
   collaborative filtering with cosine similarity.
3. Advanced model: matrix factorization / implicit latent factors, then a
   compatibility re-ranking layer (justified hybrid).
4. Evaluation: time-based split; Precision@K, Recall@K, NDCG@K; coverage and
   novelty / popularity bias; interpretation for the business user.

## Open questions to resolve with data
- Does the sampled dataset meet 10,000 interactions, 500 users and 500 items after
  filtering? Confirm and report the actual numbers.
- Are same-day baskets large and diverse enough to act as an outfit proxy?
- MAE/RMSE are not applicable to purchase data; the omission will be justified.
