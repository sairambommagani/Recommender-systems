# Recommender Systems: MovieLens 100K and Book-Crossing

The project builds two recommender systems. Both predict **ratings** and recommend a **top-10 list** for
each user, and both are served through one **API** that explains every recommendation.

| | Task 1: MovieLens 100K | Task 2: Book-Crossing |
|---|---|---|
| Data | 100,000 ratings (1–5 stars), 943 users, 1,682 movies | ~1.1M ratings, 278k users, 271k books, plus book metadata and user country and age |
| Feedback | explicit and clean | 0 = "read but not rated" (implicit, about 60%) and 1–10 explicit; very sparse and messy |
| Split | per user **by time**: each user's latest 20% of ratings is the test set | per user at random (the data has no timestamps) |
| Top-10 models | popularity, item-kNN, implicit ALS, EASE | same, plus content-based (title, author, publisher) and a **hybrid** |
| Rating models | global mean, user/item biases, biased matrix factorisation (ALS) | same, on the 1–10 explicit ratings |
| Cold start | new users who rate 5 movies at sign-up | new users with no books, new users with 3 books, and **new books nobody has read** |

| Code | What it does |
|---|---|
| `recsys/models.py` | every model, behind one interface (`fit`, `score`, `explain`) |
| `recsys/metrics.py`, `recsys/evaluation.py` | NDCG/recall/precision/MAP/hit-rate@10, coverage, novelty, RMSE/MAE; splits; k-core filtering; tuning |
| `recsys/pipeline.py` | the two model ladders shared by both tasks |
| `recsys/movielens.py` | Task 1: load → split → tune on validation → test → cold start → save |
| `recsys/books.py` | Task 2: load → **clean** → k-core → cold-start hold-outs → split → tune → test → save |
| `recsys/api.py` | FastAPI service for both tasks |
| `tests/` | end-to-end tests on synthetic data that uses the real file formats |

## 1. Method

**Fair comparison.** Every model sees the same split.
- Hyper-parameters are chosen on a **validation** set carved out of the training data. These cover item-kNN
  neighbours, the EASE regulariser, the ALS factors, regularisation and confidence, and the hybrid weight.
- The test set is used once, for the final numbers.
- For top-10, each user's training items are excluded, and **all** remaining items are ranked. There are no
  sampled negatives, because sampling inflates the metrics.

**Metrics.**
- **NDCG@10** is the headline top-10 metric. It rewards putting the right item near the top.
- Also reported: recall@10, precision@10, hit-rate@10 and MAP@10.
- **Coverage** is the share of the catalogue that is ever recommended. Popularity models score about 1%.
- **Novelty** is how un-obvious the recommendations are.
- **RMSE** measures rating prediction.

**Models.**

| Model | Idea |
|---|---|
| Popularity | most-interacted items; the floor every model must beat |
| Item-kNN | cosine similarity between items from who interacted with them; keeps each item's top-k neighbours |
| Implicit ALS | matrix factorisation for implicit feedback (Hu, Koren & Volinsky 2008); observed cells get confidence 1 + α |
| EASE | closed-form item-to-item model (Steck 2019): B = I − P/diag(P), P = (XᵀX + λI)⁻¹. Strong and explainable |
| Content (Task 2) | TF-IDF of title words, author, publisher and decade; a user profile is the mean of the books they read |
| Hybrid (Task 2) | w · z(collaborative) + (1 − w) · z(content), with w tuned on validation |
| Biased MF (ratings) | r̂ = μ + b_u + b_i + p_u·q_i, fitted by ALS; biases get a constant penalty, so items with few ratings are pulled toward the average |

**New users without retraining.** Item-kNN, EASE and ALS can score any list of items a user liked. A new user is
therefore folded in at request time, and the API does exactly that.

**Task 2 cleaning** (`books.py`):
- ISBNs are normalised (uppercase, digits and X only), and duplicate (user, book) rows are removed.
- Ratings whose ISBN isn't in `Books.csv` are dropped.
- Years that aren't numbers (publisher names in that column), 0, or after 2006 are set to missing.
- Ages outside 5–100 (0, 244 and so on) are set to missing.
- The country is taken from `Location`.
- The model then keeps users and books with at least 5 interactions (k-core), because ones with 1–2
  interactions can't be learned or evaluated.
- Very heavy "users" with thousands of rows (bulk imports) are sampled down to 500.
- Everything removed is counted in `metrics.json → data.cleaning`.

**Task 2 cold-start evaluation.** 5% of books and 10% of users are removed from training entirely, then:
- **New user, nothing known:** most popular in their country, or country + age group, vs. global popularity.
- **New user, 3 books known:** every model folds the user in; their other books are the test set.
- **New book, no readers yet:** for readers who later read these books, rank the new books by content
  similarity to their history, vs. random ranking.

## 2. Results

All numbers come from the real datasets (`artifacts/*/metrics.json`).

**Task 1, MovieLens 100K**

**Rating prediction** (test = each user's latest 20% of ratings):

| Model | RMSE | Note |
|---|---|---|
| global mean | 1.211 | always predicts the average rating |
| user + movie biases | 0.988 | "this user rates high, this movie is liked" |
| **biased MF** (30 factors, ALS) | **0.969** | adds personal taste |

These are higher than the ~0.91–0.93 often quoted for MovieLens 100K. Those figures use random splits, where the
model sees ratings made *after* the ones it predicts. Predicting each user's future ratings is harder and closer to
real use.

**Top-10** (relevant = test movie rated ≥ 4; all unseen movies ranked):

| Model | NDCG@10 | Recall@10 | Coverage |
|---|---|---|---|
| popularity | 0.0977 | 0.0820 | 0.043 |
| **item-kNN** (k = 200, chosen on validation, served by the API) | **0.1675** | 0.1512 | 0.213 |
| implicit ALS (64 factors) | 0.1649 | 0.1515 | 0.304 |
| EASE (λ = 50) | 0.1716 | 0.1576 | 0.314 |

- All three collaborative models are about **1.7× better than popularity**, and they recommend 5–7× more of the
  catalogue.
- Validation scores were within 0.004 of each other. Item-kNN was picked on validation, so it's the served model,
  even though EASE was 0.004 higher on test. Choosing on test would be leakage.

**New users** (20% of users removed from training; each model sees only their first 5 ratings):

| Model | NDCG@10 | Recall@10 |
|---|---|---|
| **popularity** | **0.352** | 0.109 |
| item-kNN | 0.296 | 0.116 |
| implicit ALS | 0.282 | **0.125** |
| EASE | 0.269 | 0.110 |

With only 5 sign-up ratings, popularity gives the best top-10 order. The collaborative models find slightly more of
the movies the user liked later (higher recall), but rank them lower. A model + popularity rank blend was tuned on
separate validation users, and every model weight above 0 lowered NDCG. The API therefore serves popular unseen
movies to new users. Asking new users to rate more movies, or adding movie metadata (genre, year), is the next step.

**Task 2, Book-Crossing**

**Data after cleaning:**
- 1,149,780 raw ratings → 1,031,585 clean. 118,149 ratings had an ISBN that isn't in `Books.csv`, and 46 were
  duplicates.
- 4,620 invalid years and 1,248 impossible ages were set to missing.
- **62.8% of the ratings are implicit 0s.**
- The 5-core keeps 500,215 interactions from 15,247 users on 36,076 books. 142 bulk users were capped at 500.
- 1,803 books and 1,524 users were held out entirely for the cold-start tests.

**Top-10** (random 20% of each user's books; all unseen books ranked):

| Model | NDCG@10 | Recall@10 | Coverage |
|---|---|---|---|
| popularity | 0.0158 | 0.0170 | 0.001 |
| item-kNN (k = 50) | 0.0520 | 0.0531 | 0.671 |
| implicit ALS | 0.0376 | 0.0464 | 0.054 |
| EASE (λ = 100, 8,000 books) | 0.0588 | 0.0622 | 0.143 |
| content (TF-IDF) | 0.0281 | 0.0422 | 0.542 |
| **hybrid: 0.8 × EASE + 0.2 × content** | **0.0663** | **0.0727** | – |

- The hybrid is **4.2× better than popularity**.
- Adding content to EASE raises NDCG by **13%**. Content matters on sparse data, where many books have only a
  handful of readers.
- Absolute numbers are low because Book-Crossing is extremely sparse: 99.91% of the 5-core matrix is empty, and
  readers rarely share books.

**Cold start** (users and books never seen in training):

| Scenario | Best approach | NDCG@10 | Baseline |
|---|---|---|---|
| new user, nothing known | most read in their **country** | 0.0541 | global popularity 0.0491 |
| new user, 3 books known | **hybrid** (folded in, no retraining) | 0.0771 | popularity 0.0443 · EASE 0.0691 · item-kNN 0.0476 |
| new book, no readers yet | **content** (title, author, publisher) | 0.1107 | random 0.0035 (**about 30× better**) |

**Rating prediction** (explicit 1–10 ratings only):

| Model | RMSE |
|---|---|
| global mean | 1.805 |
| biases | 1.571 |
| biased MF (10 factors, strongly regularised) | 1.571 |

On Book-Crossing, matrix factorisation adds nothing over user and book biases: the explicit ratings are too sparse
for the factors to learn tastes. Validation picked the strongest regularisation, which shrinks MF toward the bias
model. This is a known result for this dataset. Implicit "read" signals (above) carry more information than the
few explicit ratings.

## 3. Run it (Windows PowerShell)

```powershell
cd recommender-systems
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest                       # about 1 minute, offline
```

1. **Data.**
   - Unzip `ml-100k.zip` ([GroupLens](https://grouplens.org/datasets/movielens/100k/)) into `data\movielens\`.
   - Unzip the [Kaggle Book-Crossing download](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
     (`Books.csv`, `Ratings.csv`, `Users.csv`) into `data\books\`.
2. **Train.**
   ```powershell
   python -m recsys.movielens --raw data\movielens     # a few minutes
   python -m recsys.books --raw data\books             # longer; --fast uses smaller grids
   ```
   Outputs go to `artifacts\movielens\` and `artifacts\books\`: models, `metrics.json` and the tuning tables.
3. **API.**
   ```powershell
   python -m uvicorn recsys.api:app --port 8003
   ```
   Open http://localhost:8003/docs.

## 4. API

| Endpoint | Purpose |
|---|---|
| `GET /movies/users/{user_id}/recommendations` | top-N for a user, each with "because you liked …" |
| `POST /movies/new-user` | `{"liked_movie_ids": [50, 172, 181]}` returns recommendations for a brand-new user, with no retraining |
| `POST /movies/predict-rating` | predicted stars, split into global mean + user bias + movie bias + taste match |
| `GET /movies/{movie_id}/similar` | "people who liked this also liked" |
| `GET /movies/search?q=star` | find movie ids |
| `GET /books/users/{user_id}/recommendations` | a known reader gets the hybrid model with "because you read …"; a user without enough history gets the most read in their country and age group |
| `POST /books/new-user` | liked ISBNs and/or country + age |
| `POST /books/predict-rating` | predicted 1–10 rating with breakdown |
| `GET /books/{isbn}/similar` | readers-also-read + similar by title/author/publisher; **works for books with no readers** |
| `GET /books/search?q=grisham` | find ISBNs by title or author |
| `GET /model-info`, `GET /health` | test results of every model, cold-start results, status |

## 5. Limitations and next steps

- **No timestamps in Book-Crossing.** The split is random per user, which is easier than predicting the future.
  MovieLens uses a time split.
- **EASE on large catalogues.** It needs an items × items inverse, so on Book-Crossing it uses the 8,000 most
  read books (`--ease-max-items`). Item-kNN, ALS and the content model cover all books.
- **Several ISBNs per book.** Book-Crossing gives each edition its own ISBN, so a reader of one edition can be
  recommended another edition of the same book. Merging editions (same normalised title and author) before
  training is the fix.
- **Implicit vs explicit.** All Book-Crossing interactions are treated as equal "read" signals for top-10.
  Weighting explicit high ratings more is a natural next step.
- **Next steps:**
  - neural models (two-tower, or sequential models such as SASRec on MovieLens)
  - re-ranking for diversity
  - richer content features (book descriptions or embeddings)
  - online A/B testing, because offline NDCG doesn't always predict live clicks
