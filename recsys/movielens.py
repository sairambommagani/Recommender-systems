"""Task 1 - MovieLens 100K: rating prediction + top-10 movie recommendations.

    python -m recsys.movielens --raw data/movielens          # folder containing u.data, u.item, u.user
    python -m recsys.movielens --synthetic                   # offline smoke run

Data: 100,000 ratings (1-5 stars) from 943 users on 1,682 movies, every user has >= 20 ratings.

Split: for each user, their LATEST 20% of ratings (by timestamp) are the test set. This mimics
deployment - learn from what a user did, predict what they do next - and avoids the leakage of a
random split (training on ratings made after the ones you test on).

Two tasks:
  * rating prediction  global mean -> user/item biases -> biased matrix factorisation (ALS)   [RMSE]
  * top-10 ranking     popularity -> item-kNN -> implicit ALS -> EASE                         [NDCG@10]
    A test movie counts as relevant when the user rated it 4 or 5.
  * cold start         new users (never seen in training) who rate 5 movies at sign-up: the
                       models fold them in without retraining; a model + popularity blend,
                       weighted on validation users, is used for them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .evaluation import evaluate_fold_in, reindex, split_per_user, summarize_split, to_matrix
from .models import EASE, BiasedMF, ImplicitALS, ItemKNN, Popularity, PopularityBlend
from .pipeline import FACTORIES, long_tail, run_ranking_ladder, run_rating_ladder

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
          "War", "Western"]
RELEVANT = 4            # rating >= 4 = "liked"
K = 10


def _find(raw: Path, name: str) -> Path:
    hits = sorted(raw.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {raw} - unzip the Kaggle MovieLens 100K download there")
    return hits[0]


def load_movielens(raw) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = Path(raw)
    ratings = pd.read_csv(_find(raw, "u.data"), sep="\t", names=["user", "item", "rating", "ts"])
    items = pd.read_csv(_find(raw, "u.item"), sep="|", encoding="latin-1", header=None,
                        names=["item", "title", "release_date", "video_release", "imdb_url"] + GENRES)
    items["genres"] = items[GENRES].apply(lambda r: [g for g in GENRES if r[g] == 1], axis=1)
    items["year"] = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce").dt.year
    items = items[["item", "title", "year", "genres"]]
    users = pd.read_csv(_find(raw, "u.user"), sep="|", names=["user", "age", "gender", "occupation", "zip"])
    return ratings, items, users


def _fold_in_matrix(cu: pd.DataFrame, n_items: int, n_given: int):
    """Each user's FIRST n_given ratings are known (sign-up); their later liked movies are hidden."""
    import scipy.sparse as sp
    rows, cols, hidden = [], [], []
    for u, g in cu.sort_values(["user", "ts"]).groupby("user"):
        given, rest = g.iloc[:n_given], g.iloc[n_given:]
        rows += [len(hidden)] * len(given)
        cols += given["item"].tolist()
        hidden.append(rest.loc[rest["rating"] >= RELEVANT, "item"].to_numpy())
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(hidden), n_items)), hidden


def cold_user_test(ratings: pd.DataFrame, n_users: int, n_items: int, params: dict, best: str,
                   frac: float = 0.2, n_given: int = 5, seed: int = 7, log=print) -> dict:
    """New users. 20% of users are removed from training entirely and the models are refitted
    without them. Each model then sees only a user's first 5 ratings (what they'd rate at sign-up)
    and must find the movies they liked later. Half of these users (validation) choose the weight
    of the model + popularity blend; the other half (test) give the reported numbers."""
    rng = np.random.default_rng(seed)
    users = ratings["user"].unique()
    cold = rng.choice(users, size=max(2, int(len(users) * frac)), replace=False)
    val_users, test_users = set(cold[: len(cold) // 2].tolist()), set(cold[len(cold) // 2:].tolist())
    warm = ratings[~ratings["user"].isin(set(cold.tolist()))]
    X = to_matrix(warm, n_users, n_items)
    pop_model = Popularity().fit(X)
    models = {"popularity": pop_model}
    for name, p in params.items():
        models[name] = FACTORIES[name](**p).fit(X)
    pop = np.asarray(X.sum(0)).ravel()
    G_val, h_val = _fold_in_matrix(ratings[ratings["user"].isin(val_users)], n_items, n_given)
    G_te, h_te = _fold_in_matrix(ratings[ratings["user"].isin(test_users)], n_items, n_given)
    grid = []
    for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        r = evaluate_fold_in({"b": PopularityBlend(models[best], pop_model, w)}, G_val, h_val, K, pop)["b"]
        grid.append({"w_model": w, f"val_ndcg@{K}": r[f"ndcg@{K}"]})
        log(f"    blend {w:.1f} x {best} + {1 - w:.1f} x popularity: val NDCG@{K} {r[f'ndcg@{K}']:.4f}")
    w_best = max(grid, key=lambda d: d[f"val_ndcg@{K}"])["w_model"]
    blend_name = f"{best}+popularity"
    models[blend_name] = PopularityBlend(models[best], pop_model, w_best)
    res = evaluate_fold_in(models, G_te, h_te, K, pop)
    for name, r in res.items():
        log(f"  new users ({n_given} ratings known) {name:22s} NDCG@{K} {r[f'ndcg@{K}']:.4f}  recall@{K} {r[f'recall@{K}']:.4f}")
    return {"n_cold_users_validation": len(h_val), "n_cold_users_test": len(h_te), "ratings_given": n_given,
            "blend": blend_name, "blend_w_model": w_best, "blend_tuning": grid, "results": res}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/movielens")
    ap.add_argument("--out", default="artifacts/movielens")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fast", action="store_true", help="smaller hyper-parameter grids")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:5.0f}s]", *x, flush=True)

    if a.synthetic:
        from scripts.make_synthetic import movielens_frames
        ratings, items, users = movielens_frames()
    else:
        ratings, items, users = load_movielens(a.raw)
    ratings, maps = reindex(ratings)
    n_users, n_items = len(maps["user"]), len(maps["item"])
    items = items.set_index("item").reindex(maps["item"]).reset_index()
    log(f"{len(ratings)} ratings, {n_users} users, {n_items} movies, density {len(ratings) / n_users / n_items:.4f}")

    train_full, test = split_per_user(ratings, 0.2, by_time=True)
    tr, val = split_per_user(train_full, 0.125, by_time=True)
    data = {"ratings": int(len(ratings)), "users": n_users, "items": n_items,
            "rating_distribution": {int(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()},
            "mean_rating": round(float(ratings["rating"].mean()), 3), **long_tail(ratings),
            "split": "per user by time: latest 20% of each user's ratings = test; validation = latest 12.5% of the rest",
            **summarize_split(train_full, test, n_users, n_items)}

    # ---------------- rating prediction
    log("RATING PREDICTION (RMSE, 1-5 stars)")
    mf_grid = [{"factors": f, "reg": r, "iters": 12} for f in (10, 30) for r in (0.1, 0.2, 0.4)]
    if a.fast:
        mf_grid = mf_grid[:2]
    rating_res, rating_models, rating_table = run_rating_ladder(
        tr, val, train_full, test, n_users, n_items, [0.5, 1.0, 2.0, 5.0, 15.0], mf_grid, (1, 5), log)

    # ---------------- top-N
    log(f"TOP-{K} RECOMMENDATION (relevant = test movie rated >= {RELEVANT}; all unseen movies ranked)")
    grids = {"item_knn": [{"k": k} for k in (20, 50, 100, 200)],
             "implicit_als": [{"factors": f, "reg": r, "alpha": al, "iters": 10}
                              for f in (32, 64) for r in (1.0, 10.0, 50.0) for al in (10.0, 40.0)],
             "ease": [{"lam": l} for l in (50.0, 100.0, 250.0, 500.0, 1000.0)]}
    if a.fast:
        grids = {k: v[:2] for k, v in grids.items()}
    val_rel = val[val["rating"] >= RELEVANT]
    test_rel = test[test["rating"] >= RELEVANT]
    rank_res, rank_models, best_params, tuning = run_ranking_ladder(
        tr, val_rel, train_full, test_rel, n_users, n_items, grids, K, log)
    val_best = {n: max(r[f"val_ndcg@{K}"] for r in t) for n, t in tuning.items()}
    best = max(val_best, key=val_best.get)                       # chosen on VALIDATION, not test
    log(f"best ranking model on validation: {best}")

    log("COLD START: new users not seen in training, 5 ratings known")
    cold = cold_user_test(ratings, n_users, n_items, best_params, best, log=log)

    # ---------------- serving models: refit on ALL ratings
    X_all = to_matrix(ratings, n_users, n_items)
    serve = {"ranker": FACTORIES[best](**best_params[best]).fit(X_all), "ranker_name": best,
             "similar": ItemKNN(**best_params["item_knn"]).fit(X_all),
             "rating": BiasedMF(rating_range=(1, 5), **{k: v for k, v in rating_res["biased_mf"].items()
                                                         if k in ("factors", "reg", "iters", "bias_reg")})
             .fit(ratings["user"], ratings["item"], ratings["rating"], n_users, n_items),
             "popularity": Popularity().fit(X_all), "new_user_w": cold["blend_w_model"]}
    joblib.dump(serve, out / "models.joblib")
    joblib.dump(X_all, out / "interactions.joblib")
    items.assign(n_ratings=np.asarray(X_all.sum(0)).ravel().astype(int),
                 mean_rating=ratings.groupby("item")["rating"].mean().reindex(range(n_items)).round(2).to_numpy()) \
        .to_json(out / "items.json", orient="records")
    report = {"task": "movielens-100k", "data": data, "rating_prediction": rating_res, "rating_tuning": rating_table,
              "ranking": rank_res, "ranking_best_params": best_params, "ranking_tuning": tuning,
              "best_ranking_model": best, "cold_start_users": cold, "k": K, "relevant_threshold": RELEVANT}
    (out / "metrics.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps({"user_ids": maps["user"].tolist(), "item_ids": maps["item"].tolist(),
                                               "ranker": best, "trained_at": time.strftime("%Y-%m-%d %H:%M")}))
    b = rank_res[best]
    log(f"BEST top-{K}: {best} NDCG@{K} {b[f'ndcg@{K}']:.4f} recall@{K} {b[f'recall@{K}']:.4f} | "
        f"BEST rating: biased_mf RMSE {rating_res['biased_mf']['rmse']:.4f}. Artifacts in {out}/")
    return report


if __name__ == "__main__":
    main()
