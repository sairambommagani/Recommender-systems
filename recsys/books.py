"""Task 2 - Book-Crossing: top-10 book recommendations, rating prediction and cold start.

    python -m recsys.books --raw data/books        # folder with Books.csv, Ratings.csv, Users.csv
    python -m recsys.books --synthetic             # offline smoke run

Why this is harder than MovieLens
  * ~1.1M ratings over 278k users and 271k books: >99.99% of the user x book matrix is empty,
    and most users/books have only 1-2 interactions.
  * Book-Rating 0 means "read / interacted, but not rated" (implicit), 1-10 is an explicit rating.
    About 60% of rows are 0s. Treating 0 as "hated it" would be a serious mistake.
  * Messy data: ISBNs that aren't in Books.csv, lowercase 'x' check digits, publisher names in the
    year column, years 0 or in the future, ages 0 or 244, a few "users" with thousands of rows.
  * No timestamps, so the split is random per user.

What we build
  * top-10 (all interactions, 0 included, are implicit "read" signals): popularity, item-kNN,
    implicit ALS, EASE, content-based (TF-IDF of title/author/publisher) and a hybrid
  * rating prediction (explicit 1-10 only): global mean, biases, biased matrix factorisation
  * cold start, measured separately on users and books held out of training entirely:
      - new user, no history  -> most popular in their country / age group
      - new user, 3 books     -> models fold the user in without retraining
      - new book, no readers  -> content model (title, author, publisher)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .evaluation import (evaluate_fold_in, evaluate_ranking, k_core, reindex, split_per_user,
                         summarize_split, to_matrix, tune)
from .metrics import RankingAccumulator, top_k
from .models import ContentTFIDF, GroupPopularity, Hybrid, ItemKNN, Popularity
from .pipeline import FACTORIES, long_tail, run_ranking_ladder, run_rating_ladder

K = 10
MAX_YEAR = 2006         # data was collected in 2004
AGE_BANDS = [0, 18, 25, 35, 50, 120]
AGE_LABELS = ["<18", "18-24", "25-34", "35-49", "50+"]


# ----------------------------------------------------------------------------- loading + cleaning
def _find(raw: Path, name: str) -> Path:
    for p in raw.rglob("*"):
        if p.name.lower() in (name.lower(), name.lower().replace(".csv", "") + ".csv",
                              "bx-" + name.lower(), "bx_" + name.lower()):
            return p
    hits = [p for p in raw.rglob("*.csv") if name.lower().replace(".csv", "").rstrip("s") in p.name.lower()]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {raw} - unzip the Kaggle Book-Crossing download there")
    return hits[0]


def _read(path: Path) -> pd.DataFrame:
    head = path.open("rb").readline().decode("latin-1")
    sep = ";" if head.count(";") > head.count(",") else ","
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, dtype=str, on_bad_lines="skip",
                               escapechar="\\" if sep == ";" else None, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not read {path}")


def norm_isbn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"[^0-9X]", "", regex=True)


def load_books(raw) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = Path(raw)
    books = _read(_find(raw, "Books.csv"))
    ratings = _read(_find(raw, "Ratings.csv"))
    users = _read(_find(raw, "Users.csv"))
    books.columns = ["isbn", "title", "author", "year", "publisher"] + list(books.columns[5:])
    ratings.columns = ["user", "isbn", "rating"]
    users.columns = ["user", "location", "age"][: len(users.columns)]
    rep = {"raw_books": len(books), "raw_ratings": len(ratings), "raw_users": len(users)}

    # books
    books["isbn"] = norm_isbn(books["isbn"])
    books = books.drop_duplicates("isbn")
    year = pd.to_numeric(books["year"], errors="coerce")
    bad_year = year.isna() | (year <= 0) | (year > MAX_YEAR)
    rep["books_invalid_year_set_missing"] = int(bad_year.sum())
    books["year"] = year.where(~bad_year)
    for c in ("title", "author", "publisher"):
        books[c] = books[c].fillna("").str.strip()
    books = books[["isbn", "title", "author", "year", "publisher"] +
                  [c for c in books.columns if c.lower().startswith("image")][:2]]

    # users
    users["user"] = pd.to_numeric(users["user"], errors="coerce")
    users = users.dropna(subset=["user"]).astype({"user": int})
    age = pd.to_numeric(users.get("age"), errors="coerce")
    bad_age = age.notna() & ((age < 5) | (age > 100))
    rep["users_impossible_age_set_missing"] = int(bad_age.sum())
    users["age"] = age.where(~bad_age)
    country = users.get("location", pd.Series("", index=users.index)).fillna("").str.split(",").str[-1] \
        .str.strip().str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    users["country"] = country.where(~country.isin(["", "na", "n a", "nan"]), "unknown")
    users["age_band"] = pd.cut(users["age"], AGE_BANDS, labels=AGE_LABELS, right=False).astype(str) \
        .replace("nan", "unknown")
    users = users[["user", "country", "age", "age_band"]]

    # ratings
    ratings["user"] = pd.to_numeric(ratings["user"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user", "rating"]).astype({"user": int, "rating": int})
    ratings["isbn"] = norm_isbn(ratings["isbn"])
    n = len(ratings)
    ratings = ratings.sort_values("rating").drop_duplicates(["user", "isbn"], keep="last")
    rep["duplicate_ratings_removed"] = n - len(ratings)
    known = ratings["isbn"].isin(set(books["isbn"]))
    rep["ratings_with_isbn_not_in_books_removed"] = int((~known).sum())
    ratings = ratings[known & ratings["rating"].between(0, 10)]
    rep["clean_ratings"] = len(ratings)
    rep["implicit_share (rating 0)"] = round(float((ratings["rating"] == 0).mean()), 3)
    rep["clean_users_with_ratings"] = int(ratings["user"].nunique())
    rep["clean_books_with_ratings"] = int(ratings["isbn"].nunique())
    return books, ratings, users, rep


def book_texts(books: pd.DataFrame) -> list[str]:
    """Text for the content model: title words + the author and publisher as single tokens
    (so 'John Grisham' is one feature, not 'john' + 'grisham') + the publication decade."""
    tok = lambda s: "x_" + re.sub(r"\W+", "_", str(s).lower()).strip("_") if s else ""
    dec = pd.to_numeric(books["year"], errors="coerce").map(lambda y: f"decade_{int(y) // 10 * 10}" if pd.notna(y) else "")
    return [f"{t} {tok(a).replace('x_', 'author_', 1)} {tok(p).replace('x_', 'publisher_', 1)} {d}"
            for t, a, p, d in zip(books["title"], books["author"], books["publisher"], dec)]


# ----------------------------------------------------------------------------- cold-start evaluations
def cold_item_eval(content: ContentTFIDF, X_users: sp.csr_matrix, cold_inter: pd.DataFrame,
                   cold_items: np.ndarray, log=print) -> dict:
    """New books nobody has read in training. For each user who later read some of them, rank ONLY
    the cold books by content similarity to the user's history; random ranking is the baseline."""
    pos = {it: j for j, it in enumerate(cold_items)}
    rel = cold_inter.groupby("user")["item"].apply(lambda s: np.array([pos[i] for i in s]))
    users = rel.index.to_numpy()
    users = users[np.asarray(X_users[users].sum(1)).ravel() > 0]
    rel = rel.loc[users]
    rng = np.random.default_rng(0)
    out = {}
    for name in ("content", "random"):
        acc = RankingAccumulator(K, len(cold_items), np.ones(len(cold_items)))
        for s in range(0, len(users), 1024):
            rows = X_users[users[s:s + 1024]]
            S = content.score(rows)[:, cold_items] if name == "content" else rng.random((rows.shape[0], len(cold_items)))
            acc.add(top_k(S, K), [rel.iloc[i] for i in range(s, s + rows.shape[0])])
        out[name] = acc.result()
        log(f"  new books ({len(cold_items)} never seen in training) {name:8s} NDCG@{K} {out[name][f'ndcg@{K}']:.4f}  "
            f"recall@{K} {out[name][f'recall@{K}']:.4f}")
    return {"n_cold_books": int(len(cold_items)), "users_evaluated": int(len(users)), "results": out}


def cold_user_eval(models: dict, group_pops: dict, cold_df: pd.DataFrame, users: pd.DataFrame,
                   n_items: int, item_pop: np.ndarray, n_given: int = 3, seed: int = 3, log=print) -> dict:
    rng = np.random.default_rng(seed)
    ua = users.set_index("user")
    res = {}
    # (a) no history at all: only country / age known
    rel_all = cold_df.groupby("user")["item"].apply(lambda s: s.to_numpy())
    zero = {}
    for name, gp in group_pops.items():
        acc = RankingAccumulator(K, n_items, item_pop)
        uid = rel_all.index.to_numpy()
        S = np.stack([gp.scores_for(ua.loc[u].to_dict() if u in ua.index else {}) for u in uid])
        acc.add(top_k(S, K), list(rel_all.to_numpy()))
        zero[name] = acc.result()
        log(f"  new users, no books known      {name:22s} NDCG@{K} {zero[name][f'ndcg@{K}']:.4f}  recall@{K} {zero[name][f'recall@{K}']:.4f}")
    res["no_history"] = zero
    # (b) a few books known (e.g. picked at sign-up), the rest hidden
    rows, cols, hidden = [], [], []
    for u, g in cold_df.groupby("user"):
        if len(g) <= n_given:
            continue
        idx = rng.permutation(len(g))
        given, rest = g["item"].to_numpy()[idx[:n_given]], g["item"].to_numpy()[idx[n_given:]]
        rows += [len(hidden)] * n_given
        cols += given.tolist()
        hidden.append(rest)
    G = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(hidden), n_items))
    few = evaluate_fold_in(models, G, hidden, K, item_pop)
    for name, r in few.items():
        log(f"  new users, {n_given} books known       {name:22s} NDCG@{K} {r[f'ndcg@{K}']:.4f}  recall@{K} {r[f'recall@{K}']:.4f}")
    res[f"{n_given}_books_known"] = few
    res["n_cold_users"] = int(rel_all.size)
    return res


# ----------------------------------------------------------------------------- main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/books")
    ap.add_argument("--out", default="artifacts/books")
    ap.add_argument("--min-user", type=int, default=5, help="k-core: keep users with >= this many interactions")
    ap.add_argument("--min-item", type=int, default=5, help="k-core: keep books with >= this many interactions")
    ap.add_argument("--max-user", type=int, default=500, help="cap very heavy users (bulk imports) by sampling")
    ap.add_argument("--ease-max-items", type=int, default=8000, help="EASE uses the N most popular books (RAM)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fast", action="store_true", help="smaller hyper-parameter grids")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:5.0f}s]", *x, flush=True)
    rng = np.random.default_rng(42)

    raw = a.raw
    if a.synthetic:
        from scripts.make_synthetic import write_books
        raw = write_books(out / "_synthetic_raw")
    books, ratings, users, rep = load_books(raw)
    log(f"cleaning: {rep}")

    # ---- k-core on all interactions (implicit 0 + explicit 1-10), then cap bulk users
    inter = ratings.rename(columns={"isbn": "item"})
    core = k_core(inter, a.min_user, a.min_item)
    heavy = core["user"].map(core["user"].value_counts()) > a.max_user
    n_heavy = core.loc[heavy, "user"].nunique()
    if n_heavy:
        keep = core[heavy].groupby("user", group_keys=False).sample(n=a.max_user, random_state=0)
        core = pd.concat([core[~heavy], keep])
    rep.update({"kcore_min_user": a.min_user, "kcore_min_item": a.min_item, "kcore_interactions": len(core),
                "kcore_users": int(core["user"].nunique()), "kcore_books": int(core["item"].nunique()),
                "heavy_users_capped": int(n_heavy), "max_interactions_per_user": a.max_user})
    log(f"k-core ({a.min_user}/{a.min_item}): {len(core)} interactions, {rep['kcore_users']} users, "
        f"{rep['kcore_books']} books; {n_heavy} heavy users capped at {a.max_user}")
    core, maps = reindex(core)
    n_users, n_items = len(maps["user"]), len(maps["item"])
    item_meta = books.set_index("isbn").reindex(maps["item"]).reset_index().rename(columns={"index": "isbn"})
    user_attr = users[users["user"].isin(maps["user"])].copy()
    user_attr["user"] = np.searchsorted(maps["user"], user_attr["user"])

    # ---- cold-start hold-outs: 5% of books and 10% of users never enter training
    cold_items = np.sort(rng.choice(n_items, size=max(1, int(n_items * 0.05)), replace=False))
    is_cold_item = core["item"].isin(cold_items)
    warm = core[~is_cold_item]
    cand = warm["user"].value_counts()
    cand = cand[cand >= 5].index.to_numpy()
    cold_users = rng.choice(cand, size=max(1, int(n_users * 0.10)) if len(cand) > 10 else 0, replace=False)
    cold_user_df = warm[warm["user"].isin(cold_users)]
    warm = warm[~warm["user"].isin(cold_users)]
    train_full, test = split_per_user(warm, 0.2, by_time=False, seed=42)
    tr, val = split_per_user(train_full, 0.125, by_time=False, seed=43)
    cold_item_df = core[is_cold_item & core["user"].isin(train_full["user"])]
    data = {"cleaning": rep, **long_tail(core), **summarize_split(train_full, test, n_users, n_items),
            "split": "random per user (no timestamps): 20% test, validation = 12.5% of the rest",
            "cold_books_held_out": int(len(cold_items)), "cold_users_held_out": int(len(cold_users)),
            "rating_distribution": {int(k): int(v) for k, v in core["rating"].value_counts().sort_index().items()}}

    # ---- content model (text only: allowed to see every book, including cold ones)
    texts = book_texts(item_meta.fillna({"title": "", "author": "", "publisher": ""}))
    content = ContentTFIDF().fit(None, texts)

    # ---- top-N ladder
    log(f"TOP-{K} RECOMMENDATION (every interaction, 0 included, is a 'read' signal; all unseen books ranked)")
    grids = {"item_knn": [{"k": k} for k in (50, 200)],
             "implicit_als": [{"factors": 64, "reg": r, "alpha": al, "iters": 10} for r in (1.0, 10.0) for al in (10.0, 40.0)],
             "ease": [{"lam": l, "max_items": a.ease_max_items} for l in (100.0, 500.0, 2000.0)]}
    if a.fast:
        grids = {k: v[:2] for k, v in grids.items()}
    rank_res, rank_models, best_params, tuning = run_ranking_ladder(tr, val, train_full, test, n_users, n_items, grids, K, log)
    X_tr, X_full = to_matrix(tr, n_users, n_items), to_matrix(train_full, n_users, n_items)
    pop = np.asarray(X_full.sum(0)).ravel()
    rank_res["content"] = evaluate_ranking(content, X_full, test, K, pop)
    log(f"  content         NDCG@{K} {rank_res['content'][f'ndcg@{K}']:.4f}  recall@{K} {rank_res['content'][f'recall@{K}']:.4f}  "
        f"coverage {rank_res['content']['coverage']:.3f}")
    val_best = {n: max(r[f"val_ndcg@{K}"] for r in t) for n, t in tuning.items()}
    cf_name = max(val_best, key=val_best.get)
    cf_val = FACTORIES[cf_name](**best_params[cf_name]).fit(X_tr)
    w_rows = []
    for w in (1.0, 0.9, 0.8, 0.7, 0.5):
        r = evaluate_ranking(Hybrid(cf_val, content, w), X_tr, val, K)
        w_rows.append({"w_collaborative": w, f"val_ndcg@{K}": r[f"ndcg@{K}"]})
        log(f"    hybrid {cf_name} weight {w}: val NDCG@{K} {r[f'ndcg@{K}']:.4f}")
    w_best = max(w_rows, key=lambda d: d[f"val_ndcg@{K}"])["w_collaborative"]
    hybrid = Hybrid(rank_models[cf_name], content, w_best)
    rank_res["hybrid"] = {**evaluate_ranking(hybrid, X_full, test, K, pop), "collaborative": cf_name, "w_collaborative": w_best}
    tuning["hybrid"] = w_rows
    log(f"  hybrid          NDCG@{K} {rank_res['hybrid'][f'ndcg@{K}']:.4f}  recall@{K} {rank_res['hybrid'][f'recall@{K}']:.4f}  "
        f"({w_best} x {cf_name} + {1 - w_best:.1f} x content)")
    val_best["hybrid"] = max(d[f"val_ndcg@{K}"] for d in w_rows)
    best = max(val_best, key=val_best.get)
    log(f"best ranking model on validation: {best}")

    # ---- rating prediction on explicit ratings only
    log("RATING PREDICTION (explicit 1-10 ratings only)")
    ex = lambda d: d[d["rating"] > 0]
    mf_grid = [{"factors": f, "reg": r, "iters": 10} for f in (10, 30) for r in (0.1, 0.3, 1.0)]
    if a.fast:
        mf_grid = mf_grid[:2]
    rating_res, rating_models, rating_table = run_rating_ladder(
        ex(tr), ex(val), ex(train_full), ex(test), n_users, n_items, [2.0, 5.0, 15.0, 40.0], mf_grid, (1, 10), log)

    # ---- cold start
    log("COLD START")
    gp_global = GroupPopularity().fit(train_full, user_attr, n_items, [])
    gp_country = GroupPopularity().fit(train_full, user_attr, n_items, ["country"])
    gp_both = GroupPopularity().fit(train_full, user_attr, n_items, ["country", "age_band"])
    fold_models = {"popularity": rank_models["popularity"], **{n: rank_models[n] for n in grids},
                   "content": content, "hybrid": hybrid}
    cold_users_res = cold_user_eval(fold_models, {"global popularity": gp_global, "country popularity": gp_country,
                                                  "country + age popularity": gp_both},
                                    cold_user_df, user_attr, n_items, pop, log=log) if len(cold_user_df) else {}
    cold_items_res = cold_item_eval(content, X_full, cold_item_df, cold_items, log) if len(cold_item_df) else {}

    # ---- serving: refit on everything in the k-core
    X_all = to_matrix(core, n_users, n_items)
    cf_all = FACTORIES[cf_name](**best_params[cf_name]).fit(X_all)
    ex_all = core[core["rating"] > 0]
    from .models import BiasedMF
    rp = {k: v for k, v in rating_res["biased_mf"].items() if k in ("factors", "reg", "iters", "bias_reg")}
    serve = {"cf": cf_all, "cf_name": cf_name, "hybrid_w": w_best, "content": content,
             "similar": ItemKNN(**best_params["item_knn"]).fit(X_all),
             "rating": BiasedMF(rating_range=(1, 10), **rp).fit(ex_all["user"], ex_all["item"], ex_all["rating"], n_users, n_items),
             "popularity": Popularity().fit(X_all),
             "group_pop": GroupPopularity().fit(core, user_attr, n_items, ["country", "age_band"]),
             "best": best}
    joblib.dump(serve, out / "models.joblib")
    joblib.dump(X_all, out / "interactions.joblib")
    meta_cols = [c for c in item_meta.columns if c in ("isbn", "title", "author", "year", "publisher") or c.lower().startswith("image")]
    item_meta[meta_cols].assign(n_readers=np.asarray(X_all.sum(0)).ravel().astype(int)) \
        .to_json(out / "items.json", orient="records")
    books[["isbn", "title", "author", "year", "publisher"]].to_parquet(out / "books_all.parquet", index=False)
    users.to_parquet(out / "users_all.parquet", index=False)
    report = {"task": "book-crossing", "data": data, "ranking": rank_res, "ranking_best_params": best_params,
              "ranking_tuning": tuning, "best_ranking_model": best, "rating_prediction": rating_res,
              "rating_tuning": rating_table, "cold_start_users": cold_users_res, "cold_start_books": cold_items_res, "k": K}
    (out / "metrics.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps({"user_ids": maps["user"].tolist(), "isbns": maps["item"].tolist(),
                                               "best": best, "cf": cf_name, "trained_at": time.strftime("%Y-%m-%d %H:%M")}))
    b = rank_res[best]
    log(f"BEST top-{K}: {best} NDCG@{K} {b[f'ndcg@{K}']:.4f} recall@{K} {b[f'recall@{K}']:.4f} | "
        f"BEST rating: biased_mf RMSE {rating_res['biased_mf']['rmse']:.4f} (1-10). Artifacts in {out}/")
    return report


if __name__ == "__main__":
    main()
