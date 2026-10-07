"""Splitting, filtering and evaluation shared by both tasks.

Protocol (the same for every model, so the comparison is fair):
  1. split each user's interactions into train and test (by time when timestamps exist, else at random)
  2. carve a validation set out of TRAIN the same way; every hyper-parameter (neighbours, lambda,
     factors, blend weight) is chosen on validation, then the model is refitted on all of train
  3. the test set is used once, for the final numbers
  4. top-N: each test user's items seen in training are excluded, then ALL remaining items are ranked
     (no sampled negatives, which are known to inflate metrics)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .metrics import RankingAccumulator, mae, rmse, top_k


# ----------------------------------------------------------------------------- data shaping
def k_core(df: pd.DataFrame, min_user: int, min_item: int, max_rounds: int = 20) -> pd.DataFrame:
    """Repeatedly drop users with < min_user and items with < min_item interactions (until stable).
    Standard for sparse data: users/items with 1-2 interactions can't be learned or evaluated."""
    for _ in range(max_rounds):
        n = len(df)
        uc = df["user"].map(df["user"].value_counts())
        ic = df["item"].map(df["item"].value_counts())
        df = df[(uc >= min_user) & (ic >= min_item)]
        if len(df) == n:
            break
    return df


def reindex(df: pd.DataFrame, cols=("user", "item")) -> tuple[pd.DataFrame, dict]:
    """Map raw ids to 0..n-1. Returns the frame and {col: array of raw ids by index}."""
    df = df.copy()
    maps = {}
    for c in cols:
        codes, uniques = pd.factorize(df[c], sort=True)
        df[c] = codes
        maps[c] = np.asarray(uniques)
    return df, maps


def split_per_user(df: pd.DataFrame, test_frac: float, by_time: bool, seed: int = 42,
                   min_train: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `test_frac` of each user's interactions (their LATEST ones if by_time).
    Users with too few interactions keep everything in train."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    if by_time:
        df["_r"] = df.groupby("user")["ts"].rank(method="first", ascending=True)
    else:
        df["_r"] = rng.random(len(df))
        df["_r"] = df.groupby("user")["_r"].rank(method="first")
    n = df.groupby("user")["item"].transform("size")
    n_test = np.floor(n * test_frac).astype(int)
    n_test = np.where((n_test == 0) & (n >= min_train + 1), 1, n_test)
    n_test = np.minimum(n_test, np.maximum(n - min_train, 0))
    is_test = df["_r"] > (n - n_test)
    return df[~is_test].drop(columns="_r"), df[is_test].drop(columns="_r")


def to_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    X = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df["user"].to_numpy(), df["item"].to_numpy())),
                      shape=(n_users, n_items))
    X.data[:] = 1.0                                       # duplicates -> 1
    return X


# ----------------------------------------------------------------------------- top-N evaluation
def rank_users(model, X_known: sp.csr_matrix, k: int, batch: int = 1024, exclude_known: bool = True):
    """Yield (row positions, top-k item indices) for every row of X_known."""
    for s in range(0, X_known.shape[0], batch):
        rows = X_known[s:s + batch]
        S = model.score(rows)
        if exclude_known:
            r, c = rows.nonzero()
            S[r, c] = -np.inf
        yield s, top_k(S, k)


def evaluate_ranking(model, X_train: sp.csr_matrix, test: pd.DataFrame, k: int = 10,
                     item_pop: np.ndarray | None = None, batch: int = 1024) -> dict:
    """Rank all unseen items for each user that has relevant test items; `test` holds only the
    relevant (user, item) pairs."""
    t0 = time.time()
    rel = test.groupby("user")["item"].apply(lambda s: s.to_numpy())
    users = rel.index.to_numpy()
    pop = item_pop if item_pop is not None else np.asarray(X_train.sum(0)).ravel()
    acc = RankingAccumulator(k, X_train.shape[1], pop)
    X_known = X_train[users]
    for s, recs in rank_users(model, X_known, k, batch):
        acc.add(recs, [rel.iloc[i] for i in range(s, s + len(recs))])
    out = acc.result()
    out["seconds"] = round(time.time() - t0, 1)
    return out


def evaluate_fold_in(models: dict, given: sp.csr_matrix, hidden: list[np.ndarray], k: int,
                     item_pop: np.ndarray) -> dict:
    """Cold-start users: the models never saw them; each user shows only a few items (`given`,
    e.g. picked at sign-up) and we check whether their remaining items (`hidden`) are recommended."""
    out = {}
    for name, m in models.items():
        acc = RankingAccumulator(k, given.shape[1], item_pop)
        for s, recs in rank_users(m, given, k):
            acc.add(recs, hidden[s:s + len(recs)])
        out[name] = acc.result()
    return out


def evaluate_rating(model, test: pd.DataFrame) -> dict:
    p = model.predict(test["user"].to_numpy(), test["item"].to_numpy())
    return {"rmse": round(rmse(test["rating"], p), 4), "mae": round(mae(test["rating"], p), 4),
            "n": int(len(test))}


def tune(name: str, grid: list[dict], factory, X_tr: sp.csr_matrix, val_rel: pd.DataFrame, k: int,
         log=print, metric: str = "ndcg") -> tuple[dict, list[dict]]:
    """Fit factory(**params) on the training part, score on validation; return best params + table."""
    rows = []
    for params in grid:
        t0 = time.time()
        m = factory(**params).fit(X_tr)
        r = evaluate_ranking(m, X_tr, val_rel, k)
        rows.append({**params, f"val_{metric}@{k}": r[f"{metric}@{k}"], "fit_seconds": round(time.time() - t0, 1)})
        log(f"    {name} {params}: val {metric}@{k} {r[f'{metric}@{k}']:.4f}")
    best = max(rows, key=lambda d: d[f"val_{metric}@{k}"])
    return {p: best[p] for p in grid[0]}, rows


def summarize_split(train: pd.DataFrame, test: pd.DataFrame, n_users: int, n_items: int) -> dict:
    return {"users": int(n_users), "items": int(n_items), "train_interactions": int(len(train)),
            "test_interactions": int(len(test)),
            "density": round(float((len(train) + len(test)) / (n_users * n_items)), 6)}
