"""The two model ladders shared by both tasks (top-N ranking and rating prediction)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .evaluation import evaluate_rating, evaluate_ranking, to_matrix, tune
from .models import EASE, BiasedMF, ImplicitALS, ItemKNN, Popularity

FACTORIES = {"item_knn": ItemKNN, "ease": EASE, "implicit_als": ImplicitALS}


def run_ranking_ladder(tr, val_rel, train_full, test_rel, n_users, n_items, grids: dict, k: int = 10, log=print):
    """Popularity, then every model in `grids`: tuned on validation, refitted on all of train,
    scored once on test. Returns (test results, fitted models, best params, tuning tables)."""
    X_tr = to_matrix(tr, n_users, n_items)
    X_full = to_matrix(train_full, n_users, n_items)
    pop = np.asarray(X_full.sum(0)).ravel()
    results, fitted, best_params, tables = {}, {}, {}, {}

    t0 = time.time()
    m = Popularity().fit(X_full)
    results["popularity"] = evaluate_ranking(m, X_full, test_rel, k, pop)
    results["popularity"]["fit_seconds"] = round(time.time() - t0, 1)
    fitted["popularity"] = m
    log(f"  popularity      NDCG@{k} {results['popularity'][f'ndcg@{k}']:.4f}  "
        f"recall@{k} {results['popularity'][f'recall@{k}']:.4f}  coverage {results['popularity']['coverage']:.3f}")

    for name, grid in grids.items():
        log(f"  tuning {name} on validation ({len(grid)} settings)")
        params, table = tune(name, grid, FACTORIES[name], X_tr, val_rel, k, log)
        t0 = time.time()
        m = FACTORIES[name](**params).fit(X_full)
        fit_s = round(time.time() - t0, 1)
        r = evaluate_ranking(m, X_full, test_rel, k, pop)
        r["fit_seconds"] = fit_s
        results[name], fitted[name], best_params[name], tables[name] = r, m, params, table
        log(f"  {name:15s} NDCG@{k} {r[f'ndcg@{k}']:.4f}  recall@{k} {r[f'recall@{k}']:.4f}  "
            f"coverage {r['coverage']:.3f}  (best {params})")
    return results, fitted, best_params, tables


def run_rating_ladder(tr, val, train_full, test, n_users, n_items, bias_regs: list[float], mf_grid: list[dict],
                      rating_range, log=print):
    """Global mean -> user/item biases -> biased matrix factorisation (factors, reg tuned on validation)."""
    results, models = {}, {}
    mu = float(train_full["rating"].mean())
    from .metrics import mae, rmse
    results["global_mean"] = {"rmse": round(rmse(test["rating"], np.full(len(test), mu)), 4),
                              "mae": round(mae(test["rating"], np.full(len(test), mu)), 4), "n": int(len(test))}
    log(f"  global mean     RMSE {results['global_mean']['rmse']:.4f}")

    def fit(params, df):
        return BiasedMF(rating_range=rating_range, **params).fit(df["user"], df["item"], df["rating"], n_users, n_items)

    # biases: tune reg only
    rows = []
    for reg in bias_regs:
        r = evaluate_rating(fit({"factors": 0, "bias_reg": reg, "iters": 6}, tr), val)
        rows.append({"factors": 0, "bias_reg": reg, "val_rmse": r["rmse"]})
        log(f"    biases bias_reg={reg}: val RMSE {r['rmse']:.4f}")
    b = min(rows, key=lambda d: d["val_rmse"])
    m = fit({"factors": 0, "bias_reg": b["bias_reg"], "iters": 6}, train_full)
    results["biases"] = {**evaluate_rating(m, test), "bias_reg": b["bias_reg"]}
    models["biases"] = m
    log(f"  biases          RMSE {results['biases']['rmse']:.4f}")

    table = []
    for p in mf_grid:
        t0 = time.time()
        r = evaluate_rating(fit({**p, "bias_reg": b["bias_reg"]}, tr), val)
        table.append({**p, "val_rmse": r["rmse"], "fit_seconds": round(time.time() - t0, 1)})
        log(f"    biased_mf {p}: val RMSE {r['rmse']:.4f}")
    best = min(table, key=lambda d: d["val_rmse"])
    params = {**{k: best[k] for k in mf_grid[0]}, "bias_reg": b["bias_reg"]}
    m = fit(params, train_full)
    results["biased_mf"] = {**evaluate_rating(m, test), **params}
    models["biased_mf"] = m
    log(f"  biased_mf       RMSE {results['biased_mf']['rmse']:.4f}  (best {params})")
    return results, models, rows + table


def long_tail(df: pd.DataFrame) -> dict:
    c = df["item"].value_counts().to_numpy()
    top = max(1, int(round(len(c) * 0.1)))
    return {"share_of_interactions_on_top10pct_items": round(float(c[:top].sum() / c.sum()), 3),
            "items_with_1_interaction": int((c == 1).sum()),
            "median_interactions_per_item": float(np.median(c)),
            "median_interactions_per_user": float(df["user"].value_counts().median())}
