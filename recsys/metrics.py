"""Evaluation metrics for recommenders.

Two different questions, two families of metrics:

* Rating prediction ("what will user u rate movie i?"): RMSE and MAE on held-out ratings.
* Top-N recommendation ("which 10 items should we show user u?"): for each test user we rank
  ALL items they have not interacted with in training, take the top K, and compare with the
  items they actually interacted with in the test period.
    - Precision@K  share of the K recommendations that were relevant
    - Recall@K     share of the user's relevant test items that made it into the top K
    - NDCG@K       like recall, but a hit at rank 1 counts more than a hit at rank 10 (headline metric)
    - HitRate@K    share of users with at least one relevant item in their top K
    - MAP@K        mean average precision (order-aware precision)
  and two catalogue-level metrics that accuracy alone hides:
    - Coverage     share of the catalogue that appears in anyone's top K (popularity models are ~1-3%)
    - Novelty      mean self-information -log2(popularity) of recommended items (higher = less obvious)
"""
from __future__ import annotations

import numpy as np


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise indices of the k largest scores, best first. scores: (users, items)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1, kind="stable")
    return part[rows, order]


class RankingAccumulator:
    """Collects per-user ranking metrics batch by batch (so huge catalogues can be scored in chunks)."""

    def __init__(self, k: int, n_items: int, item_popularity: np.ndarray):
        self.k = k
        self.n_items = n_items
        pop = np.asarray(item_popularity, float)
        self.self_info = -np.log2(np.clip(pop / max(pop.sum(), 1.0), 1e-12, None))
        self.discount = 1.0 / np.log2(np.arange(2, k + 2))
        self.per_user = {"precision": [], "recall": [], "ndcg": [], "hit": [], "map": [], "novelty": []}
        self.recommended = np.zeros(n_items, dtype=bool)

    def add(self, recs: np.ndarray, relevant: list[np.ndarray]):
        k = self.k
        for r, rel in zip(recs, relevant):
            if len(rel) == 0:
                continue
            hits = np.isin(r, rel)
            n_rel = len(rel)
            dcg = float((hits * self.discount[: len(r)]).sum())
            idcg = float(self.discount[: min(n_rel, k)].sum())
            prec_at = np.cumsum(hits) / np.arange(1, len(r) + 1)
            self.per_user["precision"].append(hits.sum() / k)
            self.per_user["recall"].append(hits.sum() / n_rel)
            self.per_user["ndcg"].append(dcg / idcg if idcg else 0.0)
            self.per_user["hit"].append(float(hits.any()))
            self.per_user["map"].append(float((prec_at * hits).sum() / min(n_rel, k)))
            self.per_user["novelty"].append(float(self.self_info[r].mean()))
            self.recommended[r] = True

    def result(self) -> dict:
        k = self.k
        out = {f"{m}@{k}": round(float(np.mean(v)), 4) if v else 0.0
               for m, v in self.per_user.items() if m != "novelty"}
        out["coverage"] = round(float(self.recommended.mean()), 4)
        out["novelty"] = round(float(np.mean(self.per_user["novelty"])), 3) if self.per_user["novelty"] else 0.0
        out["users_evaluated"] = len(self.per_user["ndcg"])
        return out
