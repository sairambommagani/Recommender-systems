"""Recommender models, from simplest to strongest.

All top-N models share one interface so they can be compared fairly:

    m.fit(X)                X = users x items sparse interaction matrix (1 = interacted)
    m.score(X_rows)         -> dense (rows x items) scores for any users, including users that were
                            never seen in training: they are passed as their interaction rows
                            ("fold-in"), which is how the API serves brand-new users without retraining
    m.explain(x_row, item)  -> which history items pushed `item` up ("because you liked ...")

Models
  Popularity      most-interacted items; the floor every model must beat
  ItemKNN         item-item cosine similarity on co-occurrence, top-k neighbours per item
  EASE            Embarrassingly Shallow AutoEncoder (Steck 2019): closed-form item-item weights,
                  B = I - P / diag(P) with P = (X'X + lambda I)^-1; a very strong baseline
  ImplicitALS     matrix factorisation for implicit feedback (Hu, Koren & Volinsky 2008), alternating
                  least squares with confidence 1 + alpha for observed interactions
  ContentTFIDF    content-based: TF-IDF of item text (title, author, publisher); a user's profile is
                  the mean of the items they read. Works for books nobody has read yet (cold start)
  Hybrid          per-user standardised blend of a collaborative model and the content model

Rating prediction
  BiasedMF        r_ui = mu + b_u + b_i + p_u . q_i, fitted by ALS with count-weighted L2
                  regularisation. factors=0 gives the classic "baseline biases" model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _csr(X) -> sp.csr_matrix:
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)


# ----------------------------------------------------------------------------- top-N models
class Recommender:
    name = "base"

    def fit(self, X):
        raise NotImplementedError

    def score(self, X_rows) -> np.ndarray:
        raise NotImplementedError

    def explain(self, x_row, item: int, top: int = 3) -> list[tuple[int, float]]:
        return []


class Popularity(Recommender):
    name = "popularity"

    def fit(self, X):
        self.pop = np.asarray(_csr(X).sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, X_rows):
        return np.tile(self.pop, (X_rows.shape[0], 1))


class ItemKNN(Recommender):
    """Item-based collaborative filtering. sim(i, j) = co-occurrence / sqrt(n_i * n_j) (cosine);
    each item keeps only its `k` most similar items, which removes noise and keeps it fast."""
    name = "item_knn"

    def __init__(self, k: int = 100):
        self.k = k

    def fit(self, X):
        X = _csr(X).astype(np.float32)
        X.data[:] = 1.0
        norms = np.sqrt(np.asarray(X.sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        S = (X.T @ X).tocsr().astype(np.float32)
        S.setdiag(0)
        S.eliminate_zeros()
        S = sp.diags(1 / norms) @ S @ sp.diags(1 / norms)
        S = S.tocsr()
        rows, cols, vals = [], [], []
        for i in range(S.shape[0]):                       # keep top-k neighbours of each item
            a, b = S.indptr[i], S.indptr[i + 1]
            if b == a:
                continue
            idx, v = S.indices[a:b], S.data[a:b]
            if len(v) > self.k:
                keep = np.argpartition(-v, self.k - 1)[: self.k]
                idx, v = idx[keep], v[keep]
            rows.append(np.full(len(idx), i))
            cols.append(idx)
            vals.append(v)
        n = S.shape[0]
        if rows:
            self.N = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(n, n), dtype=np.float32)
        else:
            self.N = sp.csr_matrix((n, n), dtype=np.float32)
        self.NT = self.N.T.tocsr()                        # score_ui = sum_j x_uj * sim(j -> i), j in N(i)
        return self

    def score(self, X_rows):
        X_rows = _csr(X_rows).astype(np.float32)
        return np.asarray((X_rows @ self.NT).todense())

    def explain(self, x_row, item, top=3):
        hist = _csr(x_row).indices
        contrib = np.asarray(self.N[item, hist].todense()).ravel()
        order = np.argsort(-contrib)[:top]
        return [(int(hist[o]), float(contrib[o])) for o in order if contrib[o] > 0]


class EASE(Recommender):
    """EASE. One matrix inverse of size items x items, so for large catalogues it is trained on
    the `max_items` most popular items (the rest get no score from this model)."""
    name = "ease"

    def __init__(self, lam: float = 300.0, max_items: int = 8000):
        self.lam = lam
        self.max_items = max_items

    def fit(self, X):
        X = _csr(X).astype(np.float32)
        X.data[:] = 1.0
        n_items = X.shape[1]
        pop = np.asarray(X.sum(axis=0)).ravel()
        if n_items > self.max_items:
            self.items = np.sort(np.argsort(-pop)[: self.max_items])
        else:
            self.items = np.arange(n_items)
        Xs = X[:, self.items]
        dtype = np.float64 if len(self.items) <= 4000 else np.float32
        G = (Xs.T @ Xs).toarray().astype(dtype)
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        del G
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        self.n_items = n_items
        return self

    def score(self, X_rows):
        X_rows = _csr(X_rows).astype(np.float32)
        X_rows.data[:] = 1.0
        out = np.full((X_rows.shape[0], self.n_items), -np.inf, dtype=np.float32)
        out[:, self.items] = np.asarray(X_rows[:, self.items] @ self.B)
        return out

    def explain(self, x_row, item, top=3):
        pos = np.searchsorted(self.items, item)
        if pos >= len(self.items) or self.items[pos] != item:
            return []
        hist = _csr(x_row).indices
        hist = hist[np.isin(hist, self.items)]
        hpos = np.searchsorted(self.items, hist)
        contrib = self.B[hpos, pos]
        order = np.argsort(-contrib)[:top]
        return [(int(hist[o]), float(contrib[o])) for o in order if contrib[o] > 0]


class ImplicitALS(Recommender):
    """Hu-Koren-Volinsky implicit ALS. Every (user, item) cell is a 0/1 preference; observed cells
    get confidence 1 + alpha, unobserved cells confidence 1. Users and items alternate closed-form
    least-squares solves. Scoring a user = solving their vector against the fixed item factors."""
    name = "implicit_als"

    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 20.0, iters: int = 12, seed: int = 0):
        self.factors, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve(self, X: sp.csr_matrix, Y: np.ndarray) -> np.ndarray:
        """For each row u of X: x_u = (Y'Y + alpha Y_u'Y_u + reg I)^-1 Y_u'(1 + alpha) 1."""
        k = Y.shape[1]
        YtY = Y.T @ Y
        reg = self.reg * np.eye(k)
        out = np.zeros((X.shape[0], k), dtype=np.float64)
        for u in range(X.shape[0]):
            a, b = X.indptr[u], X.indptr[u + 1]
            if a == b:
                continue
            Yu = Y[X.indices[a:b]]
            A = YtY + self.alpha * (Yu.T @ Yu) + reg
            out[u] = np.linalg.solve(A, (1 + self.alpha) * Yu.sum(axis=0))
        return out

    def fit(self, X):
        X = _csr(X).astype(np.float64)
        X.data[:] = 1.0
        rng = np.random.default_rng(self.seed)
        self.V = rng.normal(0, 0.01, (X.shape[1], self.factors))
        XT = X.T.tocsr()
        for _ in range(self.iters):
            U = self._solve(X, self.V)
            self.V = self._solve(XT, U)
        self.V = self.V.astype(np.float64)
        return self

    def score(self, X_rows):
        X_rows = _csr(X_rows).astype(np.float64)
        X_rows.data[:] = 1.0
        return (self._solve(X_rows, self.V) @ self.V.T).astype(np.float32)


class ContentTFIDF(Recommender):
    """Content-based recommender from item text. Needs no other users at all, so it is the model
    for brand-new books (no ratings yet) and a useful signal for users with very short histories."""
    name = "content"

    def __init__(self, max_features: int = 50000):
        self.max_features = max_features

    def fit(self, X, texts: list[str] | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        self.vec = TfidfVectorizer(max_features=self.max_features, min_df=2, sublinear_tf=True,
                                   token_pattern=r"(?u)\b\w[\w']+\b", dtype=np.float32)
        self.T = normalize(self.vec.fit_transform(texts)).tocsr()          # items x vocab
        self.TT = self.T.T.tocsr()
        return self

    def profile(self, X_rows):
        from sklearn.preprocessing import normalize
        return normalize(_csr(X_rows).astype(np.float32) @ self.T)

    def score(self, X_rows):
        return np.asarray((self.profile(X_rows) @ self.TT).todense(), dtype=np.float32)

    def similar(self, item: int, n: int = 10) -> list[tuple[int, float]]:
        s = np.asarray((self.T[item] @ self.TT).todense()).ravel()
        s[item] = -1
        order = np.argsort(-s)[:n]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]

    def explain(self, x_row, item, top=3):
        hist = _csr(x_row).indices
        s = np.asarray((self.T[hist] @ self.T[item].T).todense()).ravel()
        order = np.argsort(-s)[:top]
        return [(int(hist[o]), float(s[o])) for o in order if s[o] > 0]


def standardize_rows(S: np.ndarray) -> np.ndarray:
    """Per-user z-score so models with different score scales can be blended; -inf -> row minimum."""
    S = np.array(S, dtype=np.float32, copy=True)
    finite = np.isfinite(S)
    big = np.where(finite, S, np.nan)
    mn = np.nanmin(np.where(finite.any(1, keepdims=True), big, 0), axis=1, keepdims=True)
    S = np.where(finite, S, mn)
    mu, sd = S.mean(axis=1, keepdims=True), S.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (S - mu) / sd


class Hybrid(Recommender):
    """score = w * z(collaborative) + (1 - w) * z(content). w is chosen on validation data."""
    name = "hybrid"

    def __init__(self, cf: Recommender, content: ContentTFIDF, w: float = 0.8):
        self.cf, self.content, self.w = cf, content, w

    def fit(self, X):
        return self

    def score(self, X_rows):
        return self.w * standardize_rows(self.cf.score(X_rows)) + \
            (1 - self.w) * standardize_rows(self.content.score(X_rows))

    def explain(self, x_row, item, top=3):
        return self.cf.explain(x_row, item, top) or self.content.explain(x_row, item, top)


def rank_rows(S: np.ndarray) -> np.ndarray:
    """Per-user percentile rank in [0, 1] (ties share their average rank; -inf ranks lowest).
    Unlike a z-score, a handful of large scores can't swamp the blend."""
    from scipy.stats import rankdata
    S = np.where(np.isfinite(S), S, -np.finfo(np.float32).max)
    return (rankdata(S, axis=1, method="average") / S.shape[1]).astype(np.float32)


class PopularityBlend(Recommender):
    """score = w * rank(model) + (1 - w) * rank(popularity). For users with very short histories
    (new sign-ups) a collaborative model alone has little to go on, and mixing in popularity is the
    standard remedy. Ranks, not raw scores, are blended: with 5 known movies item-kNN gives most
    movies exactly 0, and standardising such scores lets a few outliers dominate. w is tuned on
    validation users."""
    name = "popularity_blend"

    def __init__(self, cf: Recommender, popularity: "Popularity", w: float = 0.5):
        self.cf, self.popularity, self.w = cf, popularity, w

    def fit(self, X):
        return self

    def score(self, X_rows):
        return self.w * rank_rows(self.cf.score(X_rows)) + (1 - self.w) * rank_rows(self.popularity.score(X_rows))

    def explain(self, x_row, item, top=3):
        return self.cf.explain(x_row, item, top)


# ----------------------------------------------------------------------------- cold start: group popularity
class GroupPopularity:
    """Most popular books within the user's group (country, or country + age band), falling back
    to the coarser group, then to global popularity, when the group has too little data."""

    def __init__(self, min_interactions: int = 50):
        self.min = min_interactions

    def fit(self, inter: pd.DataFrame, users: pd.DataFrame, n_items: int, keys: list[str]):
        self.keys, self.n_items = keys, n_items
        d = inter.merge(users, on="user", how="left").fillna({"country": "unknown", "age_band": "unknown"})
        self.glob = np.bincount(d["item"], minlength=n_items).astype(np.float32)
        self.tables = {}
        for level in range(1, len(keys) + 1):
            ks = keys[:level]
            for g, part in d.groupby(ks):
                if len(part) >= self.min:
                    g = g if isinstance(g, tuple) else (g,)
                    self.tables[tuple(g)] = np.bincount(part["item"], minlength=n_items).astype(np.float32)
        return self

    def scores_for(self, attrs: dict) -> np.ndarray:
        for level in range(len(self.keys), 0, -1):
            key = tuple(attrs.get(k, "unknown") for k in self.keys[:level])
            if key in self.tables:
                return self.tables[key] + 1e-3 * self.glob          # break ties by global popularity
        return self.glob


# ----------------------------------------------------------------------------- rating prediction
class BiasedMF:
    """Explicit-rating model r_ui = mu + b_u + b_i + p_u . q_i, fitted with ALS.

    Factor regularisation is weighted by the number of ratings (ALS-WR); biases get a constant
    penalty, so a movie with 3 ratings is pulled toward the average and one with 300 is not.
    factors=0 is the "baseline biases" model (user and item effects only)."""

    def __init__(self, factors: int = 20, reg: float = 0.1, bias_reg: float = 10.0, iters: int = 15,
                 rating_range=(1, 5), seed: int = 0):
        self.factors, self.reg, self.bias_reg, self.iters = factors, reg, bias_reg, iters
        self.lo_hi, self.seed = rating_range, seed

    @property
    def name(self):
        return "biases" if self.factors == 0 else f"biased_mf_{self.factors}"

    @staticmethod
    def _group(keys, n):
        order = np.argsort(keys, kind="stable")
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(indptr, keys + 1, 1)
        return order, np.cumsum(indptr)

    def _half_step(self, order, indptr, other_idx, target, F_other, b_other, n):
        """Solve [f, b] for each entity on one side, the other side fixed."""
        k = self.factors
        F = np.zeros((n, k))
        b = np.zeros(n)
        for e in range(n):
            a, z = indptr[e], indptr[e + 1]
            if a == z:
                continue
            rows = order[a:z]
            o = other_idx[rows]
            A = np.hstack([F_other[o], np.ones((z - a, 1))])
            t = target[rows] - b_other[o]
            M = A.T @ A + np.diag(np.r_[np.full(k, self.reg * (z - a)), self.bias_reg])
            x = np.linalg.solve(M, A.T @ t)
            F[e], b[e] = x[:k], x[k]
        return F, b

    def fit(self, users, items, ratings, n_users, n_items):
        users, items = np.asarray(users), np.asarray(items)
        r = np.asarray(ratings, dtype=np.float64)
        self.mu = float(r.mean())
        t = r - self.mu
        rng = np.random.default_rng(self.seed)
        self.P = np.zeros((n_users, self.factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.factors))
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        uo, up = self._group(users, n_users)
        io, ip = self._group(items, n_items)
        self.n_u = np.diff(up)
        self.n_i = np.diff(ip)
        for _ in range(self.iters):
            self.P, self.bu = self._half_step(uo, up, items, t, self.Q, self.bi, n_users)
            self.Q, self.bi = self._half_step(io, ip, users, t, self.P, self.bu, n_items)
        return self

    def predict(self, users, items) -> np.ndarray:
        users, items = np.asarray(users), np.asarray(items)
        ku = (users >= 0) & (users < len(self.bu))
        ki = (items >= 0) & (items < len(self.bi))
        ku[ku] = self.n_u[users[ku]] > 0
        ki[ki] = self.n_i[items[ki]] > 0
        p = np.full(len(users), self.mu)
        p[ku] += self.bu[users[ku]]
        p[ki] += self.bi[items[ki]]
        both = ku & ki
        if self.factors:
            p[both] += np.einsum("ij,ij->i", self.P[users[both]], self.Q[items[both]])
        return np.clip(p, *self.lo_hi)

    def explain(self, user: int, item: int) -> dict:
        known_u = 0 <= user < len(self.bu) and self.n_u[user] > 0
        known_i = 0 <= item < len(self.bi) and self.n_i[item] > 0
        inter = float(self.P[user] @ self.Q[item]) if (known_u and known_i and self.factors) else 0.0
        return {"global_mean": round(self.mu, 3),
                "user_bias": round(float(self.bu[user]), 3) if known_u else 0.0,
                "item_bias": round(float(self.bi[item]), 3) if known_i else 0.0,
                "user_item_match": round(inter, 3)}
