"""Recommendation API for both tasks.

    uvicorn recsys.api:app --port 8003         (then open http://localhost:8003/docs)

MovieLens (Task 1)
  GET  /movies/users/{user_id}/recommendations   top-N for a known user + "because you liked ..."
  POST /movies/new-user                          top-N for a new user from a few liked movies (no retraining)
  POST /movies/predict-rating                    predicted stars for (user, movie) with a breakdown
  GET  /movies/{movie_id}/similar                "people who liked this also liked"
  GET  /movies/search?q=                         find movie ids by title
Book-Crossing (Task 2)
  GET  /books/users/{user_id}/recommendations    known reader -> hybrid model; user without enough
                                                 history -> most popular in their country / age group
  POST /books/new-user                           a few liked ISBNs and/or country + age
  POST /books/predict-rating                     predicted 1-10 rating
  GET  /books/{isbn}/similar                     readers-also-read + similar by title/author/publisher
                                                 (works for books nobody has rated yet)
  GET  /books/search?q=
GET /model-info, GET /health
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .books import AGE_BANDS, AGE_LABELS, norm_isbn
from .models import Hybrid, PopularityBlend

ART = Path(os.environ.get("RS_ARTIFACTS", "artifacts"))


def _row(items: list[int], n: int) -> sp.csr_matrix:
    items = sorted(set(int(i) for i in items))
    return sp.csr_matrix((np.ones(len(items), np.float32), (np.zeros(len(items), int), items)), shape=(1, n))


def _top(scores: np.ndarray, exclude, k: int) -> list[int]:
    s = scores.astype(np.float64).copy()
    s[list(exclude)] = -np.inf
    s[~np.isfinite(s)] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    idx = np.argpartition(-s, k - 1)[:k] if k else np.array([], int)
    return idx[np.argsort(-s[idx])].tolist()


# ----------------------------------------------------------------------------- MovieLens
class Movies:
    def __init__(self, art: Path):
        if not (art / "models.joblib").exists():
            raise FileNotFoundError(f"{art}/models.joblib missing - run `python -m recsys.movielens` first")
        self.m = joblib.load(art / "models.joblib")
        self.X = joblib.load(art / "interactions.joblib").tocsr()
        meta = json.loads((art / "meta.json").read_text())
        self.metrics = json.loads((art / "metrics.json").read_text())
        self.user_ids = np.asarray(meta["user_ids"])
        self.item_ids = np.asarray(meta["item_ids"])
        self.items = pd.read_json(art / "items.json", orient="records")
        self.n = len(self.item_ids)

    def uidx(self, user_id: int) -> int:
        i = np.searchsorted(self.user_ids, user_id)
        if i >= len(self.user_ids) or self.user_ids[i] != user_id:
            raise HTTPException(404, f"user {user_id} not found (MovieLens user ids are 1..943)")
        return int(i)

    def iidx(self, movie_id: int) -> int:
        i = np.searchsorted(self.item_ids, movie_id)
        if i >= len(self.item_ids) or self.item_ids[i] != movie_id:
            raise HTTPException(404, f"movie {movie_id} not found (movie ids are 1..1682)")
        return int(i)

    def card(self, i: int, **extra) -> dict:
        r = self.items.iloc[i]
        return {"movie_id": int(self.item_ids[i]), "title": r["title"], "year": None if pd.isna(r["year"]) else int(r["year"]),
                "genres": list(r["genres"]), **extra}

    def recommend(self, row: sp.csr_matrix, k: int, new_user: bool = False) -> list[dict]:
        ranker = self.m["ranker"]
        if new_user:          # short history: model + popularity blend (weight tuned on validation users)
            ranker = PopularityBlend(ranker, self.m["popularity"], self.m.get("new_user_w", 1.0))
        s = ranker.score(row)[0]
        out = []
        for i in _top(s, row.indices, k):
            why = ranker.explain(row, i) or self.m["similar"].explain(row, i)
            out.append(self.card(i, score=round(float(s[i]), 4),
                                 because_you_liked=[self.items.iloc[j]["title"] for j, _ in why]))
        return out


# ----------------------------------------------------------------------------- Book-Crossing
class Books:
    def __init__(self, art: Path):
        if not (art / "models.joblib").exists():
            raise FileNotFoundError(f"{art}/models.joblib missing - run `python -m recsys.books` first")
        self.m = joblib.load(art / "models.joblib")
        self.X = joblib.load(art / "interactions.joblib").tocsr()
        meta = json.loads((art / "meta.json").read_text())
        self.metrics = json.loads((art / "metrics.json").read_text())
        self.user_ids = np.asarray(meta["user_ids"])
        self.isbns = np.asarray(meta["isbns"])
        self.pos = {s: i for i, s in enumerate(self.isbns)}
        self.items = pd.read_json(art / "items.json", orient="records", dtype={"isbn": str})
        self.all_books = pd.read_parquet(art / "books_all.parquet").set_index("isbn")
        self.users = pd.read_parquet(art / "users_all.parquet").set_index("user")
        self.hybrid = Hybrid(self.m["cf"], self.m["content"], self.m["hybrid_w"])
        self.n = len(self.isbns)

    def card(self, i: int, **extra) -> dict:
        r = self.items.iloc[i]
        return {"isbn": self.isbns[i], "title": html.unescape(str(r["title"])), "author": html.unescape(str(r["author"])),
                "publisher": html.unescape(str(r["publisher"])),
                "year": None if pd.isna(r["year"]) else int(r["year"]), "readers": int(r["n_readers"]), **extra}

    def uidx(self, user_id: int) -> Optional[int]:
        i = np.searchsorted(self.user_ids, user_id)
        return int(i) if i < len(self.user_ids) and self.user_ids[i] == user_id else None

    def attrs(self, user_id: Optional[int] = None, country: Optional[str] = None, age: Optional[float] = None) -> dict:
        a = {"country": "unknown", "age_band": "unknown"}
        if user_id is not None and user_id in self.users.index:
            r = self.users.loc[user_id]
            a = {"country": r["country"], "age_band": r["age_band"]}
        if country:
            a["country"] = country.strip().lower()
        if age is not None:
            a["age_band"] = str(pd.cut([age], AGE_BANDS, labels=AGE_LABELS, right=False)[0])
        return a

    def recommend(self, row: sp.csr_matrix, k: int, attrs: dict) -> dict:
        if row.nnz == 0:
            s = self.m["group_pop"].scores_for(attrs)
            return {"strategy": f"cold start: most read by readers in country={attrs['country']}, "
                                f"age={attrs['age_band']} (falls back to wider groups when data is thin)",
                    "recommendations": [self.card(i, score=round(float(s[i]), 1)) for i in _top(s, [], k)]}
        s = self.hybrid.score(row)[0]
        recs = []
        for i in _top(s, row.indices, k):
            why = self.hybrid.explain(row, i)
            recs.append(self.card(i, score=round(float(s[i]), 3),
                                  because_you_read=[self.items.iloc[j]["title"] for j, _ in why]))
        return {"strategy": f"hybrid: {self.m['hybrid_w']} x {self.m['cf_name']} (collaborative) + "
                            f"{1 - self.m['hybrid_w']:.1f} x content (title/author/publisher), on {row.nnz} known books",
                "recommendations": recs}


app = FastAPI(title="Recommender Systems API", version="1.0",
              description="Task 1: MovieLens 100K (ratings + top-N). Task 2: Book-Crossing (implicit + explicit "
                          "feedback, cold start, hybrid). Each recommendation says why it was made.")
_cache: dict = {}


def movies() -> Movies:
    if "movies" not in _cache:
        try:
            _cache["movies"] = Movies(ART / "movielens")
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _cache["movies"]


def books() -> Books:
    if "books" not in _cache:
        try:
            _cache["books"] = Books(ART / "books")
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _cache["books"]


@app.get("/health")
def health():
    out = {}
    for name, f in (("movielens", movies), ("books", books)):
        try:
            f()
            out[name] = "ready"
        except HTTPException as e:
            out[name] = f"not ready: {e.detail}"
    return out


@app.get("/model-info")
def model_info():
    out = {}
    for name, f in (("movielens", movies), ("books", books)):
        try:
            m = f().metrics
        except HTTPException as e:
            out[name] = {"status": e.detail}
            continue
        out[name] = {"data": m["data"], "best_ranking_model": m["best_ranking_model"],
                     "ranking_test": m["ranking"], "rating_test": m["rating_prediction"],
                     "cold_start_users": m.get("cold_start_users"), "cold_start_books": m.get("cold_start_books")}
    return out


# --------------------------------------------------------------- movie endpoints
class NewMovieUser(BaseModel):
    liked_movie_ids: list[int] = Field(..., min_length=1, max_length=200)
    k: int = Field(10, ge=1, le=50)
    model_config = {"json_schema_extra": {"examples": [{"liked_movie_ids": [50, 172, 181], "k": 10}]}}


class MovieRating(BaseModel):
    user_id: int
    movie_id: int
    model_config = {"json_schema_extra": {"examples": [{"user_id": 196, "movie_id": 50}]}}


@app.get("/movies/users/{user_id}/recommendations")
def movie_recs(user_id: int, k: int = Query(10, ge=1, le=50)):
    mv = movies()
    u = mv.uidx(user_id)
    row = mv.X[u]
    hist = mv.items.iloc[row.indices].assign(i=row.indices).sort_values("n_ratings", ascending=False)
    return {"user_id": user_id, "model": mv.m["ranker_name"], "movies_rated": int(row.nnz),
            "sample_of_history": hist["title"].head(5).tolist(), "recommendations": mv.recommend(row, k)}


@app.post("/movies/new-user")
def movie_new_user(body: NewMovieUser):
    mv = movies()
    idx = [mv.iidx(i) for i in body.liked_movie_ids]
    w = mv.m.get("new_user_w", 1.0)
    recs = mv.recommend(_row(idx, mv.n), body.k, new_user=True)
    if w == 0:
        strategy = ("most popular movies the user hasn't rated: on held-out new users with 5 ratings, popularity "
                    "gave the best top-10 (NDCG) - see metrics.json cold_start_users")
        for r in recs:
            r.pop("because_you_liked", None)
    else:
        strategy = f"{w:.1f} x {mv.m['ranker_name']} + {1 - w:.1f} x popularity (rank blend tuned on validation new users)"
    return {"strategy": strategy, "note": "new user handled at request time; no retraining",
            "liked": [mv.items.iloc[i]["title"] for i in idx], "recommendations": recs}


@app.post("/movies/predict-rating")
def movie_rating(body: MovieRating):
    mv = movies()
    u, i = mv.uidx(body.user_id), mv.iidx(body.movie_id)
    p = float(mv.m["rating"].predict([u], [i])[0])
    return {**mv.card(i), "user_id": body.user_id, "predicted_rating": round(p, 2),
            "already_rated": bool(mv.X[u, i]), "breakdown": mv.m["rating"].explain(u, i)}


@app.get("/movies/{movie_id}/similar")
def movie_similar(movie_id: int, k: int = Query(10, ge=1, le=50)):
    mv = movies()
    i = mv.iidx(movie_id)
    row = mv.m["similar"].N[i]
    order = np.argsort(-row.data)[:k]
    return {**mv.card(i), "similar": [mv.card(int(row.indices[o]), similarity=round(float(row.data[o]), 3)) for o in order]}


@app.get("/movies/search")
def movie_search(q: str = Query(..., min_length=2), k: int = 10):
    mv = movies()
    hit = mv.items[mv.items["title"].str.contains(q, case=False, regex=False)].sort_values("n_ratings", ascending=False)
    return [mv.card(int(i), ratings=int(r)) for i, r in zip(hit.index[:k], hit["n_ratings"][:k])]


# --------------------------------------------------------------- book endpoints
class NewBookUser(BaseModel):
    liked_isbns: list[str] = Field(default_factory=list, max_length=200)
    country: Optional[str] = None
    age: Optional[float] = Field(None, ge=5, le=100)
    k: int = Field(10, ge=1, le=50)
    model_config = {"json_schema_extra": {"examples": [
        {"liked_isbns": ["0316666343", "0385504209"], "country": "usa", "age": 30, "k": 10},
        {"liked_isbns": [], "country": "germany", "age": 25, "k": 10}]}}


class BookRating(BaseModel):
    user_id: int
    isbn: str
    model_config = {"json_schema_extra": {"examples": [{"user_id": 11676, "isbn": "0316666343"}]}}


@app.get("/books/users/{user_id}/recommendations")
def book_recs(user_id: int, k: int = Query(10, ge=1, le=50)):
    bk = books()
    u = bk.uidx(user_id)
    row = bk.X[u] if u is not None else sp.csr_matrix((1, bk.n), dtype=np.float32)
    known = user_id in bk.users.index or u is not None
    out = bk.recommend(row, k, bk.attrs(user_id))
    return {"user_id": user_id, "known_user": bool(known), "books_in_history": int(row.nnz),
            "history_sample": [bk.items.iloc[j]["title"] for j in row.indices[:5]], **out}


@app.post("/books/new-user")
def book_new_user(body: NewBookUser):
    bk = books()
    clean = norm_isbn(pd.Series(body.liked_isbns, dtype=str)).tolist() if body.liked_isbns else []
    idx = [bk.pos[s] for s in clean if s in bk.pos]
    unknown = [s for s in clean if s not in bk.pos]
    out = bk.recommend(_row(idx, bk.n) if idx else sp.csr_matrix((1, bk.n), dtype=np.float32), body.k,
                       bk.attrs(None, body.country, body.age))
    return {"liked_used": [bk.items.iloc[i]["title"] for i in idx], "liked_not_in_model": unknown, **out}


@app.post("/books/predict-rating")
def book_rating(body: BookRating):
    bk = books()
    isbn = norm_isbn(pd.Series([body.isbn])).iloc[0]
    if isbn not in bk.pos:
        raise HTTPException(404, f"ISBN {isbn} is not among the books the model was trained on")
    u, i = bk.uidx(body.user_id), bk.pos[isbn]
    uu = -1 if u is None else u
    p = float(bk.m["rating"].predict([uu], [i])[0])
    return {**bk.card(i), "user_id": body.user_id, "predicted_rating_1_to_10": round(p, 2),
            "user_known": u is not None, "breakdown": bk.m["rating"].explain(uu, i) if u is not None else
            {"global_mean": round(bk.m["rating"].mu, 3), "note": "unknown user: average + book effect only"}}


@app.get("/books/{isbn}/similar")
def book_similar(isbn: str, k: int = Query(10, ge=1, le=50)):
    bk = books()
    isbn = norm_isbn(pd.Series([isbn])).iloc[0]
    content = bk.m["content"]
    if isbn in bk.pos:
        i = bk.pos[isbn]
        row = bk.m["similar"].N[i]
        order = np.argsort(-row.data)[:k]
        return {**bk.card(i), "readers_also_read": [bk.card(int(row.indices[o]), similarity=round(float(row.data[o]), 3)) for o in order],
                "similar_by_title_author_publisher": [bk.card(j, similarity=round(s, 3)) for j, s in content.similar(i, k)]}
    if isbn not in bk.all_books.index:
        raise HTTPException(404, f"ISBN {isbn} not found")
    r = bk.all_books.loc[isbn]
    from .books import book_texts
    from sklearn.preprocessing import normalize
    v = normalize(content.vec.transform(book_texts(pd.DataFrame([r.to_dict()]).fillna(""))))
    s = np.asarray((v @ content.TT).todense()).ravel()
    order = np.argsort(-s)[:k]
    return {"isbn": isbn, "title": r["title"], "author": r["author"], "readers": 0,
            "note": "cold-start book (too few readers for collaborative filtering): similar by title/author/publisher only",
            "similar_by_title_author_publisher": [bk.card(int(j), similarity=round(float(s[j]), 3)) for j in order if s[j] > 0]}


@app.get("/books/search")
def book_search(q: str = Query(..., min_length=2), k: int = 10):
    bk = books()
    hit = bk.items[bk.items["title"].str.contains(q, case=False, regex=False) |
                   bk.items["author"].str.contains(q, case=False, regex=False)].sort_values("n_readers", ascending=False)
    return [bk.card(int(i)) for i in hit.index[:k]]
