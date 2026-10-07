"""End-to-end tests on synthetic data with the real file formats (no downloads, ~1 minute)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

from recsys import api
from recsys.evaluation import k_core, split_per_user
from recsys.metrics import RankingAccumulator, top_k
from recsys.models import EASE, BiasedMF, ImplicitALS, ItemKNN, Popularity


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def art(tmp_path_factory):
    """Train exactly as a user would (`python -m ...` in a separate process), so the saved models
    must load in the API process."""
    out = tmp_path_factory.mktemp("artifacts")
    for mod, sub in (("recsys.movielens", "movielens"), ("recsys.books", "books")):
        subprocess.run([sys.executable, "-m", mod, "--synthetic", "--fast", "--out", str(out / sub)],
                       cwd=ROOT, check=True, capture_output=True)
    ml = json.loads((out / "movielens" / "metrics.json").read_text())
    bk = json.loads((out / "books" / "metrics.json").read_text())
    return out, ml, bk


def test_metrics_by_hand():
    acc = RankingAccumulator(k=3, n_items=5, item_popularity=np.ones(5))
    acc.add(np.array([[0, 1, 2]]), [np.array([1, 4])])
    r = acc.result()
    assert r["precision@3"] == pytest.approx(1 / 3, abs=1e-3)
    assert r["recall@3"] == pytest.approx(0.5)
    assert r["ndcg@3"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)), abs=1e-3)
    assert list(top_k(np.array([[0.1, 0.9, 0.5]]), 2)[0]) == [1, 2]


def test_split_and_kcore():
    df = pd.DataFrame({"user": [0] * 10 + [1] * 2 + [2], "item": list(range(10)) + [0, 1, 0], "ts": range(13)})
    tr, te = split_per_user(df, 0.2, by_time=True)
    assert set(te[te.user == 0]["item"]) == {8, 9}                 # latest ratings are the test set
    assert len(te[te.user == 2]) == 0                               # single-rating user stays in train
    assert k_core(df, 2, 2)["user"].nunique() == 2


def test_models_learn_structure():
    rng = np.random.default_rng(0)
    # two taste groups: users 0-49 read items 0-19, users 50-99 read items 20-39
    rows, cols = [], []
    for u in range(100):
        base = 0 if u < 50 else 20
        for i in rng.choice(20, 8, replace=False):
            rows.append(u)
            cols.append(base + i)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(100, 40))
    probe = sp.csr_matrix((np.ones(3), ([0, 0, 0], [21, 22, 23])), shape=(1, 40))   # new user, group-2 taste
    for m in (ItemKNN(10), EASE(10.0), ImplicitALS(8, 1.0, 10.0, 8)):
        s = m.fit(X).score(probe)[0]
        s[[21, 22, 23]] = -np.inf
        assert np.argmax(s) >= 20, m.name                           # recommends from the right group
    assert Popularity().fit(X).score(probe).shape == (1, 40)
    mf = BiasedMF(factors=4, iters=5).fit(np.array(rows), np.array(cols), np.where(np.array(cols) < 20, 5, 1), 100, 40)
    assert mf.predict([0], [5])[0] > mf.predict([0], [25])[0]


def test_pipelines_beat_popularity(art):
    _, ml, bk = art
    assert ml["ranking"]["ease"]["ndcg@10"] > ml["ranking"]["popularity"]["ndcg@10"]
    assert ml["rating_prediction"]["biased_mf"]["rmse"] < ml["rating_prediction"]["global_mean"]["rmse"]
    c = bk["data"]["cleaning"]
    assert c["ratings_with_isbn_not_in_books_removed"] > 0 and c["books_invalid_year_set_missing"] >= 3
    assert c["users_impossible_age_set_missing"] >= 2
    cb = bk["cold_start_books"]["results"]
    assert cb["content"]["ndcg@10"] > cb["random"]["ndcg@10"]       # content model handles brand-new books


def test_api(art, monkeypatch):
    out, _, _ = art
    monkeypatch.setattr(api, "ART", out)
    api._cache.clear()
    c = TestClient(api.app)
    assert c.get("/health").json() == {"movielens": "ready", "books": "ready"}
    r = c.get("/movies/users/1/recommendations?k=5").json()
    assert len(r["recommendations"]) == 5 and "because_you_liked" in r["recommendations"][0]
    r = c.post("/movies/new-user", json={"liked_movie_ids": [1, 2, 3], "k": 5}).json()
    assert len(r["recommendations"]) == 5
    r = c.post("/movies/predict-rating", json={"user_id": 1, "movie_id": 2}).json()
    assert 1 <= r["predicted_rating"] <= 5
    assert c.get("/movies/1/similar").status_code == 200
    assert c.get("/movies/99999/similar").status_code == 404
    bk = api._cache["books"] if "books" in api._cache else api.books()
    uid = int(bk.user_ids[0])
    r = c.get(f"/books/users/{uid}/recommendations").json()
    assert r["strategy"].startswith("hybrid") and len(r["recommendations"]) == 10
    r = c.get("/books/users/99999999/recommendations").json()             # unknown user -> popularity
    assert r["strategy"].startswith("cold start")
    r = c.post("/books/new-user", json={"liked_isbns": [], "country": "germany", "age": 30}).json()
    assert r["strategy"].startswith("cold start")
    isbn = bk.isbns[0]
    r = c.post("/books/new-user", json={"liked_isbns": [isbn.lower()]}).json()
    assert r["strategy"].startswith("hybrid")
    assert "readers_also_read" in c.get(f"/books/{isbn}/similar").json()
    cold = next(s for s in bk.all_books.index if s not in bk.pos)              # book too rare for the k-core
    r = c.get(f"/books/{cold}/similar").json()
    assert r["readers"] == 0 and len(r["similar_by_title_author_publisher"]) > 0
    assert c.post("/books/predict-rating", json={"user_id": uid, "isbn": isbn}).status_code == 200
    assert c.get("/model-info").json()["books"]["best_ranking_model"]
