"""Synthetic data with the SAME files and columns as the real datasets (for offline tests).

Users and items get hidden "taste" vectors, so collaborative models have real structure to find;
book titles/authors carry genre words, so the content model has signal too. The Book-Crossing
files also contain the real data's quirks: implicit 0 ratings, ISBNs missing from Books.csv,
non-numeric years, impossible ages, lowercase ISBN check digits and duplicate rows.

    python scripts/make_synthetic.py --books data/books_synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
          "War", "Western"]


def _interactions(n_users, n_items, rng, k=6, per_user=(20, 80), pop_power=1.0):
    U = rng.normal(0, 1, (n_users, k))
    V = rng.normal(0, 1, (n_items, k))
    pop = rng.pareto(1.2, n_items) + 1
    pop = np.log(pop) * pop_power
    rows = []
    for u in range(n_users):
        aff = V @ U[u] / np.sqrt(k)
        logits = 1.5 * aff + pop
        p = np.exp(logits - logits.max())
        p /= p.sum()
        n = int(rng.integers(*per_user))
        items = rng.choice(n_items, size=min(n, n_items), replace=False, p=p)
        rows += [(u, i, aff[i]) for i in items]
    return pd.DataFrame(rows, columns=["u", "i", "aff"]), V


def movielens_frames(n_users=300, n_items=400, seed=0):
    rng = np.random.default_rng(seed)
    df, V = _interactions(n_users, n_items, rng)
    r = np.clip(np.round(3.5 + 1.2 * df["aff"] + rng.normal(0, 0.7, len(df))), 1, 5).astype(int)
    ratings = pd.DataFrame({"user": df["u"] + 1, "item": df["i"] + 1, "rating": r,
                            "ts": 874_700_000 + rng.integers(0, 20_000_000, len(df))})
    g = np.argmax(V[:, :5], axis=1)
    items = pd.DataFrame({"item": np.arange(1, n_items + 1), "title": [f"Movie {i} ({1990 + i % 8})" for i in range(n_items)],
                          "year": 1990 + np.arange(n_items) % 8,
                          "genres": [[GENRES[1 + gi]] for gi in g]})
    users = pd.DataFrame({"user": np.arange(1, n_users + 1), "age": rng.integers(18, 70, n_users),
                          "gender": rng.choice(["M", "F"], n_users), "occupation": "other", "zip": "00000"})
    return ratings, items, users


BOOK_GENRE_WORDS = [["dragon", "kingdom", "magic", "sword"], ["murder", "detective", "case", "night"],
                    ["love", "summer", "heart", "wedding"], ["galaxy", "robot", "star", "planet"],
                    ["war", "soldier", "empire", "history"], ["cook", "kitchen", "garden", "recipes"]]
COUNTRIES = ["usa", "canada", "united kingdom", "germany", "spain", "australia"]


def write_books(out: Path, n_users=1500, n_items=1200, seed=1):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    df, V = _interactions(n_users, n_items, rng, per_user=(3, 40))
    genre = np.argmax(V[:, :len(BOOK_GENRE_WORDS)], axis=1)
    isbn = [f"{100000000 + i:09d}X" if i % 37 == 0 else f"{100000000 + i:09d}{i % 10}" for i in range(n_items)]
    authors = [f"Author{g}_{rng.integers(0, 25)}" for g in genre]
    titles = [" ".join(rng.choice(BOOK_GENRE_WORDS[g], 2, replace=False)) + f" vol {i % 7}" for i, g in enumerate(genre)]
    years = [str(1980 + i % 25) for i in range(n_items)]
    years[3], years[5], years[7] = "DK Publishing Inc", "0", "2050"
    books = pd.DataFrame({"ISBN": isbn, "Book-Title": titles, "Book-Author": authors, "Year-Of-Publication": years,
                          "Publisher": [f"Pub{g}" for g in genre],
                          "Image-URL-S": "http://images.example/s.jpg", "Image-URL-M": "http://images.example/m.jpg",
                          "Image-URL-L": "http://images.example/l.jpg"})
    explicit = rng.random(len(df)) < 0.38
    score = np.clip(np.round(7 + 1.5 * df["aff"] + rng.normal(0, 1.2, len(df))), 1, 10).astype(int)
    ratings = pd.DataFrame({"User-ID": df["u"] + 1, "ISBN": [isbn[i] for i in df["i"]],
                            "Book-Rating": np.where(explicit, score, 0)})
    ratings.loc[ratings.sample(frac=0.03, random_state=1).index, "ISBN"] = "0000000000"   # not in Books.csv
    lower = ratings["ISBN"].str.endswith("X") & (rng.random(len(ratings)) < 0.5)
    ratings.loc[lower, "ISBN"] = ratings.loc[lower, "ISBN"].str.lower()                   # 'x' check digit
    ratings = pd.concat([ratings, ratings.head(20)])                                       # duplicates
    user_country = rng.choice(len(COUNTRIES), n_users)
    age = rng.integers(15, 75, n_users).astype(float)
    age[rng.random(n_users) < 0.35] = np.nan
    age[:3] = [0, 244, 5]
    users = pd.DataFrame({"User-ID": np.arange(1, n_users + 1),
                          "Location": [f"city{u % 50}, state, {COUNTRIES[c]}" for u, c in enumerate(user_country)],
                          "Age": age})
    books.to_csv(out / "Books.csv", index=False)
    ratings.to_csv(out / "Ratings.csv", index=False)
    users.to_csv(out / "Users.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="data/books_synthetic")
    a = ap.parse_args()
    print("wrote", write_books(Path(a.books)))
