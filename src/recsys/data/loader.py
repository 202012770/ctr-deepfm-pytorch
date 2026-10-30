"""Parse the MovieLens-1M files into clean tables."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from recsys.logging_utils import get_logger

logger = get_logger(__name__)

GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


def genre_column(genre: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")
    return f"genre_{slug}"


GENRE_COLUMNS: tuple[str, ...] = tuple(genre_column(genre) for genre in GENRES)
YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    frame["rating"] = frame["rating"].astype("int8")
    return frame


def load_users(raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    frame["zip_prefix"] = frame["zip_code"].astype(str).str[:3]
    frame["age"] = frame["age"].astype(str)
    frame["occupation"] = frame["occupation"].astype(str)
    return frame[["user_id", "gender", "age", "occupation", "zip_prefix"]]


def load_movies(raw_dir: Path) -> pd.DataFrame:
    rows: list[tuple[int, str, str]] = []
    with (raw_dir / "movies.dat").open("r", encoding="latin-1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("::", 2)
            if len(parts) == 3:
                rows.append((int(parts[0]), parts[1], parts[2]))
    frame = pd.DataFrame(rows, columns=["movie_id", "title", "genres"])

    extracted = frame["title"].str.extract(YEAR_PATTERN)[0]
    year = pd.to_numeric(extracted, errors="coerce")
    frame["year_bucket"] = (year // 10 * 10).astype("Int64").astype(str)
    frame.loc[year.isna(), "year_bucket"] = "unknown"

    for genre in GENRES:
        frame[genre_column(genre)] = (
            frame["genres"].str.split("|").apply(lambda items, g=genre: int(g in items))
        ).astype("int8")
    return frame[["movie_id", "title", "genres", "year_bucket", *GENRE_COLUMNS]]


def load_all(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = load_ratings(raw_dir)
    users = load_users(raw_dir)
    movies = load_movies(raw_dir)
    logger.info("Loaded ratings=%d users=%d movies=%d", len(ratings), len(users), len(movies))
    return ratings, users, movies
