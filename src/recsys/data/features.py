"""Categorical feature encoding shared by training and serving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.data.loader import GENRE_COLUMNS

USER_FIELDS: tuple[str, ...] = (
    "user_id",
    "gender",
    "age",
    "occupation",
    "zip_prefix",
    "user_activity",
    "user_avg_rating",
    "user_top_genre",
)
MOVIE_FIELDS: tuple[str, ...] = (
    "movie_id",
    "year_bucket",
    "item_popularity",
    "item_avg_rating",
)
FIELD_NAMES: tuple[str, ...] = USER_FIELDS + MOVIE_FIELDS + GENRE_COLUMNS


def _with_missing_fields(
    table: pd.DataFrame, fields: tuple[str, ...], value: str = "unknown"
) -> pd.DataFrame:
    frame = table.copy()
    for field in fields:
        if field not in frame.columns:
            frame[field] = value
    return frame


class FeatureEncoder:
    """Vocabulary per field; index 0 is reserved for unseen values (cold start)."""

    def __init__(self) -> None:
        self.field_names: list[str] = list(FIELD_NAMES)
        self.vocab: dict[str, dict[str, int]] = {}

    def fit(self, users: pd.DataFrame, movies: pd.DataFrame) -> FeatureEncoder:
        self.vocab = {}
        users = _with_missing_fields(users, USER_FIELDS)
        movies = _with_missing_fields(movies, MOVIE_FIELDS)
        for field in USER_FIELDS:
            values = sorted(users[field].astype(str).unique())
            self.vocab[field] = {value: index + 1 for index, value in enumerate(values)}
        for field in MOVIE_FIELDS:
            values = sorted(movies[field].astype(str).unique())
            self.vocab[field] = {value: index + 1 for index, value in enumerate(values)}
        for column in GENRE_COLUMNS:
            self.vocab[column] = {"0": 1, "1": 2}
        return self

    @property
    def field_sizes(self) -> list[int]:
        return [len(self.vocab[field]) + 1 for field in self.field_names]

    def encode(
        self,
        samples: pd.DataFrame,
        users: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> np.ndarray:
        users = _with_missing_fields(users, USER_FIELDS)
        movies = _with_missing_fields(movies, MOVIE_FIELDS)
        merged = (
            samples[["user_id", "movie_id"]]
            .merge(users, on="user_id", how="left")
            .merge(movies.drop(columns=["title", "genres"]), on="movie_id", how="left")
        )

        matrix = np.zeros((len(merged), len(self.field_names)), dtype=np.int32)
        for position, field in enumerate(self.field_names):
            values = merged[field].astype(str)
            matrix[:, position] = values.map(self.vocab[field]).fillna(0).astype(np.int32)
        return matrix

    def encode_user(self, features: dict[str, Any]) -> np.ndarray:
        return np.array(
            [self.vocab[field].get(str(features.get(field, "")), 0) for field in USER_FIELDS],
            dtype=np.int32,
        )

    def encode_movie(self, features: dict[str, Any]) -> np.ndarray:
        return np.array(
            [
                self.vocab[field].get(str(features.get(field, "")), 0)
                for field in MOVIE_FIELDS + GENRE_COLUMNS
            ],
            dtype=np.int32,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"field_names": self.field_names, "vocab": self.vocab},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> FeatureEncoder:
        payload = json.loads(path.read_text(encoding="utf-8"))
        encoder = cls()
        encoder.field_names = payload["field_names"]
        encoder.vocab = payload["vocab"]
        return encoder
