from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from recsys.data.features import FeatureEncoder
from recsys.data.loader import GENRE_COLUMNS
from recsys.models.deepfm import build_model


@pytest.fixture()
def toy_tables():
    users = pd.DataFrame(
        {
            "user_id": [1, 2],
            "gender": ["F", "M"],
            "age": ["25", "35"],
            "occupation": ["1", "2"],
            "zip_prefix": ["123", "456"],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": list(range(1, 13)),
            "title": [f"Movie {index} (199{index % 10})" for index in range(1, 13)],
            "genres": ["Drama" if index % 2 else "Comedy" for index in range(1, 13)],
            "year_bucket": ["1990", "2000"] * 6,
        }
    )
    for column in GENRE_COLUMNS:
        movies[column] = [
            int(column == "genre_drama" and index % 2)
            or int(column == "genre_comedy" and index % 2 == 0)
            for index in range(1, 13)
        ]
    ratings = pd.DataFrame(
        [
            (1, 1, 5, 1),
            (1, 2, 4, 2),
            (1, 3, 3, 3),
            (1, 4, 5, 4),
            (1, 5, 4, 5),
            (1, 6, 5, 6),
            (2, 2, 5, 1),
            (2, 3, 4, 2),
            (2, 7, 5, 3),
            (2, 8, 4, 4),
            (2, 9, 3, 5),
            (2, 10, 5, 6),
        ],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    return ratings, users, movies


@pytest.fixture()
def toy_encoder(toy_tables) -> FeatureEncoder:
    _, users, movies = toy_tables
    return FeatureEncoder().fit(users, movies)


@pytest.fixture()
def toy_bundle(tmp_path, toy_tables, toy_encoder):
    _, users, movies = toy_tables
    model = build_model(
        "deepfm", toy_encoder.field_sizes, embedding_dim=8, hidden_dims=[16], dropout=0.0
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "deepfm",
            "field_sizes": toy_encoder.field_sizes,
            "model_config": {"embedding_dim": 8, "hidden_dims": [16], "dropout": 0.0},
        },
        bundle / "model.pt",
    )
    toy_encoder.save(bundle / "encoder.json")
    movies.to_parquet(bundle / "movies.parquet", index=False)
    users.to_parquet(bundle / "users.parquet", index=False)
    (bundle / "user_history.json").write_text(
        json.dumps({"1": [1, 2, 3, 4, 5], "2": [2, 3, 7, 8, 9]}), encoding="utf-8"
    )
    (bundle / "metrics.json").write_text(
        json.dumps({"model": "deepfm", "test_auc": 0.9, "test_hr@10": 0.5}), encoding="utf-8"
    )
    return bundle
