from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data.features import FIELD_NAMES, FeatureEncoder
from recsys.data.split import make_splits


def test_encoder_shapes_and_unknown(toy_tables) -> None:
    ratings, users, movies = toy_tables
    encoder = FeatureEncoder().fit(users, movies)
    samples = pd.DataFrame({"user_id": [1, 999], "movie_id": [2, 3]})
    matrix = encoder.encode(samples, users, movies)
    assert matrix.shape == (2, len(FIELD_NAMES))
    assert matrix[0, 0] > 0
    assert matrix[1, 0] == 0  # unseen user -> reserved index
    assert len(encoder.field_sizes) == len(FIELD_NAMES)
    assert all(size >= 2 for size in encoder.field_sizes)


def test_encoder_roundtrip(tmp_path, toy_encoder) -> None:
    path = tmp_path / "encoder.json"
    toy_encoder.save(path)
    loaded = FeatureEncoder.load(path)
    assert loaded.field_names == toy_encoder.field_names
    assert loaded.vocab == toy_encoder.vocab


def test_split_protocol(toy_tables) -> None:
    ratings, _, _ = toy_tables
    split = make_splits(ratings, min_rating=4, train_negatives=1, eval_negatives=5, seed=0)
    for frame in (split.val, split.test):
        positives = frame[frame["label"] == 1]
        assert positives.groupby("user_id").size().eq(1).all()
        assert len(frame) == 2 * (1 + 5)
        negatives = frame[frame["label"] == 0]
        for row in negatives.itertuples():
            assert row.movie_id not in split.user_history[row.user_id]
    assert set(split.train["label"].unique()) == {0, 1}
    assert split.train["label"].mean() == pytest.approx(0.5)


def test_negative_count_matches_ratio(toy_tables) -> None:
    ratings, _, _ = toy_tables
    split = make_splits(ratings, min_rating=4, train_negatives=2, eval_negatives=3, seed=1)
    positives = int((split.train["label"] == 1).sum())
    negatives = int((split.train["label"] == 0).sum())
    assert negatives == positives * 2
    assert np.all(split.all_movies > 0)


def test_popularity_negative_sampling(toy_tables) -> None:
    ratings, _, _ = toy_tables
    split = make_splits(
        ratings,
        min_rating=4,
        train_negatives=1,
        eval_negatives=3,
        seed=3,
        negative_sampling="popularity",
    )
    negatives = split.train[split.train["label"] == 0]
    assert len(negatives) > 0
    for row in negatives.itertuples():
        assert row.movie_id not in split.user_history[row.user_id]
    positives = int((split.train["label"] == 1).sum())
    assert int((split.train["label"] == 0).sum()) == positives


def test_enrich_features_adds_buckets(toy_tables) -> None:
    from recsys.pipeline import enrich_user_movie_features

    ratings, users, movies = toy_tables
    split = make_splits(ratings, min_rating=4, train_negatives=1, eval_negatives=2, seed=0)
    users_enriched, movies_enriched = enrich_user_movie_features(split, ratings, users, movies)
    for column in ("user_activity", "user_avg_rating", "user_top_genre"):
        assert column in users_enriched.columns
        assert users_enriched[column].notna().all()
    for column in ("item_popularity", "item_avg_rating"):
        assert column in movies_enriched.columns
        assert movies_enriched[column].notna().all()
