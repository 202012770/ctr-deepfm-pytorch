"""Leave-one-out split with negative sampling.

Protocol:
- positive interaction = rating >= min_rating
- per user: last positive -> test, second last -> validation, the rest -> train
- training samples pair every positive with ``train_negatives`` unrated items
- validation/test candidates are 1 positive + ``eval_negatives`` unrated items
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from recsys.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    user_history: dict[int, set[int]] = field(repr=False)
    all_movies: np.ndarray = field(repr=False)


def _sample_negatives(
    pool: np.ndarray,
    count: int,
    rng: np.random.Generator,
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    if len(pool) == 0:
        raise ValueError("No candidate items available for negative sampling")
    replace = len(pool) < count
    return rng.choice(pool, size=count, replace=replace, p=probabilities)


def make_splits(
    ratings: pd.DataFrame,
    *,
    min_rating: int = 4,
    train_negatives: int = 2,
    eval_negatives: int = 99,
    seed: int = 42,
    negative_sampling: str = "uniform",
    popularity_alpha: float = 0.75,
) -> Split:
    """Negative sampling scheme applies to training only.

    Evaluation negatives stay uniform so metrics remain comparable across
    training schemes.
    """
    rng = np.random.default_rng(seed)
    positives = ratings.loc[ratings["rating"] >= min_rating].sort_values(
        ["user_id", "timestamp", "movie_id"]
    )
    history = ratings.groupby("user_id")["movie_id"].apply(set).to_dict()
    all_movies = np.sort(ratings["movie_id"].unique())

    popularity_weights: pd.Series | None = None
    if negative_sampling == "popularity":
        popularity = ratings.groupby("movie_id").size().astype("float64")
        popularity_weights = popularity.pow(popularity_alpha)
    elif negative_sampling != "uniform":
        raise ValueError(f"Unknown negative sampling: {negative_sampling}")

    train_rows: list[tuple[int, int, int]] = []
    val_rows: list[tuple[int, int, int]] = []
    test_rows: list[tuple[int, int, int]] = []

    for user_id, group in positives.groupby("user_id", sort=False):
        items = group["movie_id"].to_numpy()
        if len(items) < 3:
            train_items = items
            val_item = test_item = None
        else:
            train_items = items[:-2]
            val_item = int(items[-2])
            test_item = int(items[-1])

        seen = history.get(user_id, set())
        pool = np.fromiter(
            (movie for movie in all_movies if movie not in seen),
            dtype=np.int64,
            count=len(all_movies) - len(seen & set(all_movies)),
        )
        if len(train_items):
            train_rows.extend((int(user_id), int(item), 1) for item in train_items)
            probabilities = None
            if popularity_weights is not None:
                probabilities = (
                    popularity_weights.reindex(pool)
                    .fillna(0.0)
                    .to_numpy(dtype="float64")
                )
                probabilities = probabilities / probabilities.sum()
            negatives = _sample_negatives(
                pool, train_negatives * len(train_items), rng, probabilities
            )
            train_rows.extend((int(user_id), int(item), 0) for item in negatives)
        if val_item is not None:
            val_rows.append((int(user_id), val_item, 1))
            val_rows.extend(
                (int(user_id), int(item), 0)
                for item in _sample_negatives(pool, eval_negatives, rng)
            )
        if test_item is not None:
            test_rows.append((int(user_id), test_item, 1))
            test_rows.extend(
                (int(user_id), int(item), 0)
                for item in _sample_negatives(pool, eval_negatives, rng)
            )

    def _frame(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=["user_id", "movie_id", "label"])
        return frame.astype({"user_id": "int32", "movie_id": "int32", "label": "int8"})

    train = _frame(train_rows)
    val = _frame(val_rows)
    test = _frame(test_rows)
    logger.info(
        "Split sizes train=%d val=%d test=%d (positives=%.1f%%)",
        len(train),
        len(val),
        len(test),
        train["label"].mean() * 100,
    )
    return Split(
        train=train,
        val=val,
        test=test,
        user_history=history,
        all_movies=all_movies,
    )
