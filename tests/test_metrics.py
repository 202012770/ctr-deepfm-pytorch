from __future__ import annotations

import numpy as np
import pytest

from recsys.train.metrics import binary_metrics, ranking_metrics


def test_ranking_metrics_known_case() -> None:
    user_ids = np.array([1, 1, 1, 2, 2, 2])
    labels = np.array([0, 0, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.9, 0.1, 0.2])
    metrics = ranking_metrics(user_ids, labels, scores, k_values=(2, 3))
    assert metrics["hr@2"] == pytest.approx(0.5)
    assert metrics["hr@3"] == pytest.approx(1.0)
    assert metrics["ndcg@2"] == pytest.approx(0.5)
    assert metrics["ndcg@3"] == pytest.approx(0.75)


def test_binary_metrics_perfect_separation() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = binary_metrics(labels, scores)
    assert metrics["auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["logloss"] < 0.3


def test_ranking_requires_positive() -> None:
    metrics = ranking_metrics(
        np.array([1, 1]), np.array([0, 0]), np.array([0.5, 0.4]), k_values=(10,)
    )
    assert metrics == {}
