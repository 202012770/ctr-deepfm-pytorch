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


def test_bootstrap_ci_ordering() -> None:
    from recsys.experiments import bootstrap_ranking_ci

    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(50), 5)
    labels = (rng.random(len(users)) < 0.2).astype(int)
    scores = rng.random(len(users))
    result = bootstrap_ranking_ci(users, labels, scores, k=10, n_boot=100, seed=0)
    low, high = result["hr@10_ci95"]
    assert 0.0 <= low <= high <= 1.0
    low_n, high_n = result["ndcg@10_ci95"]
    assert 0.0 <= low_n <= high_n <= 1.0
