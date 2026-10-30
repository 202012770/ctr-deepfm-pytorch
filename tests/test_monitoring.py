from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.monitoring.drift import (
    build_reference_stats,
    compute_drift,
    write_html_report,
)
from recsys.monitoring.prediction_log import PredictionLog


def _users(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "gender": rng.choice(["F", "M"], n),
            "age": rng.choice(["18", "25", "35", "45"], n),
            "occupation": rng.choice(["1", "2", "3"], n),
            "zip_prefix": rng.choice(["100", "200", "300"], n),
        }
    )


def test_prediction_log_roundtrip(tmp_path) -> None:
    log = PredictionLog(tmp_path / "predictions.db")
    log.log(
        model_version="deepfm-1",
        user_id=1,
        cold_start=False,
        latency_ms=5.5,
        items=[{"movie_id": 2, "score": 0.9}, {"movie_id": 3, "score": 0.8}],
        profile={"gender": "F", "age": "25", "occupation": "1", "zip_prefix": "123"},
    )
    assert log.count() == 1
    frame = log.recent(limit=10)
    assert frame.iloc[0]["gender"] == "F"
    assert frame.iloc[0]["score_max"] == 0.9
    assert frame.iloc[0]["n_items"] == 2


def test_drift_stable_on_same_distribution(tmp_path) -> None:
    users = _users()
    rng = np.random.default_rng(1)
    scores = rng.normal(0.7, 0.1, len(users))
    reference = build_reference_stats(users, scores, "deepfm", "v1")

    sample = users.sample(len(users), replace=True, random_state=2)
    log_frame = pd.DataFrame(
        {
            "gender": sample["gender"].to_numpy(),
            "age": sample["age"].to_numpy(),
            "occupation": sample["occupation"].to_numpy(),
            "zip_prefix": sample["zip_prefix"].to_numpy(),
            "score_max": rng.normal(0.7, 0.1, len(users)),
        }
    )
    result = compute_drift(reference, log_frame)
    assert result["status"] in {"stable", "moderate"}
    assert result["n_checked"] >= 4
    path = write_html_report(result, tmp_path / "drift.html")
    assert path.exists()


def test_drift_detects_shift(tmp_path) -> None:
    users = _users()
    rng = np.random.default_rng(3)
    reference = build_reference_stats(users, rng.normal(0.7, 0.1, len(users)), "deepfm", "v1")
    shifted = pd.DataFrame(
        {
            "gender": ["M"] * len(users),
            "age": ["35"] * len(users),
            "occupation": ["3"] * len(users),
            "zip_prefix": ["900"] * len(users),
            "score_max": [0.2] * len(users),
        }
    )
    result = compute_drift(reference, shifted)
    assert result["status"] == "drift"
    assert "gender" in result["drifted_features"]
    assert "score_top1" in result["drifted_features"]


def test_drift_requires_rows() -> None:
    users = _users()
    reference = build_reference_stats(users, np.full(len(users), 0.7), "deepfm", "v1")
    result = compute_drift(reference, pd.DataFrame())
    assert result["status"] == "insufficient_data"
