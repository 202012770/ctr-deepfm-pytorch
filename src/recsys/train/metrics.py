"""Classification and ranking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def binary_metrics(labels: np.ndarray, scores: np.ndarray, prefix: str = "") -> dict[str, float]:
    labels = np.asarray(labels).astype("int32")
    scores = np.asarray(scores, dtype="float64")
    return {
        f"{prefix}auc": round(float(roc_auc_score(labels, scores)), 4),
        f"{prefix}logloss": round(float(log_loss(labels, scores)), 4),
        f"{prefix}accuracy": round(float(((scores >= 0.5) == labels).mean()), 4),
    }


def ranking_metrics(
    user_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    k_values: tuple[int, ...] = (10, 20),
    prefix: str = "",
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "user_id": np.asarray(user_ids),
            "label": np.asarray(labels).astype("int8"),
            "score": np.asarray(scores, dtype="float64"),
        }
    )
    frame = frame.sort_values(["user_id", "score"], ascending=[True, False])
    frame["rank"] = frame.groupby("user_id").cumcount() + 1
    positives = frame[frame["label"] == 1]
    if positives.empty:
        return {}

    metrics: dict[str, float] = {}
    for k in k_values:
        hits = positives["rank"] <= k
        metrics[f"{prefix}hr@{k}"] = round(float(hits.mean()), 4)
        gains = np.where(hits, 1.0 / np.log2(positives["rank"].clip(upper=k) + 1), 0.0)
        # single relevant item per user -> ideal DCG is 1
        metrics[f"{prefix}ndcg@{k}"] = round(float(gains.mean()), 4)
    return metrics
