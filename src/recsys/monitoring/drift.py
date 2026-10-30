"""Population Stability Index drift checks for the recommendation service."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROFILE_FIELDS: tuple[str, ...] = ("gender", "age", "occupation", "zip_prefix")


def _profile_values(field: str, values: pd.Series) -> pd.Series:
    """Monitor zip codes at region granularity to avoid high-cardinality noise."""
    text = values.astype(str)
    return text.str[:1] if field == "zip_prefix" else text


EPSILON = 1e-6
MAX_BINS = 10
MIN_ROWS = 100
RARE_CATEGORY_THRESHOLD = 0.02


def _bin_edges(values: np.ndarray) -> list[float]:
    quantiles = np.linspace(0, 1, MAX_BINS + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        edges = np.array([values.min(), values.max()])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(edge) for edge in edges]


def _bin_proportions(values: np.ndarray, edges: list[float]) -> np.ndarray:
    binned = pd.cut(values, bins=edges, include_lowest=True)
    counts = pd.Series(binned).value_counts(sort=False).to_numpy(dtype="float64")
    return counts / counts.sum() if counts.sum() else np.full(len(counts), 1 / len(counts))


def _per_user_max(user_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"user_id": np.asarray(user_ids), "score": np.asarray(scores)})
    return frame.groupby("user_id")["score"].max().to_numpy(dtype="float64")


def build_reference_stats(
    users: pd.DataFrame,
    scores: np.ndarray,
    model_name: str,
    version: str,
    *,
    user_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reference distribution for drift detection.

    The score reference is the per-user **Top-1** score, which is independent of
    the requested ``top_k`` and identical to the ``score_max`` value written to
    the prediction log.
    """
    profiles: dict[str, Any] = {}
    for field in PROFILE_FIELDS:
        values = _profile_values(field, users[field])
        frequencies = values.value_counts(normalize=True).head(50)
        profiles[field] = {str(key): round(float(value), 6) for key, value in frequencies.items()}

    score_values = (
        _per_user_max(user_ids, scores)
        if user_ids is not None
        else np.asarray(scores, dtype="float64")
    )
    edges = _bin_edges(score_values)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": version,
        "rows": int(len(users)),
        "profiles": profiles,
        "scores": {
            "edges": edges,
            "proportions": _bin_proportions(score_values, edges).round(6).tolist(),
            "mean": round(float(np.mean(score_values)), 6),
            "p95": round(float(np.quantile(score_values, 0.95)), 6),
        },
    }


def save_reference_stats(stats: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_reference_stats(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype="float64"), EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype="float64"), EPSILON, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _categorical_psi(reference: dict[str, float], values: pd.Series) -> float:
    if not reference:
        return 0.0
    actual = values.astype(str).value_counts(normalize=True)
    categories = set(reference) | set(actual.index)
    rare = {
        category
        for category in categories
        if reference.get(category, 0.0) < RARE_CATEGORY_THRESHOLD
    }
    expected_bucket: dict[str, float] = {}
    actual_bucket: dict[str, float] = {}
    for category, value in reference.items():
        key = "other" if category in rare else category
        expected_bucket[key] = expected_bucket.get(key, 0.0) + value
    for category, value in actual.items():
        key = "other" if category in rare else category
        actual_bucket[key] = actual_bucket.get(key, 0.0) + value
    keys = sorted(set(expected_bucket) | set(actual_bucket))
    expected = np.array([expected_bucket.get(key, 0.0) for key in keys])
    observed = np.array([actual_bucket.get(key, 0.0) for key in keys])
    return population_stability_index(expected, observed)


def compute_drift(
    reference: dict[str, Any],
    log_frame: pd.DataFrame,
    *,
    psi_stable: float = 0.1,
    psi_moderate: float = 0.25,
    min_rows: int = MIN_ROWS,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    if not log_frame.empty:
        for field in PROFILE_FIELDS:
            if field not in log_frame.columns:
                continue
            values = _profile_values(field, log_frame[field].dropna())
            if len(values) < min_rows:
                continue
            features[field] = {
                "type": "profile",
                "psi": round(_categorical_psi(reference["profiles"].get(field, {}), values), 4),
            }

        scores = log_frame["score_max"].dropna().to_numpy(dtype="float64")
        if len(scores) >= min_rows:
            actual = _bin_proportions(scores, reference["scores"]["edges"])
            features["score_top1"] = {
                "type": "score",
                "psi": round(
                    population_stability_index(reference["scores"]["proportions"], actual),
                    4,
                ),
                "current_mean": round(float(np.mean(scores)), 4),
                "reference_mean": reference["scores"]["mean"],
            }

    if not features:
        status = "insufficient_data"
    elif any(item["psi"] > psi_moderate for item in features.values()):
        status = "drift"
    elif any(item["psi"] > psi_stable for item in features.values()):
        status = "moderate"
    elif len(log_frame) < min_rows:
        status = "insufficient_data"
    else:
        status = "stable"

    drifted = [name for name, item in features.items() if item["psi"] > psi_moderate]
    return {
        "status": status,
        "rows": int(len(log_frame)),
        "n_checked": len(features),
        "drifted_features": drifted,
        "max_psi": max((item["psi"] for item in features.values()), default=0.0),
        "features": dict(sorted(features.items(), key=lambda item: item[1]["psi"], reverse=True)),
        "thresholds": {"stable": psi_stable, "moderate": psi_moderate},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def write_html_report(result: dict[str, Any], path: Path) -> Path:
    rows = []
    for name, details in result.get("features", {}).items():
        flag = ""
        if details["psi"] > result["thresholds"]["moderate"]:
            flag = "drift"
        elif details["psi"] > result["thresholds"]["stable"]:
            flag = "moderate"
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td><td>{details['type']}</td>"
            f"<td>{details['psi']:.4f}</td><td>{flag}</td></tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Recommendation Drift Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; max-width: 720px; width: 100%; }}
th, td {{ border-bottom: 1px solid #d8dee4; padding: 6px 10px; text-align: left; }}
th {{ background: #f6f8fa; }}
</style></head><body>
<h1>Recommendation Drift Report</h1>
<p>Status: <strong>{result["status"]}</strong> · rows: {result["rows"]} ·
checked: {result["n_checked"]} · max PSI: {result["max_psi"]:.4f}</p>
<table><thead><tr><th>Feature</th><th>Type</th><th>PSI</th><th>Flag</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p>PSI thresholds: &lt;0.1 stable, 0.1–0.25 moderate, &gt;0.25 drift.
Generated at {html.escape(result["checked_at"])}.</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
