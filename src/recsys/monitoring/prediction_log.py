"""SQLite log of served recommendations, used as the drift input."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.config import PROJECT_ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    cold_start INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    n_items INTEGER NOT NULL,
    score_mean REAL NOT NULL,
    score_max REAL NOT NULL,
    top_items TEXT NOT NULL,
    profile TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recommendations_created_at ON recommendations(created_at);
"""


def default_db_path() -> Path:
    return Path(os.environ.get("CTR_PREDICTION_LOG", PROJECT_ROOT / "data" / "predictions.db"))


class PredictionLog:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def log(
        self,
        *,
        model_version: str,
        user_id: int,
        cold_start: bool,
        latency_ms: float,
        items: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> None:
        if not items:
            return
        scores = [float(item["score"]) for item in items]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recommendations (
                    created_at, model_version, user_id, cold_start, latency_ms,
                    n_items, score_mean, score_max, top_items, profile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    model_version,
                    int(user_id),
                    int(cold_start),
                    float(latency_ms),
                    len(items),
                    float(np.mean(scores)),
                    float(np.max(scores)),
                    json.dumps([item["movie_id"] for item in items]),
                    json.dumps(profile, ensure_ascii=False),
                ),
            )

    def recent(self, limit: int = 1000) -> pd.DataFrame:
        with self._connect() as connection:
            frame = pd.read_sql_query(
                "SELECT * FROM recommendations ORDER BY id DESC LIMIT ?",
                connection,
                params=(limit,),
            )
        if frame.empty:
            return frame
        profiles = frame["profile"].map(json.loads)
        for field in ("gender", "age", "occupation", "zip_prefix"):
            frame[field] = profiles.map(lambda payload, f=field: payload.get(f, "unknown"))
        return frame

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0])
