"""Load the trained bundle and serve top-K recommendations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recsys.config import load_config
from recsys.data.features import FeatureEncoder
from recsys.logging_utils import get_logger
from recsys.models.deepfm import build_model

logger = get_logger(__name__)


class Recommender:
    def __init__(self, bundle_dir: Path | None = None) -> None:
        cfg = load_config()
        self.bundle_dir = bundle_dir or cfg.path("paths.bundle_dir")
        if not (self.bundle_dir / "model.pt").exists():
            raise FileNotFoundError(
                f"No model bundle at {self.bundle_dir}. Run `ctr train` or `ctr compare` first."
            )
        payload = torch.load(self.bundle_dir / "model.pt", map_location="cpu", weights_only=False)
        self.model = build_model(
            payload["model_name"], payload["field_sizes"], **payload["model_config"]
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.model_name = payload["model_name"]
        self.encoder = FeatureEncoder.load(self.bundle_dir / "encoder.json")
        self.movies = pd.read_parquet(self.bundle_dir / "movies.parquet")
        self.users = pd.read_parquet(self.bundle_dir / "users.parquet")
        history = json.loads((self.bundle_dir / "user_history.json").read_text(encoding="utf-8"))
        self.history = {int(user): set(items) for user, items in history.items()}
        metrics_path = self.bundle_dir / "metrics.json"
        self.metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        )
        self.version = (
            str(self.metrics.get("model", self.model_name))
            + "-"
            + str(self.metrics.get("train_seconds", "unknown"))
        )
        self.device = "cpu"
        logger.info("Loaded %s bundle from %s", self.model_name, self.bundle_dir)

    def _user_table(
        self, user_id: int, profile: dict[str, Any] | None
    ) -> tuple[pd.DataFrame, bool]:
        known = bool((self.users["user_id"] == user_id).any())
        if known or not profile:
            return self.users, known
        row = {
            "user_id": user_id,
            "gender": profile.get("gender", "unknown"),
            "age": profile.get("age", "unknown"),
            "occupation": profile.get("occupation", "unknown"),
            "zip_prefix": profile.get("zip_prefix", "unknown"),
        }
        return pd.concat([self.users, pd.DataFrame([row])], ignore_index=True), False

    def _score_frame(self, frame: pd.DataFrame, users: pd.DataFrame) -> np.ndarray:
        X = self.encoder.encode(frame, users, self.movies)
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), 8192):
                batch = torch.from_numpy(X[start : start + 8192].astype(np.int64))
                scores.append(torch.sigmoid(self.model(batch)).numpy())
        return np.concatenate(scores) if scores else np.array([])

    def score(
        self, user_id: int, movie_ids: list[int], profile: dict[str, Any] | None = None
    ) -> tuple[list[float], bool]:
        users, known = self._user_table(user_id, profile)
        samples = pd.DataFrame({"user_id": user_id, "movie_id": movie_ids})
        scores = self._score_frame(samples, users)
        return [round(float(score), 6) for score in scores], known

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        *,
        exclude_seen: bool = True,
        profile: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        users, known = self._user_table(user_id, profile)
        candidates = self.movies
        if exclude_seen:
            seen = self.history.get(user_id, set())
            if seen:
                candidates = candidates[~candidates["movie_id"].isin(seen)]
        samples = pd.DataFrame({"user_id": user_id, "movie_id": candidates["movie_id"].to_numpy()})
        scores = self._score_frame(samples, users)
        order = np.argsort(scores)[::-1][:top_k]
        selected = candidates.iloc[order]
        items = [
            {
                "movie_id": int(row.movie_id),
                "title": str(row.title),
                "genres": str(row.genres),
                "score": round(float(score), 6),
            }
            for (_, row), score in zip(selected.iterrows(), scores[order], strict=True)
        ]
        return items, known


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender()
