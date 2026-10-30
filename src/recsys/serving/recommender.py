"""Load the trained bundle and serve top-K recommendations."""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
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
from recsys.serving.observability import CACHE_EVENTS

logger = get_logger(__name__)

MAX_CACHE_TOP_K = 50


class Recommender:
    def __init__(self, bundle_dir: Path | None = None) -> None:
        cfg = load_config()
        env_bundle = os.environ.get("CTR_BUNDLE_DIR")
        self.bundle_dir = bundle_dir or (
            Path(env_bundle) if env_bundle else cfg.path("paths.bundle_dir")
        )
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

        self.cache_size = int(os.environ.get("CTR_CACHE_SIZE", "4096"))
        self.cache_ttl = float(os.environ.get("CTR_CACHE_TTL", "300"))
        self._cache: OrderedDict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = (
            OrderedDict()
        )
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Loaded %s bundle from %s", self.model_name, self.bundle_dir)

    def _user_table(
        self, user_id: int, profile: dict[str, Any] | None
    ) -> tuple[pd.DataFrame, bool, dict[str, str]]:
        numeric = self.users["user_id"].to_numpy()
        known = bool(np.isin(user_id, numeric))
        if known:
            row = self.users.loc[self.users["user_id"] == user_id].iloc[0]
            used = {
                "gender": str(row["gender"]),
                "age": str(row["age"]),
                "occupation": str(row["occupation"]),
                "zip_prefix": str(row["zip_prefix"]),
            }
            return self.users, True, used

        used = {
            field: str((profile or {}).get(field, "unknown"))
            for field in ("gender", "age", "occupation", "zip_prefix")
        }
        row = {"user_id": user_id, **used}
        return pd.concat([self.users, pd.DataFrame([row])], ignore_index=True), False, used

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
        users, known, _ = self._user_table(user_id, profile)
        samples = pd.DataFrame({"user_id": user_id, "movie_id": movie_ids})
        scores = self._score_frame(samples, users)
        return [round(float(score), 6) for score in scores], known

    def _cache_key(self, user_id: int, exclude_seen: bool, used: dict[str, str]) -> tuple[Any, ...]:
        return (
            self.version,
            user_id,
            exclude_seen,
            tuple(sorted(used.items())),
        )

    def _cache_get(self, key: tuple[Any, ...]) -> list[dict[str, Any]] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, items = entry
        if expires_at < time.time():
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return items

    def _cache_set(self, key: tuple[Any, ...], items: list[dict[str, Any]]) -> None:
        if self.cache_size <= 0:
            return
        self._cache[key] = (time.time() + self.cache_ttl, items)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        *,
        exclude_seen: bool = True,
        profile: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], bool, dict[str, str], bool]:
        users, known, used = self._user_table(user_id, profile)
        key = self._cache_key(user_id, exclude_seen, used)

        cached = self._cache_get(key)
        if cached is not None and len(cached) >= min(top_k, MAX_CACHE_TOP_K):
            self.cache_hits += 1
            CACHE_EVENTS.labels("hit").inc()
            return cached[:top_k], known, used, True

        self.cache_misses += 1
        CACHE_EVENTS.labels("miss").inc()
        candidates = self.movies
        if exclude_seen:
            seen = self.history.get(user_id, set())
            if seen:
                candidates = candidates[~candidates["movie_id"].isin(seen)]
        samples = pd.DataFrame({"user_id": user_id, "movie_id": candidates["movie_id"].to_numpy()})
        scores = self._score_frame(samples, users)
        limit = min(max(top_k, MAX_CACHE_TOP_K), len(candidates))
        order = np.argsort(scores)[::-1][:limit]
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
        self._cache_set(key, items)
        return items[:top_k], known, used, False


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender()
