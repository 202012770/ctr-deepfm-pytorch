"""Data preparation and single-experiment orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recsys.config import Config, load_config
from recsys.data.features import FeatureEncoder
from recsys.data.loader import load_all
from recsys.data.split import Split, make_splits
from recsys.logging_utils import get_logger
from recsys.models.deepfm import build_model
from recsys.train.metrics import binary_metrics, ranking_metrics
from recsys.train.trainer import Trainer

logger = get_logger(__name__)


@dataclass
class Prepared:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    val_users: np.ndarray
    test_users: np.ndarray
    encoder: FeatureEncoder
    users: pd.DataFrame
    movies: pd.DataFrame
    user_history: dict[int, set[int]]

    @property
    def field_sizes(self) -> list[int]:
        return self.encoder.field_sizes


def build_arrays(
    split: Split, encoder: FeatureEncoder, users: pd.DataFrame, movies: pd.DataFrame
) -> Prepared:
    def encode(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            encoder.encode(frame, users, movies),
            frame["label"].to_numpy(dtype="int8"),
            frame["user_id"].to_numpy(dtype="int32"),
        )

    X_train, y_train, _ = encode(split.train)
    X_val, y_val, val_users = encode(split.val)
    X_test, y_test, test_users = encode(split.test)
    return Prepared(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        val_users=val_users,
        test_users=test_users,
        encoder=encoder,
        users=users,
        movies=movies,
        user_history=split.user_history,
    )


def prepare_data(cfg: Config | None = None, *, force: bool = False) -> Prepared:
    cfg = cfg or load_config()
    processed = cfg.path("paths.processed_dir")
    required = [
        processed / "train_X.npy",
        processed / "val_X.npy",
        processed / "test_X.npy",
        processed / "encoder.json",
    ]
    if all(path.exists() for path in required) and not force:
        logger.info("Loading prepared arrays from %s", processed)
        return load_prepared(cfg)

    raw_dir = cfg.path("data.raw_dir")
    ratings, users, movies = load_all(raw_dir)
    split = make_splits(
        ratings,
        min_rating=int(cfg.data.min_rating),
        train_negatives=int(cfg.data.train_negatives),
        eval_negatives=int(cfg.data.eval_negatives),
        seed=int(cfg.data.seed),
    )
    encoder = FeatureEncoder().fit(users, movies)
    prepared = build_arrays(split, encoder, users, movies)

    processed.mkdir(parents=True, exist_ok=True)
    np.save(processed / "train_X.npy", prepared.X_train)
    np.save(processed / "train_y.npy", prepared.y_train)
    np.save(processed / "val_X.npy", prepared.X_val)
    np.save(processed / "val_y.npy", prepared.y_val)
    np.save(processed / "val_users.npy", prepared.val_users)
    np.save(processed / "test_X.npy", prepared.X_test)
    np.save(processed / "test_y.npy", prepared.y_test)
    np.save(processed / "test_users.npy", prepared.test_users)
    encoder.save(processed / "encoder.json")
    movies.to_parquet(processed / "movies.parquet", index=False)
    users.to_parquet(processed / "users.parquet", index=False)
    (processed / "user_history.json").write_text(
        json.dumps({str(user): sorted(items) for user, items in split.user_history.items()}),
        encoding="utf-8",
    )
    stats = {
        "train_samples": int(len(prepared.y_train)),
        "val_samples": int(len(prepared.y_val)),
        "test_samples": int(len(prepared.y_test)),
        "train_positive_rate": round(float(prepared.y_train.mean()), 4),
        "users": int(len(users)),
        "movies": int(len(movies)),
        "field_sizes": encoder.field_sizes,
        "fields": encoder.field_names,
    }
    (processed / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Prepared train=%d val=%d test=%d",
        len(prepared.y_train),
        len(prepared.y_val),
        len(prepared.y_test),
    )
    return prepared


def load_prepared(cfg: Config | None = None) -> Prepared:
    cfg = cfg or load_config()
    processed = cfg.path("paths.processed_dir")
    encoder = FeatureEncoder.load(processed / "encoder.json")
    history = json.loads((processed / "user_history.json").read_text(encoding="utf-8"))
    return Prepared(
        X_train=np.load(processed / "train_X.npy"),
        y_train=np.load(processed / "train_y.npy"),
        X_val=np.load(processed / "val_X.npy"),
        y_val=np.load(processed / "val_y.npy"),
        X_test=np.load(processed / "test_X.npy"),
        y_test=np.load(processed / "test_y.npy"),
        val_users=np.load(processed / "val_users.npy"),
        test_users=np.load(processed / "test_users.npy"),
        encoder=encoder,
        users=pd.read_parquet(processed / "users.parquet"),
        movies=pd.read_parquet(processed / "movies.parquet"),
        user_history={int(user): set(items) for user, items in history.items()},
    )


def evaluate_scores(
    prepared: Prepared, labels: np.ndarray, users: np.ndarray, scores: np.ndarray, prefix: str
) -> dict[str, float]:
    k_values = tuple(load_config().evaluation.k_values)
    metrics = binary_metrics(labels, scores, prefix=f"{prefix}_")
    metrics.update(ranking_metrics(users, labels, scores, k_values=k_values, prefix=f"{prefix}_"))
    return metrics


def run_experiment(
    model_name: str,
    prepared: Prepared,
    cfg: Config | None = None,
    *,
    embedding_dim: int | None = None,
    hidden_dims: list[int] | None = None,
    dropout: float | None = None,
    epochs: int | None = None,
    verbose: bool = True,
) -> tuple[torch.nn.Module, Trainer, dict[str, Any]]:
    cfg = cfg or load_config()
    model = build_model(
        model_name,
        prepared.field_sizes,
        embedding_dim=embedding_dim or int(cfg.model.embedding_dim),
        hidden_dims=hidden_dims or list(cfg.model.hidden_dims),
        dropout=cfg.model.dropout if dropout is None else dropout,
    )
    trainer = Trainer(
        model,
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        batch_size=int(cfg.training.batch_size),
        epochs=epochs or int(cfg.training.epochs),
        patience=int(cfg.training.patience),
        seed=int(cfg.training.seed),
    )
    started = time.perf_counter()
    trainer.fit(prepared.X_train, prepared.y_train, prepared.X_val, prepared.y_val)
    seconds = time.perf_counter() - started

    val_scores = trainer.predict_proba(prepared.X_val)
    test_scores = trainer.predict_proba(prepared.X_test)
    metrics: dict[str, Any] = {
        "model": model_name,
        "embedding_dim": int(embedding_dim or cfg.model.embedding_dim),
        "hidden_dims": list(hidden_dims or cfg.model.hidden_dims),
        "parameters": model.count_parameters(),
        "best_epoch": trainer.best_epoch,
        "train_seconds": round(seconds, 1),
    }
    metrics.update(evaluate_scores(prepared, prepared.y_val, prepared.val_users, val_scores, "val"))
    metrics.update(
        evaluate_scores(prepared, prepared.y_test, prepared.test_users, test_scores, "test")
    )
    if verbose:
        logger.info(
            "%s: val_auc=%.4f test_auc=%.4f test_hr@10=%.4f",
            model_name,
            metrics["val_auc"],
            metrics["test_auc"],
            metrics.get("test_hr@10", float("nan")),
        )
    return model, trainer, metrics


def save_bundle(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    prepared: Prepared,
    cfg: Config | None = None,
    model_name: str = "deepfm",
    embedding_dim: int = 16,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.2,
) -> Path:
    cfg = cfg or load_config()
    bundle = cfg.path("paths.bundle_dir")
    bundle.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "field_sizes": prepared.field_sizes,
            "model_config": {
                "embedding_dim": embedding_dim,
                "hidden_dims": hidden_dims or list(cfg.model.hidden_dims),
                "dropout": dropout,
            },
        },
        bundle / "model.pt",
    )
    prepared.encoder.save(bundle / "encoder.json")
    prepared.movies.to_parquet(bundle / "movies.parquet", index=False)
    prepared.users.to_parquet(bundle / "users.parquet", index=False)
    (bundle / "user_history.json").write_text(
        json.dumps({str(user): sorted(items) for user, items in prepared.user_history.items()}),
        encoding="utf-8",
    )
    (bundle / "metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return bundle


def load_bundle(cfg: Config | None = None) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    cfg = cfg or load_config()
    bundle = cfg.path("paths.bundle_dir")
    payload = torch.load(bundle / "model.pt", map_location="cpu", weights_only=False)
    class_name = payload["model_name"]
    model = build_model(class_name, payload["field_sizes"], **payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, bundle
