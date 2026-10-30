"""Training loop with early stopping on validation AUC."""

from __future__ import annotations

import copy
import os
import sys
import time
from collections.abc import Iterator

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from recsys.logging_utils import get_logger
from recsys.models.deepfm import CTRModel
from recsys.train.metrics import binary_metrics

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: CTRModel,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 4096,
        epochs: int = 15,
        patience: int = 3,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        torch.set_num_threads(
            int(os.environ.get("CTR_NUM_THREADS", max(1, (os.cpu_count() or 4) // 2)))
        )
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.history: list[dict[str, float]] = []
        self.progress = sys.stdout.isatty()

    def _batches(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        indices = self.rng.permutation(len(X)) if shuffle else np.arange(len(X))
        for start in range(0, len(X), self.batch_size):
            batch = indices[start : start + self.batch_size]
            features = torch.from_numpy(np.asarray(X[batch], dtype=np.int64))
            labels = torch.from_numpy(np.asarray(y[batch], dtype=np.float32))
            yield features, labels

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> list[dict[str, float]]:
        best_auc = -1.0
        best_state = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        stale = 0
        batches_per_epoch = (len(X_train) + self.batch_size - 1) // self.batch_size

        for epoch in range(1, self.epochs + 1):
            started = time.perf_counter()
            self.model.train()
            total_loss = 0.0
            progress = tqdm(
                self._batches(X_train, y_train, shuffle=True),
                total=batches_per_epoch,
                desc=f"epoch {epoch}/{self.epochs}",
                leave=False,
                disable=not self.progress,
            )
            for features, labels in progress:
                features = features.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(features)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                if self.progress:
                    progress.set_postfix(loss=f"{loss.item():.4f}")

            scores = self.predict_proba(X_val)
            metrics = binary_metrics(y_val, scores, prefix="val_")
            train_loss = total_loss / len(X_train)
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                **metrics,
                "seconds": round(time.perf_counter() - started, 1),
            }
            self.history.append(record)
            logger.info(
                "epoch=%d train_loss=%.4f val_auc=%.4f (%.0fs)",
                epoch,
                train_loss,
                metrics["val_auc"],
                record["seconds"],
            )

            if metrics["val_auc"] > best_auc:
                best_auc = metrics["val_auc"]
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        self.model.load_state_dict(best_state)
        self.best_epoch = best_epoch
        self.best_val_auc = best_auc
        return self.history

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        outputs: list[np.ndarray] = []
        for features, _ in self._batches(X, np.zeros(len(X), dtype=np.float32), shuffle=False):
            logits = self.model(features.to(self.device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.array([])
