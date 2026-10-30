"""Model comparison and ablation studies."""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recsys.config import Config, load_config  # noqa: E402
from recsys.data.features import FeatureEncoder  # noqa: E402
from recsys.data.loader import load_all  # noqa: E402
from recsys.data.split import make_splits  # noqa: E402
from recsys.logging_utils import get_logger  # noqa: E402
from recsys.pipeline import (  # noqa: E402
    Prepared,
    build_arrays,
    load_prepared,
    prepare_data,
    run_experiment,
    save_bundle,
)

logger = get_logger(__name__)
MODELS = ("lr", "fm", "mlp", "deepfm")


def _plot_comparison(table: pd.DataFrame, path) -> None:
    metrics = ["test_auc", "test_hr@10", "test_ndcg@10"]
    labels = [metric.replace("test_", "") for metric in metrics]
    positions = np.arange(len(table))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for index, (metric, label) in enumerate(zip(metrics, labels, strict=True)):
        values = table[metric].to_numpy()
        offset = (index - (len(metrics) - 1) / 2) * width
        bars = ax.bar(positions + offset, values, width, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(table["model"].str.upper())
    ax.set_ylim(0, 1.0)
    ax.set_title("Model comparison on the holdout split")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_comparison(epochs: int | None = None) -> pd.DataFrame:
    cfg = load_config()
    prepared = prepare_data(cfg)
    rows: list[dict[str, Any]] = []
    trained: dict[str, tuple[Any, Any, dict[str, Any]]] = {}

    for name in MODELS:
        model, trainer, metrics = run_experiment(name, prepared, cfg, epochs=epochs)
        trained[name] = (model, trainer, metrics)
        rows.append(metrics)

    table = pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)
    comparison_path = cfg.path("paths.reports_dir") / "model_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(comparison_path, index=False)
    _plot_comparison(table, cfg.path("paths.figures_dir") / "model_comparison.png")

    best_name = str(table.iloc[0]["model"])
    model, trainer, best_metrics = trained[best_name]
    save_bundle(
        model,
        best_metrics,
        prepared,
        cfg,
        model_name=best_name,
        embedding_dim=int(best_metrics["embedding_dim"]),
        hidden_dims=list(best_metrics["hidden_dims"]),
    )
    (cfg.path("paths.reports_dir") / "metrics.json").write_text(
        json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Best model: %s (val_auc=%.4f)", best_name, best_metrics["val_auc"])
    return table


def _rebuild(prepared: Prepared, cfg: Config, negatives: int) -> Prepared:
    ratings, users, movies = load_all(cfg.path("data.raw_dir"))
    split = make_splits(
        ratings,
        min_rating=int(cfg.data.min_rating),
        train_negatives=negatives,
        eval_negatives=int(cfg.data.eval_negatives),
        seed=int(cfg.data.seed),
    )
    encoder = FeatureEncoder().fit(users, movies)
    return build_arrays(split, encoder, users, movies)


def run_ablation(epochs: int | None = None) -> pd.DataFrame:
    cfg = load_config()
    prepared = load_prepared(cfg)
    rows: list[dict[str, Any]] = []

    baseline: dict[str, Any] | None = None
    for dim in (8, 16, 32):
        _, _, metrics = run_experiment("deepfm", prepared, cfg, embedding_dim=dim, epochs=epochs)
        metrics["experiment"] = "embedding_dim"
        metrics["setting"] = f"dim={dim}"
        rows.append(metrics)
        if dim == 16:
            baseline = dict(metrics)

    for negatives in (1, 4):
        rebuilt = _rebuild(prepared, cfg, negatives)
        _, _, metrics = run_experiment("deepfm", rebuilt, cfg, epochs=epochs)
        metrics["experiment"] = "negative_ratio"
        metrics["setting"] = f"1:{negatives}"
        rows.append(metrics)

    if baseline is not None:
        baseline["experiment"] = "negative_ratio"
        baseline["setting"] = "1:2"
        rows.append(baseline)

    table = pd.DataFrame(rows)
    path = cfg.path("paths.reports_dir") / "ablation.csv"
    table.to_csv(path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, experiment, xlabel in (
        (axes[0], "embedding_dim", "Embedding dimension"),
        (axes[1], "negative_ratio", "Negative ratio"),
    ):
        subset = table[table["experiment"] == experiment]
        ax.plot(subset["setting"], subset["test_auc"], marker="o", label="test AUC")
        ax.plot(subset["setting"], subset["test_hr@10"], marker="s", label="HR@10")
        ax.set_title(experiment.replace("_", " ").title())
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    figure_path = cfg.path("paths.figures_dir") / "ablation.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return table
