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
    evaluate_scores,
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


def _rebuild_with_scheme(cfg: Config, negative_sampling: str) -> Prepared:
    ratings, users, movies = load_all(cfg.path("data.raw_dir"))
    split = make_splits(
        ratings,
        min_rating=int(cfg.data.min_rating),
        train_negatives=int(cfg.data.train_negatives),
        eval_negatives=int(cfg.data.eval_negatives),
        seed=int(cfg.data.seed),
        negative_sampling=negative_sampling,
    )
    from recsys.pipeline import enrich_user_movie_features

    users, movies = enrich_user_movie_features(split, ratings, users, movies)
    encoder = FeatureEncoder().fit(users, movies)
    return build_arrays(split, encoder, users, movies)


def bootstrap_ranking_ci(
    user_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    k: int = 10,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """User-level bootstrap CI for HR@K and NDCG@K."""
    frame = pd.DataFrame(
        {
            "user_id": np.asarray(user_ids),
            "label": np.asarray(labels).astype("int8"),
            "score": np.asarray(scores, dtype="float64"),
        }
    ).sort_values(["user_id", "score"], ascending=[True, False])
    frame["rank"] = frame.groupby("user_id").cumcount() + 1
    positives = frame[frame["label"] == 1]
    ranks = positives["rank"].to_numpy()
    hits = (ranks <= k).astype("float64")
    gains = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)

    rng = np.random.default_rng(seed)
    n_users = len(ranks)
    hr_samples = np.empty(n_boot)
    ndcg_samples = np.empty(n_boot)
    for index in range(n_boot):
        sample = rng.integers(0, n_users, n_users)
        hr_samples[index] = hits[sample].mean()
        ndcg_samples[index] = gains[sample].mean()
    return {
        f"hr@{k}_ci95": [
            float(np.quantile(hr_samples, 0.025)),
            float(np.quantile(hr_samples, 0.975)),
        ],
        f"ndcg@{k}_ci95": [
            float(np.quantile(ndcg_samples, 0.025)),
            float(np.quantile(ndcg_samples, 0.975)),
        ],
    }


def run_robustness(seeds: tuple[int, ...] = (42, 2024, 7), epochs: int = 10) -> pd.DataFrame:
    cfg = load_config()
    prepared = prepare_data(cfg)
    rows: list[dict[str, Any]] = []
    last_trainer = None
    for seed in seeds:
        _, trainer, metrics = run_experiment("deepfm", prepared, cfg, epochs=epochs, seed=seed)
        last_trainer = trainer
        metrics["seed"] = seed
        rows.append(metrics)

    table = pd.DataFrame(rows)
    summary = {
        "model": "deepfm",
        "seeds": list(seeds),
        "epochs": epochs,
        "test_auc_mean": round(float(table["test_auc"].mean()), 4),
        "test_auc_std": round(float(table["test_auc"].std(ddof=0)), 4),
        "test_hr@10_mean": round(float(table["test_hr@10"].mean()), 4),
        "test_hr@10_std": round(float(table["test_hr@10"].std(ddof=0)), 4),
        "test_ndcg@10_mean": round(float(table["test_ndcg@10"].mean()), 4),
        "test_ndcg@10_std": round(float(table["test_ndcg@10"].std(ddof=0)), 4),
    }
    if last_trainer is not None:
        scores = last_trainer.predict_proba(prepared.X_test)
        summary.update(bootstrap_ranking_ci(prepared.test_users, prepared.y_test, scores))

    reports = cfg.path("paths.reports_dir")
    table.to_csv(reports / "robustness_runs.csv", index=False)
    (reports / "robustness.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return table


def run_negative_comparison(epochs: int = 10) -> pd.DataFrame:
    cfg = load_config()
    rows: list[dict[str, Any]] = []
    for scheme in ("uniform", "popularity"):
        prepared = _rebuild_with_scheme(cfg, scheme)
        _, _, metrics = run_experiment("deepfm", prepared, cfg, epochs=epochs)
        metrics["experiment"] = "negative_sampling"
        metrics["setting"] = scheme
        rows.append(metrics)
    table = pd.DataFrame(rows)
    path = cfg.path("paths.reports_dir") / "negative_sampling.csv"
    table.to_csv(path, index=False)
    return table


def run_cold_start(epochs: int = 10, fraction: float = 0.1) -> dict[str, Any]:
    from recsys.pipeline import prepare_cold_start

    cfg = load_config()
    data = prepare_cold_start(cfg, fraction=fraction)
    _, trainer, _ = run_experiment("deepfm", data.warm, cfg, epochs=epochs)

    warm_scores = trainer.predict_proba(data.warm.X_test)
    cold_scores = trainer.predict_proba(data.X_cold)
    warm = evaluate_scores(data.warm, data.warm.y_test, data.warm.test_users, warm_scores, "warm")
    cold = evaluate_scores(data.warm, data.y_cold, data.cold_users, cold_scores, "cold")
    result = {
        "cold_user_fraction": fraction,
        "cold_users": int(len(np.unique(data.cold_users))),
        "warm": warm,
        "cold": cold,
        "delta": {
            warm_key: round(cold[warm_key.replace("warm", "cold")] - warm[warm_key], 4)
            for warm_key in ("warm_auc", "warm_hr@10", "warm_ndcg@10")
            if warm_key in warm and warm_key.replace("warm", "cold") in cold
        },
    }
    reports = cfg.path("paths.reports_dir")
    (reports / "cold_start.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
