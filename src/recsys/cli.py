"""Command line interface: ``ctr <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from recsys.config import load_config
from recsys.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def cmd_prepare(args: argparse.Namespace) -> int:
    from recsys.pipeline import prepare_data

    prepared = prepare_data(force=args.force)
    print(f"train={prepared.X_train.shape} val={prepared.X_val.shape} test={prepared.X_test.shape}")
    print(f"fields={len(prepared.field_sizes)} field_sizes={prepared.field_sizes}")
    return 0


def _build_reference(bundle, *, n_users: int = 300, seed: int = 42):
    """Score a sample of users through the serving path and snapshot Top-1 scores."""
    import numpy as np

    from recsys.monitoring.drift import build_reference_stats
    from recsys.serving.recommender import Recommender

    recommender = Recommender(bundle)
    rng = np.random.default_rng(seed)
    user_ids = recommender.users["user_id"].to_numpy()
    sample = rng.choice(user_ids, size=min(n_users, len(user_ids)), replace=False)
    scores = []
    for user_id in sample:
        items, *_ = recommender.recommend(int(user_id), top_k=1)
        scores.append(float(items[0]["score"]))
    return build_reference_stats(
        recommender.users,
        np.asarray(scores, dtype="float64"),
        recommender.model_name,
        recommender.version,
    ), recommender


def cmd_train(args: argparse.Namespace) -> int:
    from recsys.pipeline import prepare_data, run_experiment, save_bundle

    cfg = load_config()
    prepared = prepare_data()
    model, trainer, metrics = run_experiment(
        args.model,
        prepared,
        cfg,
        embedding_dim=args.dim,
        epochs=args.epochs,
    )
    save_bundle(
        model,
        metrics,
        prepared,
        cfg,
        model_name=args.model,
        embedding_dim=args.dim or int(cfg.model.embedding_dim),
    )
    reference, recommender = _build_reference(cfg.path("paths.bundle_dir"))
    (cfg.path("paths.bundle_dir") / "reference_stats.json").write_text(
        json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Reference stats built for %s", recommender.version)
    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from recsys.experiments import run_comparison

    table = run_comparison(epochs=args.epochs)
    print(table.to_string(index=False))
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    from recsys.experiments import run_ablation

    table = run_ablation(epochs=args.epochs)
    print(table.to_string(index=False))
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    from recsys.experiments import run_robustness

    table = run_robustness(epochs=args.epochs)
    print(table[["seed", "test_auc", "test_hr@10", "test_ndcg@10"]].to_string(index=False))
    return 0


def cmd_negatives(args: argparse.Namespace) -> int:
    from recsys.experiments import run_negative_comparison

    table = run_negative_comparison(epochs=args.epochs)
    print(table[["setting", "test_auc", "test_hr@10", "test_ndcg@10"]].to_string(index=False))
    return 0


def cmd_coldstart(args: argparse.Namespace) -> int:
    from recsys.experiments import run_cold_start

    run_cold_start(epochs=args.epochs)
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    cfg = load_config()
    bundle = cfg.path("paths.bundle_dir")
    reference, recommender = _build_reference(bundle)
    path = bundle / "reference_stats.json"
    path.write_text(json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved reference stats to {path} (version={recommender.version})")
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    from recsys.monitoring.drift import (
        compute_drift,
        load_reference_stats,
        write_html_report,
    )
    from recsys.monitoring.prediction_log import PredictionLog

    cfg = load_config()
    bundle = cfg.path("paths.bundle_dir")
    reference_path = bundle / "reference_stats.json"
    if not reference_path.exists():
        print(f"No reference stats at {reference_path}. Run `ctr train` first.")
        return 1
    frame = PredictionLog().recent(limit=args.limit)
    if frame.empty:
        print("No predictions logged yet. Call /recommend first.")
        return 0
    result = compute_drift(load_reference_stats(reference_path), frame)
    reports = cfg.path("paths.reports_dir")
    (reports / "drift_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html_report(result, reports / "drift_report.html")
    print(json.dumps({k: result[k] for k in ("status", "rows", "max_psi")}, indent=2))
    return 1 if result["status"] == "drift" else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("recsys.serving.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ctr", description="PyTorch CTR recommendation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build encoded train/val/test arrays")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="train one model and save the serving bundle")
    train.add_argument("--model", default="deepfm", choices=["lr", "fm", "mlp", "deepfm"])
    train.add_argument("--dim", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.set_defaults(func=cmd_train)

    compare = sub.add_parser("compare", help="train and compare all four models")
    compare.add_argument("--epochs", type=int, default=None)
    compare.set_defaults(func=cmd_compare)

    ablation = sub.add_parser("ablation", help="embedding dimension and negative-ratio ablations")
    ablation.add_argument("--epochs", type=int, default=None)
    ablation.set_defaults(func=cmd_ablation)

    robustness = sub.add_parser("robustness", help="repeat runs over seeds with bootstrap CI")
    robustness.add_argument("--epochs", type=int, default=10)
    robustness.set_defaults(func=cmd_robustness)

    negatives = sub.add_parser("negatives", help="uniform vs popularity negative sampling")
    negatives.add_argument("--epochs", type=int, default=10)
    negatives.set_defaults(func=cmd_negatives)

    coldstart = sub.add_parser("coldstart", help="hold out users and evaluate cold start")
    coldstart.add_argument("--epochs", type=int, default=10)
    coldstart.set_defaults(func=cmd_coldstart)

    reference = sub.add_parser("reference", help="rebuild the drift reference from the bundle")
    reference.set_defaults(func=cmd_reference)

    drift = sub.add_parser("drift", help="compare logged requests with the training reference")
    drift.add_argument("--limit", type=int, default=1000)
    drift.set_defaults(func=cmd_drift)

    serve = sub.add_parser("serve", help="run the recommendation API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
