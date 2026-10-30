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
