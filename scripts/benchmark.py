"""Measure recommendation latency with a cold and a warm cache.

Usage:
    python scripts/benchmark.py --users 200 --top-k 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recsys.serving.recommender import Recommender

ROOT = Path(__file__).resolve().parents[1]


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype="float64")
    return {
        "mean_ms": round(float(array.mean()), 2),
        "p50_ms": round(float(np.percentile(array, 50)), 2),
        "p95_ms": round(float(np.percentile(array, 95)), 2),
        "p99_ms": round(float(np.percentile(array, 99)), 2),
        "max_ms": round(float(array.max()), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    recommender = Recommender()
    all_users = recommender.users["user_id"].tolist()
    rng = np.random.default_rng(42)
    sampled = rng.choice(all_users, size=min(args.users, len(all_users)), replace=False).tolist()

    for user_id in sampled[:5]:  # warm-up
        recommender.recommend(int(user_id), top_k=args.top_k)

    recommender._cache.clear()
    recommender.cache_hits = recommender.cache_misses = 0
    cold_latencies: list[float] = []
    for user_id in sampled:
        started = time.perf_counter()
        recommender.recommend(int(user_id), top_k=args.top_k)
        cold_latencies.append((time.perf_counter() - started) * 1000)

    warm_latencies: list[float] = []
    for user_id in sampled:
        started = time.perf_counter()
        recommender.recommend(int(user_id), top_k=args.top_k)
        warm_latencies.append((time.perf_counter() - started) * 1000)

    result = {
        "users": len(sampled),
        "top_k": args.top_k,
        "cold_cache": _percentiles(cold_latencies),
        "warm_cache": _percentiles(warm_latencies),
        "cache_hits": recommender.cache_hits,
        "cache_misses": recommender.cache_misses,
        "speedup_p95": round(
            _percentiles(cold_latencies)["p95_ms"]
            / max(_percentiles(warm_latencies)["p95_ms"], 1e-6),
            1,
        ),
    }
    path = ROOT / "reports" / "benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
