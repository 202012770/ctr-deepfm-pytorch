"""Prometheus metrics and request-scoped context."""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from recsys.logging_utils import get_logger

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")

REQUEST_COUNT = Counter("recsys_http_requests_total", "HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram(
    "recsys_http_request_duration_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
RECOMMENDATIONS = Counter("recsys_recommendations_total", "Recommended items served")
RECOMMEND_LATENCY = Histogram(
    "recsys_recommend_latency_seconds",
    "Recommendation computation latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
SCORE_HIST = Histogram(
    "recsys_recommended_score",
    "Distribution of scores of served recommendations",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
CACHE_EVENTS = Counter("recsys_cache_events_total", "Recommendation cache events", ["event"])
MODEL_INFO = Gauge("recsys_model_info", "Loaded model metadata", ["model", "version"])

access_logger = get_logger("recsys.access")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Adds a request id, structured access log and Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = REQUEST_ID.set(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000
            REQUEST_COUNT.labels(request.method, request.url.path, "500").inc()
            REQUEST_LATENCY.labels(request.url.path).observe(latency_ms / 1000)
            access_logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": 500,
                    "latency_ms": round(latency_ms, 2),
                },
            )
            raise
        finally:
            REQUEST_ID.reset(token)

        latency_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.url.path).observe(latency_ms / 1000)
        access_logger.info(
            "request",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response


def current_request_id() -> str:
    return REQUEST_ID.get()


def set_model_info(model: str, version: str) -> None:
    with contextlib.suppress(Exception):
        MODEL_INFO.clear()
    MODEL_INFO.labels(model=model, version=version).set(1)


def configure_access_logger(level: int = logging.INFO) -> None:
    access_logger.setLevel(level)
