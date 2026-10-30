"""FastAPI application for top-K recommendation."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from recsys.logging_utils import configure_logging, get_logger
from recsys.monitoring.prediction_log import PredictionLog
from recsys.serving.observability import (
    RECOMMEND_LATENCY,
    RECOMMENDATIONS,
    SCORE_HIST,
    ObservabilityMiddleware,
    access_logger,
    current_request_id,
    set_model_info,
)
from recsys.serving.recommender import get_recommender
from recsys.serving.schemas import (
    HealthResponse,
    ModelInfoResponse,
    RecommendedItem,
    RecommendRequest,
    RecommendResponse,
    ScoreRequest,
    ScoreResponse,
)

configure_logging()
logger = get_logger("recsys.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    recommender = get_recommender()
    app.state.recommender = recommender
    app.state.prediction_log = PredictionLog()
    set_model_info(recommender.model_name, recommender.version)
    logger.info("service_ready: %s", recommender.version)
    yield


app = FastAPI(
    title="MovieLens DeepFM Recommender",
    description="Top-K movie recommendation with a from-scratch PyTorch DeepFM model.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(ObservabilityMiddleware)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {"service": "movielens-deepfm-recommender", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    recommender = request.app.state.recommender
    return HealthResponse(
        status="healthy",
        model_loaded=recommender.model is not None,
        model_version=recommender.version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info(request: Request) -> ModelInfoResponse:
    recommender = request.app.state.recommender
    return ModelInfoResponse(
        model_name=recommender.model_name,
        model_version=recommender.version,
        users=int(len(recommender.users)),
        movies=int(len(recommender.movies)),
        metrics=recommender.metrics,
    )


@app.get("/metrics", tags=["meta"])
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend", response_model=RecommendResponse, tags=["recommendation"])
def recommend(payload: RecommendRequest, request: Request) -> RecommendResponse:
    recommender = request.app.state.recommender
    profile = payload.profile.model_dump(exclude_none=True) if payload.profile else None
    started = time.perf_counter()
    items, known, used_profile, cache_hit = recommender.recommend(
        payload.user_id,
        payload.top_k,
        exclude_seen=payload.exclude_seen,
        profile=profile,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    RECOMMEND_LATENCY.observe(latency_ms / 1000)
    RECOMMENDATIONS.inc(len(items))
    for item in items:
        SCORE_HIST.observe(float(item["score"]))
    access_logger.info(
        "recommend",
        extra={
            "request_id": current_request_id(),
            "user_id": payload.user_id,
            "cold_start": not known,
            "cache": "hit" if cache_hit else "miss",
            "n_items": len(items),
            "latency_ms": round(latency_ms, 2),
        },
    )
    try:
        request.app.state.prediction_log.log(
            model_version=recommender.version,
            user_id=payload.user_id,
            cold_start=not known,
            latency_ms=latency_ms,
            items=items,
            profile=used_profile,
        )
    except Exception as exc:  # logging must never break serving
        logger.warning("prediction_log_failed: %s", exc)

    return RecommendResponse(
        user_id=payload.user_id,
        cold_start=not known,
        model_version=recommender.version,
        items=[RecommendedItem(**item) for item in items],
        request_id=current_request_id(),
        cache_hit=cache_hit,
    )


@app.post("/score", response_model=ScoreResponse, tags=["recommendation"])
def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
    recommender = request.app.state.recommender
    profile = payload.profile.model_dump(exclude_none=True) if payload.profile else None
    started = time.perf_counter()
    scores, known = recommender.score(payload.user_id, payload.movie_ids, profile)
    latency_ms = (time.perf_counter() - started) * 1000
    access_logger.info(
        "score",
        extra={
            "request_id": current_request_id(),
            "user_id": payload.user_id,
            "cold_start": not known,
            "n_items": len(scores),
            "latency_ms": round(latency_ms, 2),
        },
    )
    return ScoreResponse(
        user_id=payload.user_id,
        cold_start=not known,
        model_version=recommender.version,
        scores=scores,
        request_id=current_request_id(),
    )
