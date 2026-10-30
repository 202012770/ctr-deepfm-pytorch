"""FastAPI application for top-K recommendation."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from recsys.logging_utils import configure_logging, get_logger
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
    app.state.recommender = get_recommender()
    yield


app = FastAPI(
    title="MovieLens DeepFM Recommender",
    description="Top-K movie recommendation with a from-scratch PyTorch DeepFM model.",
    version="1.0.0",
    lifespan=lifespan,
)


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


@app.post("/recommend", response_model=RecommendResponse, tags=["recommendation"])
def recommend(payload: RecommendRequest, request: Request) -> RecommendResponse:
    recommender = request.app.state.recommender
    profile = payload.profile.model_dump(exclude_none=True) if payload.profile else None
    items, known = recommender.recommend(
        payload.user_id,
        payload.top_k,
        exclude_seen=payload.exclude_seen,
        profile=profile,
    )
    return RecommendResponse(
        user_id=payload.user_id,
        cold_start=not known,
        model_version=recommender.version,
        items=[RecommendedItem(**item) for item in items],
    )


@app.post("/score", response_model=ScoreResponse, tags=["recommendation"])
def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
    recommender = request.app.state.recommender
    profile = payload.profile.model_dump(exclude_none=True) if payload.profile else None
    scores, known = recommender.score(payload.user_id, payload.movie_ids, profile)
    return ScoreResponse(
        user_id=payload.user_id,
        cold_start=not known,
        model_version=recommender.version,
        scores=scores,
    )
