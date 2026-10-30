"""Request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserProfile(BaseModel):
    """Optional attributes used when ``user_id`` is unknown (cold start)."""

    model_config = ConfigDict(extra="forbid")
    gender: str | None = Field(None, pattern="^[FM]$")
    age: str | None = Field(None, description="MovieLens age bucket, e.g. '25'")
    occupation: str | None = Field(None, description="MovieLens occupation code")
    zip_prefix: str | None = Field(None, min_length=1, max_length=3)


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: int = Field(..., ge=1)
    top_k: int = Field(10, ge=1, le=50)
    exclude_seen: bool = True
    profile: UserProfile | None = None


class RecommendedItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    cold_start: bool
    model_version: str
    items: list[RecommendedItem]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: int = Field(..., ge=1)
    movie_ids: list[int] = Field(..., min_length=1, max_length=1000)
    profile: UserProfile | None = None


class ScoreResponse(BaseModel):
    user_id: int
    cold_start: bool
    model_version: str
    scores: list[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    users: int
    movies: int
    metrics: dict[str, Any]
