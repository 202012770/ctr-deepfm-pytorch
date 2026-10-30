from __future__ import annotations


def test_recommender_returns_sorted_top_k(toy_bundle) -> None:
    from recsys.serving.recommender import Recommender

    recommender = Recommender(toy_bundle)
    items, known, profile, cache_hit = recommender.recommend(1, top_k=3)
    assert known is True
    assert profile["gender"] in {"F", "M"}
    assert cache_hit is False
    assert len(items) == 3
    scores = [item["score"] for item in items]
    assert scores == sorted(scores, reverse=True)
    seen = {1, 2, 3, 4, 5}
    assert all(item["movie_id"] not in seen for item in items)

    _, _, _, second_hit = recommender.recommend(1, top_k=3)
    assert second_hit is True


def test_recommender_cold_start_profile(toy_bundle) -> None:
    from recsys.serving.recommender import Recommender

    recommender = Recommender(toy_bundle)
    items, known, profile, _ = recommender.recommend(
        999, top_k=2, profile={"gender": "F", "age": "25", "occupation": "1", "zip_prefix": "123"}
    )
    assert known is False
    assert profile["gender"] == "F"
    assert len(items) == 2


def test_api_endpoints(toy_bundle, monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    import recsys.serving.api as api_module
    from recsys.serving.recommender import Recommender

    recommender = Recommender(toy_bundle)
    monkeypatch.setattr(api_module, "get_recommender", lambda: recommender)
    monkeypatch.setenv("CTR_PREDICTION_LOG", str(tmp_path / "predictions.db"))
    with TestClient(api_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        info = client.get("/model/info")
        assert info.status_code == 200
        assert info.json()["model_name"] == "deepfm"

        response = client.post("/recommend", json={"user_id": 1, "top_k": 3})
        assert response.status_code == 200
        body = response.json()
        assert body["cold_start"] is False
        assert body["request_id"]
        assert len(body["items"]) == 3
        assert body["cache_hit"] is False

        again = client.post("/recommend", json={"user_id": 1, "top_k": 3})
        assert again.json()["cache_hit"] is True

        scored = client.post("/score", json={"user_id": 1, "movie_ids": [2, 3]})
        assert scored.status_code == 200
        assert len(scored.json()["scores"]) == 2

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "recsys_http_requests_total" in metrics.text
        assert "recsys_cache_events_total" in metrics.text

    assert (tmp_path / "predictions.db").exists()


def test_api_validation(toy_bundle, monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    import recsys.serving.api as api_module
    from recsys.serving.recommender import Recommender

    monkeypatch.setattr(api_module, "get_recommender", lambda: Recommender(toy_bundle))
    monkeypatch.setenv("CTR_PREDICTION_LOG", str(tmp_path / "predictions.db"))
    with TestClient(api_module.app) as client:
        assert client.post("/recommend", json={"user_id": 1, "top_k": 0}).status_code == 422
        assert client.post("/recommend", json={"user_id": 1, "extra": 1}).status_code == 422
        assert client.post("/score", json={"user_id": 1, "movie_ids": []}).status_code == 422
