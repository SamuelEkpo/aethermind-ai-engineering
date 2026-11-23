"""
API-level tests for the generation routes.

These tests do not require a trained checkpoint — when the engine is absent
the endpoints must return 503 instead of crashing.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_health_still_works() -> None:
    app = create_app()
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_generate_without_model_returns_503() -> None:
    app = create_app()
    with TestClient(app) as client:
        # Override whatever lifespan loaded
        client.app.state.engine = None
        r = client.post(
            "/v1/generate",
            json={"prompt": "hello", "max_new_tokens": 8},
        )
        assert r.status_code == 503


def test_model_info_without_model_returns_503() -> None:
    app = create_app()
    with TestClient(app) as client:
        client.app.state.engine = None
        r = client.get("/v1/model")
        assert r.status_code == 503
