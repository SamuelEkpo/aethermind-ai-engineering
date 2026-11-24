"""API tests for model routing on /v1/generate and /v1/models (no torch, no network)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from apps.api import main as api_main
from apps.api.core.config import Settings, settings
from apps.api.core.routing import build_external_backend
from fastapi.testclient import TestClient
from packages.model_router import OpenAICompatibleBackend


class FakeEngine:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def generate(self, prompt: str, **_: Any) -> dict[str, Any]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("engine exploded")
        return {
            "prompt": prompt,
            "text": prompt + " local",
            "completion": " local",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }

    def info(self) -> dict[str, Any]:
        return {"parameters": 123, "checkpoint": "x", "tokenizer": "y"}


class ExternalProbe:
    """Records how many requests reached the mocked external endpoint."""

    def __init__(self, *, status: int = 200) -> None:
        self.status = status
        self.calls = 0

    def backend(self) -> OpenAICompatibleBackend:
        def handler(request: httpx.Request) -> httpx.Response:
            self.calls += 1
            if self.status != 200:
                return httpx.Response(self.status, json={})
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": " external"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
            )

        return OpenAICompatibleBackend(
            "http://secret-internal-host:9000/v1",
            "ext-model",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Skip checkpoint loading and any real external config during lifespan.
    monkeypatch.setattr(api_main, "_try_load_engine", lambda: None)
    monkeypatch.setattr(api_main, "build_external_backend", lambda _settings: None)
    monkeypatch.setattr(settings, "external_fallback", False)
    with TestClient(api_main.create_app()) as test_client:
        yield test_client


def _body(**extra: Any) -> dict[str, Any]:
    return {"prompt": "hello", "max_new_tokens": 8, **extra}


def test_default_generate_uses_local_model_and_reports_provenance(client: TestClient) -> None:
    client.app.state.engine = FakeEngine()
    r = client.post("/v1/generate", json=_body())
    assert r.status_code == 200
    data = r.json()
    assert data["completion"] == " local"
    assert data["model"] == "aethermind"
    assert data["backend_kind"] == "custom"
    assert data["fallback_used"] is False


def test_no_local_model_returns_503_even_if_external_is_configured(client: TestClient) -> None:
    probe = ExternalProbe()
    client.app.state.external_backend = probe.backend()
    r = client.post("/v1/generate", json=_body())
    assert r.status_code == 503
    assert probe.calls == 0  # never routed implicitly


def test_explicit_external_model_is_routed_and_tagged(client: TestClient) -> None:
    client.app.state.engine = FakeEngine()
    probe = ExternalProbe()
    client.app.state.external_backend = probe.backend()
    r = client.post("/v1/generate", json=_body(model="external"))
    assert r.status_code == 200
    data = r.json()
    assert data["completion"] == " external"
    assert data["model"] == "external"
    assert data["backend_kind"] == "external"
    assert probe.calls == 1


def test_explicit_external_works_without_local_model(client: TestClient) -> None:
    client.app.state.external_backend = ExternalProbe().backend()
    r = client.post("/v1/generate", json=_body(model="external"))
    assert r.status_code == 200
    assert r.json()["backend_kind"] == "external"


def test_explicit_external_when_not_configured_returns_503(client: TestClient) -> None:
    client.app.state.engine = FakeEngine()
    r = client.post("/v1/generate", json=_body(model="external"))
    assert r.status_code == 503


def test_explicit_local_when_not_loaded_returns_503(client: TestClient) -> None:
    r = client.post("/v1/generate", json=_body(model="aethermind"))
    assert r.status_code == 503


def test_unknown_model_returns_400(client: TestClient) -> None:
    client.app.state.engine = FakeEngine()
    r = client.post("/v1/generate", json=_body(model="gpt-nonexistent"))
    assert r.status_code == 400
    assert "gpt-nonexistent" in r.json()["detail"]


def test_local_failure_does_not_reach_external_by_default(client: TestClient) -> None:
    client.app.state.engine = FakeEngine(fail=True)
    probe = ExternalProbe()
    client.app.state.external_backend = probe.backend()
    r = client.post("/v1/generate", json=_body())
    assert r.status_code == 502
    assert probe.calls == 0


def test_fallback_to_external_when_enabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "external_fallback", True)
    client.app.state.engine = FakeEngine(fail=True)
    client.app.state.external_backend = ExternalProbe().backend()
    r = client.post("/v1/generate", json=_body())
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "external"
    assert data["fallback_used"] is True
    assert data["failed_backends"] == ["aethermind"]


def test_502_detail_does_not_leak_upstream_urls(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "external_fallback", True)
    client.app.state.engine = FakeEngine(fail=True)
    client.app.state.external_backend = ExternalProbe(status=500).backend()
    r = client.post("/v1/generate", json=_body())
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "secret-internal-host" not in detail
    assert "aethermind" in detail
    assert "external" in detail


def test_models_endpoint_lists_backends_without_internal_details(client: TestClient) -> None:
    client.app.state.engine = FakeEngine()
    client.app.state.external_backend = ExternalProbe().backend()
    r = client.get("/v1/models")
    assert r.status_code == 200
    models = {m["name"]: m for m in r.json()["models"]}
    assert models["aethermind"]["kind"] == "custom"
    assert models["aethermind"]["parameters"] == 123
    assert models["external"]["kind"] == "external"
    assert models["external"]["model"] == "ext-model"
    assert "secret-internal-host" not in r.text


def test_models_endpoint_empty_when_nothing_loaded(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json() == {"models": []}


def test_build_external_backend_requires_url_and_model() -> None:
    assert build_external_backend(Settings(_env_file=None)) is None
    assert (
        build_external_backend(Settings(_env_file=None, external_llm_base_url="http://x/v1"))
        is None
    )
    backend = build_external_backend(
        Settings(
            _env_file=None,
            external_llm_base_url="http://x/v1",
            external_llm_model="m",
            external_llm_api_key="k",
        )
    )
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.model == "m"
    backend.close()


def test_lifespan_closes_external_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class Closeable:
        closed = False

        def close(self) -> None:
            self.closed = True

    fake = Closeable()
    monkeypatch.setattr(api_main, "_try_load_engine", lambda: None)
    monkeypatch.setattr(api_main, "build_external_backend", lambda _settings: fake)
    with TestClient(api_main.create_app()):
        assert fake.closed is False
    assert fake.closed is True
