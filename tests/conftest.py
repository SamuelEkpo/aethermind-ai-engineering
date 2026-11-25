"""Shared pytest fixtures for the AetherMind test suite."""

from __future__ import annotations

import pytest
from apps.api import main as api_main
from apps.api.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_api_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep API startup hermetic: never load a real checkpoint or external backend.

    Without this, any test that enters the app lifespan (``with TestClient(app)``)
    would import torch and load whatever checkpoint happens to exist on the
    developer's machine, making the suite slow and environment-dependent. Tests
    that need an engine or backend set ``app.state`` directly with fakes.
    """
    monkeypatch.setattr(api_main, "_try_load_engine", lambda: None)
    monkeypatch.setattr(api_main, "build_external_backend", lambda _settings: None)


@pytest.fixture()
def client() -> TestClient:
    """A TestClient against a freshly-built app instance (not the module-level singleton)."""
    app = create_app()
    return TestClient(app)
