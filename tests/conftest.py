"""Shared pytest fixtures for the AetherMind test suite."""

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> TestClient:
    """A TestClient against a freshly-built app instance (not the module-level singleton)."""
    app = create_app()
    return TestClient(app)
