"""Build a ModelRouter from the application's current state and settings."""

from __future__ import annotations

from fastapi import FastAPI
from packages.model_router import (
    BackendKind,
    LocalBackend,
    ModelRouter,
    OpenAICompatibleBackend,
    RoutingPolicy,
)

from apps.api.core.config import Settings

LOCAL_NAME = "aethermind"
EXTERNAL_NAME = "external"


def build_external_backend(settings: Settings) -> OpenAICompatibleBackend | None:
    """Create the optional external backend, or None when it is not configured."""
    if not settings.external_llm_base_url or not settings.external_llm_model:
        return None
    return OpenAICompatibleBackend(
        settings.external_llm_base_url,
        settings.external_llm_model,
        api_key=settings.external_llm_api_key,
        timeout=settings.external_llm_timeout,
    )


def build_router(app: FastAPI, settings: Settings) -> ModelRouter:
    """Compose a router from ``app.state.engine`` and ``app.state.external_backend``.

    The local engine stays the source of truth (so it can be swapped or cleared
    at runtime). Routing policy:
      * local model is the default whenever it is loaded;
      * the external model is only a fallback when ``external_fallback`` is on;
      * with no local model, nothing is routed by default (the API returns 503),
        but an external model can still be requested explicitly by name.
    """
    router = ModelRouter()
    engine = getattr(app.state, "engine", None)
    external = getattr(app.state, "external_backend", None)

    if engine is not None:
        router.register(LOCAL_NAME, LocalBackend(engine), kind=BackendKind.CUSTOM)
    if external is not None:
        router.register(EXTERNAL_NAME, external, kind=BackendKind.EXTERNAL)

    if engine is not None:
        fallbacks = (EXTERNAL_NAME,) if external is not None and settings.external_fallback else ()
        router.set_policy(RoutingPolicy(default=LOCAL_NAME, fallbacks=fallbacks))
    return router
