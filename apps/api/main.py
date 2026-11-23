"""
AetherMind API — application entrypoint.

This module builds the FastAPI app via a factory function so tests can spin
up isolated instances, and so lifespan-managed resources (DB pools, model
loaders, etc.) have one clear place to live as the project grows.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps.api.core.config import settings
from apps.api.core.logging import configure_logging
from apps.api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # TODO(Phase 8): initialize DB engine, Redis pool, and model router here,
    # and tear them down on shutdown.
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.app_debug,
        lifespan=lifespan,
    )

    app.include_router(health_router)

    return app


app = create_app()
