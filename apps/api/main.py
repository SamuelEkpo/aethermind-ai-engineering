"""
AetherMind API — application entrypoint.

This module builds the FastAPI app via a factory function so tests can spin
up isolated instances, and so lifespan-managed resources (DB pools, model
loaders, etc.) have one clear place to live as the project grows.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from apps.api.core.config import settings
from apps.api.core.logging import configure_logging
from apps.api.routes.generate import router as generate_router
from apps.api.routes.health import router as health_router


def _try_load_engine():
    """Load InferenceEngine if a checkpoint exists; otherwise leave unloaded."""
    checkpoint = Path(settings.model_checkpoint_dir) / "lm_tiny"
    tokenizer = Path(settings.model_checkpoint_dir) / "tokenizer" / "tokenizer.json"

    # Also accept the explicit path used by train_lm defaults
    if not checkpoint.exists():
        checkpoint = Path("training/checkpoints/lm_tiny")
    if not tokenizer.exists():
        tokenizer = Path("training/checkpoints/tokenizer/tokenizer.json")

    if not checkpoint.exists() or not tokenizer.exists():
        return None

    from packages.inference import InferenceEngine

    return InferenceEngine(
        checkpoint_dir=checkpoint,
        tokenizer_path=tokenizer,
        device=settings.device,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.engine = _try_load_engine()
    # TODO(Phase 8): initialize DB engine, Redis pool here and tear down on shutdown.
    yield
    app.state.engine = None


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.app_debug,
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(generate_router)

    return app


app = create_app()
