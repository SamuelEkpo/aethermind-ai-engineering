"""
Text generation endpoints.

The custom AetherMindLM is loaded once at application startup (see
apps.api.main lifespan) and stored on app.state.engine. Requests go through a
ModelRouter (see apps.api.core.routing) so an optional external model can be
selected explicitly or used as an opt-in fallback. Requests are synchronous for
now; we can move to a worker queue later without changing the public API shape.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from packages.model_router import NoBackendAvailableError, UnknownBackendError
from pydantic import BaseModel, Field

from apps.api.core.config import settings
from apps.api.core.routing import EXTERNAL_NAME, LOCAL_NAME, build_router

router = APIRouter(prefix="/v1", tags=["generate"])
log = structlog.get_logger(__name__)

_NOT_LOADED = "Model is not loaded. Train a checkpoint and set MODEL_CHECKPOINT_DIR."
_KNOWN_MODELS = (LOCAL_NAME, EXTERNAL_NAME)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    top_k: int | None = Field(40, ge=1, le=200)
    model: str | None = Field(
        None,
        max_length=64,
        description="Backend to use (see GET /v1/models). Defaults to the AetherMind model.",
    )


class GenerateResponse(BaseModel):
    prompt: str
    text: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str = LOCAL_NAME
    backend_kind: str = "custom"
    fallback_used: bool = False
    failed_backends: list[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    checkpoint: str
    tokenizer: str
    device: str
    parameters: int
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int


class ModelEntry(BaseModel):
    name: str
    kind: str
    model: str | None = None
    parameters: int | None = None


class ModelListResponse(BaseModel):
    models: list[ModelEntry]


def _engine(request: Request) -> Any:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail=_NOT_LOADED)
    return engine


@router.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    app = request.app
    model_router = build_router(app, settings)

    if body.model is None:
        # No explicit choice: only the local model is a default. An external model
        # alone is never picked implicitly (keeps private code on-box by default).
        _engine(request)
    elif body.model not in model_router.names:
        if body.model in _KNOWN_MODELS:
            raise HTTPException(status_code=503, detail=_NOT_LOADED)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{body.model}'. Available: {model_router.names}",
        )

    try:
        result = model_router.generate(
            body.prompt,
            model=body.model,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
        )
    except UnknownBackendError as exc:  # pragma: no cover - guarded above
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoBackendAvailableError as exc:
        # Upstream error text can contain internal URLs: log it, expose only names.
        log.warning("generation_failed", failures=exc.failures)
        raise HTTPException(
            status_code=502,
            detail=f"Generation failed on backend(s): {sorted(exc.failures)}",
        ) from exc
    return GenerateResponse(**result)


@router.get("/models", response_model=ModelListResponse)
def list_models(request: Request) -> ModelListResponse:
    entries = [
        ModelEntry(
            name=item["name"],
            kind=item["kind"],
            model=item.get("model"),
            parameters=item.get("parameters"),
        )
        for item in build_router(request.app, settings).describe()
    ]
    return ModelListResponse(models=entries)


@router.get("/model", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    engine = _engine(request)
    return ModelInfoResponse(**engine.info())
