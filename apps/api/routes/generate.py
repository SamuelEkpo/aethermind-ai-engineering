"""
Text generation endpoint backed by AetherMindLM.

The model is loaded once at application startup (see apps.api.main lifespan)
and stored on app.state.engine. Requests are synchronous for now; we can move
to a worker queue later without changing the public API shape.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1", tags=["generate"])


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    top_k: int | None = Field(40, ge=1, le=200)


class GenerateResponse(BaseModel):
    prompt: str
    text: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


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


def _engine(request: Request) -> Any:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train a checkpoint and set MODEL_CHECKPOINT_DIR.",
        )
    return engine


@router.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    engine = _engine(request)
    result = engine.generate(
        body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_k=body.top_k,
    )
    return GenerateResponse(**result)


@router.get("/model", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    engine = _engine(request)
    return ModelInfoResponse(**engine.info())
