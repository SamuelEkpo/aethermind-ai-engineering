"""
Tests for InferenceEngine and the generation API happy path.

A tiny but real tokenizer and model checkpoint are built once per module,
so these tests exercise the true load/generate code paths without needing
a trained checkpoint or any network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from apps.api.main import create_app
from fastapi.testclient import TestClient
from packages.inference.engine import InferenceEngine
from packages.model_core import AetherMindLM, ModelConfig
from packages.tokenizer import BPETokenizer

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "The quick brown fox jumps over the lazy dog.\n",
] * 10


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build a tiny tokenizer + untrained model checkpoint on disk."""
    root = tmp_path_factory.mktemp("engine_artifacts")
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=300)
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    checkpoint_dir = root / "checkpoint"
    AetherMindLM(config).save_pretrained(checkpoint_dir)
    return checkpoint_dir, tokenizer_path


@pytest.fixture(scope="module")
def engine(artifacts: tuple[Path, Path]) -> InferenceEngine:
    checkpoint_dir, tokenizer_path = artifacts
    return InferenceEngine(checkpoint_dir, tokenizer_path, device="cpu")


def test_info_reports_model_shape(engine: InferenceEngine) -> None:
    info = engine.info()
    assert info["device"] == "cpu"
    assert info["d_model"] == 32
    assert info["n_layers"] == 1
    assert info["n_heads"] == 2
    assert info["max_seq_len"] == 32
    assert info["vocab_size"] == engine.tokenizer.vocab_size
    assert info["parameters"] > 0


def test_generate_returns_consistent_token_counts(engine: InferenceEngine) -> None:
    result = engine.generate("def add", max_new_tokens=6)
    assert result["prompt"] == "def add"
    assert result["prompt_tokens"] == len(engine.tokenizer.encode("def add"))
    assert 1 <= result["completion_tokens"] <= 6
    assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
    assert isinstance(result["text"], str)
    assert isinstance(result["completion"], str)


def test_generate_empty_prompt_falls_back_to_bos(engine: InferenceEngine) -> None:
    result = engine.generate("", max_new_tokens=4)
    assert result["prompt_tokens"] == 1
    assert 1 <= result["completion_tokens"] <= 4


def test_generate_is_reproducible_with_same_seed(engine: InferenceEngine) -> None:
    torch.manual_seed(123)
    first = engine.generate("def sub", max_new_tokens=8)
    torch.manual_seed(123)
    second = engine.generate("def sub", max_new_tokens=8)
    assert first == second


def test_missing_checkpoint_raises(artifacts: tuple[Path, Path], tmp_path: Path) -> None:
    _, tokenizer_path = artifacts
    with pytest.raises(FileNotFoundError):
        InferenceEngine(tmp_path / "does_not_exist", tokenizer_path, device="cpu")


def test_missing_tokenizer_raises(artifacts: tuple[Path, Path], tmp_path: Path) -> None:
    checkpoint_dir, _ = artifacts
    with pytest.raises(FileNotFoundError):
        InferenceEngine(checkpoint_dir, tmp_path / "missing_tokenizer.json", device="cpu")


def test_api_generate_happy_path(engine: InferenceEngine) -> None:
    app = create_app()
    with TestClient(app) as client:
        client.app.state.engine = engine
        response = client.post(
            "/v1/generate",
            json={"prompt": "def add", "max_new_tokens": 5, "temperature": 0.7, "top_k": 10},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prompt"] == "def add"
        assert body["total_tokens"] == body["prompt_tokens"] + body["completion_tokens"]
        assert 1 <= body["completion_tokens"] <= 5


def test_api_model_info_happy_path(engine: InferenceEngine) -> None:
    app = create_app()
    with TestClient(app) as client:
        client.app.state.engine = engine
        response = client.get("/v1/model")
        assert response.status_code == 200
        body = response.json()
        assert body["d_model"] == 32
        assert body["device"] == "cpu"
        assert body["parameters"] > 0


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "hi", "max_new_tokens": 0},
        {"prompt": "hi", "max_new_tokens": 100000},
        {"prompt": "hi", "temperature": 0},
        {"prompt": "hi", "top_k": 0},
    ],
)
def test_api_generate_rejects_invalid_input(payload: dict[str, object]) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/v1/generate", json=payload)
        assert response.status_code == 422
