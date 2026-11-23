"""Tests for packages.evaluation."""

from __future__ import annotations

import torch

from packages.evaluation import compute_perplexity
from packages.model_core import AetherMindLM, TINY


def test_compute_perplexity_returns_finite_metrics() -> None:
    torch.manual_seed(0)
    model = AetherMindLM(TINY)
    # Random token stream long enough for a few windows
    ids = torch.randint(0, TINY.vocab_size, (500,)).tolist()
    metrics = compute_perplexity(
        model, ids, seq_len=32, batch_size=4, max_batches=5
    )
    assert "loss" in metrics and "perplexity" in metrics
    assert metrics["loss"] > 0
    assert metrics["perplexity"] > 1.0
    assert metrics["tokens_evaluated"] > 0
