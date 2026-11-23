"""
model_core — the custom AetherMind Transformer language model.

This package owns the model architecture itself (embeddings, attention,
transformer blocks, LM head) and is intentionally independent of training
infrastructure (see training/) and serving infrastructure (see
packages/inference). Keeping architecture code free of training-loop and
serving concerns makes it independently testable and benchmarkable.
"""

from packages.model_core.config import BASE, SMALL, TINY, ModelConfig
from packages.model_core.transformer import AetherMindLM

__all__ = [
    "AetherMindLM",
    "ModelConfig",
    "TINY",
    "SMALL",
    "BASE",
]
