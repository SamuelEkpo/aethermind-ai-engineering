"""
model_core — the custom AetherMind Transformer language model.

This package owns the model architecture itself (embeddings, attention,
transformer blocks, LM head) and is intentionally independent of training
infrastructure (see training/) and serving infrastructure (see
packages/inference). Keeping architecture code free of training-loop and
serving concerns makes it independently testable and benchmarkable, which
is the whole point of separating "our model" from "the system around it"
per the README's hybrid architecture principle.

Planned contents (see roadmap Phase 3-4):
    - embeddings.py      token + positional embeddings
    - attention.py        causal self-attention
    - transformer.py      transformer block / full model assembly
    - config.py            model hyperparameter dataclasses
"""

__all__: list[str] = []
