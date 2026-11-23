"""
Model hyperparameter configuration for AetherMind's custom Transformer.

Kept deliberately small and explicit so every experiment is reproducible
and comparable. Values here are the single source of truth for model shape;
training scripts and inference code must construct ModelConfig rather than
hard-coding dimensions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Hyperparameters that fully define a decoder-only Transformer LM.

    Defaults target a very small model that can train on a single consumer
    GPU (or even CPU for smoke tests) while still exercising every component
    of the architecture. Scale these up once the pipeline is proven.
    """

    # Vocabulary / sequence
    vocab_size: int = 32000
    max_seq_len: int = 512

    # Model dimensions
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024  # typically 4 * d_model

    # Regularisation
    dropout: float = 0.1
    bias: bool = True  # include bias terms in Linear / LayerNorm

    # Special token ids (must match the tokenizer)
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Derived / convenience (not user-set)
    # d_head is computed as d_model // n_heads; asserted at construction.

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_ff <= 0 or self.d_model <= 0:
            raise ValueError("d_model and d_ff must be positive")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        # Only pass known fields; ignore extra keys from older checkpoints.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# Sensible presets for different compute budgets.
TINY = ModelConfig(  # ~2-3M params, pure smoke / unit tests
    vocab_size=2000,
    max_seq_len=128,
    d_model=128,
    n_layers=2,
    n_heads=4,
    d_ff=512,
    dropout=0.0,
)

SMALL = ModelConfig(  # ~15-20M params, laptop GPU friendly
    vocab_size=8000,
    max_seq_len=512,
    d_model=384,
    n_layers=6,
    n_heads=6,
    d_ff=1536,
)

BASE = ModelConfig(  # default production target once we have real data
    vocab_size=32000,
    max_seq_len=1024,
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=3072,
)
