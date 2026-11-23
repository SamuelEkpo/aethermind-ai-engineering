"""
Token and positional embeddings for the AetherMind Transformer.

We use learned absolute positional embeddings (GPT-2 style) for the first
version. Rotary Position Embeddings (RoPE) can be introduced later without
changing the public interface of the model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from packages.model_core.config import ModelConfig


class TokenEmbedding(nn.Module):
    """Lookup table from token ids to dense vectors."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.d_model = config.d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: LongTensor of shape (batch, seq_len)
        Returns:
            FloatTensor of shape (batch, seq_len, d_model)
        """
        # Scale by sqrt(d_model) as in the original Transformer paper.
        # GPT-2 omits this; we keep it for better gradient scale at init.
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings, truncated to the actual length."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        """
        Returns:
            FloatTensor of shape (1, seq_len, d_model) ready for broadcasting.
        """
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions).unsqueeze(0)


class InputEmbedding(nn.Module):
    """Combines token + positional embeddings and applies dropout."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token = TokenEmbedding(config)
        self.position = PositionalEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)
        self.max_seq_len = config.max_seq_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: LongTensor (batch, seq_len)
        Returns:
            FloatTensor (batch, seq_len, d_model)
        """
        _, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max_seq_len {self.max_seq_len}"
            )
        x = self.token(token_ids) + self.position(seq_len, device=token_ids.device)
        return self.dropout(x)
