"""
Causal multi-head self-attention for AetherMind.

Standard scaled-dot-product attention with an explicit causal (lower-triangular)
mask so that position i can only attend to positions <= i. Implemented in pure
PyTorch first for clarity and correctness; we can swap in flash-attention or
SDPA later without changing callers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.model_core.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Single projection for Q, K, V then split — more efficient than 3 linears.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer — registered so it moves with .to(device)
        # Shape: (1, 1, max_seq_len, max_seq_len)
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: optional (batch, seq_len) boolean / 0-1 mask where
                True / 1 means "keep" (not padding). Currently only used to
                zero out padding positions in the attention scores.

        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.d_model, dim=2)

        # (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        # Causal mask (prevent attending to future tokens)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # Optional padding mask
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            att = att.masked_fill(~key_mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_heads, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads

        return self.resid_dropout(self.proj(y))
