"""
Transformer blocks and the full AetherMind language model.

Architecture choices (deliberate, documented):
- Pre-norm (LayerNorm before attention / FFN) — more stable training from scratch
  than the original post-norm Transformer.
- GELU activation in the feed-forward network (standard for modern LMs).
- Weight tying between token embeddings and the final LM head.
- Residual connections around both sub-layers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.model_core.attention import CausalSelfAttention
from packages.model_core.config import ModelConfig
from packages.model_core.embeddings import InputEmbedding


class FeedForward(nn.Module):
    """Position-wise feed-forward network (two-layer MLP)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single pre-norm Transformer decoder block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm + residual
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln_2(x))
        return x


class AetherMindLM(nn.Module):
    """Decoder-only Transformer language model (AetherMind's custom model).

    Public interface:
        forward(input_ids, attention_mask=None, labels=None) -> logits or (logits, loss)
        generate(...) -> token ids
        count_parameters() -> int
        save_pretrained / from_pretrained
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = InputEmbedding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (standard for small/medium LMs)
        self.lm_head.weight = self.embed.token.embedding.weight

        self.apply(self._init_weights)

        # Special scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) long
            attention_mask: optional (batch, seq_len) — 1 = real token, 0 = pad
            labels: optional (batch, seq_len) — if provided, cross-entropy loss
                    is computed (shifted language-modeling loss)

        Returns:
            logits (batch, seq_len, vocab_size)  or  (logits, loss)
        """
        x = self.embed(input_ids)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits

        # Shifted LM loss: predict token t+1 from tokens <= t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.config.pad_token_id,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation (sampling).

        Args:
            input_ids: (batch, prompt_len)
            max_new_tokens: how many tokens to append
            temperature: >0; 1.0 = no change, lower = more greedy
            top_k: if set, sample only from the top-k logits
            eos_token_id: stop early when this id is produced (per sequence)

        Returns:
            (batch, prompt_len + generated_len) long tensor
        """
        self.eval()
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        generated = input_ids

        for _ in range(max_new_tokens):
            # Crop to max context if needed
            context = generated[:, -self.config.max_seq_len :]
            logits = self(context)  # (B, T, V)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> ModelConfig:
        return self.config

    def save_pretrained(self, path: str | Path) -> None:
        """Save weights + config to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        (path / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def from_pretrained(
        cls, path: str | Path, map_location: str | None = None
    ) -> AetherMindLM:
        path = Path(path)
        config = ModelConfig.from_dict(
            json.loads((path / "config.json").read_text(encoding="utf-8"))
        )
        model = cls(config)
        state = torch.load(
            path / "pytorch_model.bin", map_location=map_location or "cpu", weights_only=True
        )
        model.load_state_dict(state)
        return model
