"""
High-level inference engine: load tokenizer + model once, generate on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from packages.model_core import AetherMindLM, ModelConfig
from packages.tokenizer import BPETokenizer


class InferenceEngine:
    """Load a checkpoint once and run generation requests."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        tokenizer_path: str | Path,
        device: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = BPETokenizer.load(self.tokenizer_path)
        self.model = AetherMindLM.from_pretrained(
            self.checkpoint_dir, map_location=str(self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    @property
    def config(self) -> ModelConfig:
        return self.model.config

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> dict[str, Any]:
        """Generate a completion and return text + metadata."""
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.vocab.get("<bos>", 2)]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        full_ids = output_ids[0].tolist()
        text = self.tokenizer.decode(full_ids)
        new_ids = full_ids[len(prompt_ids) :]

        return {
            "prompt": prompt,
            "text": text,
            "completion": self.tokenizer.decode(new_ids),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(new_ids),
            "total_tokens": len(full_ids),
        }

    def info(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_dir),
            "tokenizer": str(self.tokenizer_path),
            "device": str(self.device),
            "parameters": self.model.count_parameters(),
            "vocab_size": self.config.vocab_size,
            "max_seq_len": self.config.max_seq_len,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
        }
