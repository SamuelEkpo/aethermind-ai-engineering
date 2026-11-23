"""
Perplexity evaluation for AetherMindLM.

Perplexity = exp(average cross-entropy loss) over a held-out token stream.
Lower is better. Useful for comparing checkpoints on the same tokenizer/data.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from packages.model_core import AetherMindLM


class _TokenWindowDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        self.ids = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.ids[idx : idx + self.seq_len + 1]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


@torch.no_grad()
def compute_perplexity(
    model: AetherMindLM,
    token_ids: list[int] | Iterable[int],
    *,
    seq_len: int = 64,
    batch_size: int = 8,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute average loss and perplexity over token_ids.

    Returns:
        {"loss": float, "perplexity": float, "tokens_evaluated": int}
    """
    ids = list(token_ids)
    device = torch.device(device or next(model.parameters()).device)
    ds = _TokenWindowDataset(ids, seq_len=min(seq_len, model.config.max_seq_len))
    if len(ds) == 0:
        raise ValueError("token_ids too short for the requested seq_len")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels=labels)
        # loss is mean over tokens in the batch
        n_tok = labels.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    # clamp to avoid overflow on very high loss early in training
    ppl = math.exp(min(avg_loss, 20.0))
    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "tokens_evaluated": float(total_tokens),
    }
