"""
Minimal, reproducible training entrypoint for AetherMindLM.

Intentionally small and explicit so every experiment is easy to understand
and re-run. Not yet a full production trainer (no DDP, no fancy schedulers).
Those come once the core loop is proven on TINY / SMALL.

Usage (after `pip install -e ".[ml]"`):

    python -m training.scripts.train_lm \
        --config tiny \
        --steps 200 \
        --batch-size 8 \
        --seq-len 64 \
        --lr 3e-4 \
        --output training/checkpoints/lm_tiny
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from packages.model_core import AetherMindLM, BASE, ModelConfig, SMALL, TINY
from packages.tokenizer import BPETokenizer

CONFIGS = {
    "tiny": TINY,
    "small": SMALL,
    "base": BASE,
}


class TextDataset(Dataset):
    """Simple sliding-window dataset over a tokenized corpus."""

    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        self.ids = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


def load_token_ids(tokenizer_path: Path, corpus_dir: Path) -> list[int]:
    tokenizer = BPETokenizer.load(tokenizer_path)
    texts = [
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted(corpus_dir.rglob("*.txt"))
    ]
    if not texts:
        raise SystemExit(f"No .txt files found under {corpus_dir}")
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="tiny")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("training/checkpoints/tokenizer/tokenizer.json"),
    )
    parser.add_argument("--corpus-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("training/checkpoints/lm_tiny")
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    base_config: ModelConfig = CONFIGS[args.config]

    # Align vocab size with the actual tokenizer if present
    if args.tokenizer.exists():
        tok = BPETokenizer.load(args.tokenizer)
        config = ModelConfig(
            vocab_size=tok.vocab_size,
            max_seq_len=base_config.max_seq_len,
            d_model=base_config.d_model,
            n_layers=base_config.n_layers,
            n_heads=base_config.n_heads,
            d_ff=base_config.d_ff,
            dropout=base_config.dropout,
            bias=base_config.bias,
        )
    else:
        config = base_config

    print(f"Config: {args.config} | d_model={config.d_model} n_layers={config.n_layers}")
    device = torch.device(args.device)

    token_ids = load_token_ids(args.tokenizer, args.corpus_dir)
    print(f"Corpus tokens: {len(token_ids):,}")

    seq_len = min(args.seq_len, config.max_seq_len)
    dataset = TextDataset(token_ids, seq_len=seq_len)
    if len(dataset) == 0:
        raise SystemExit("Corpus too short for the requested seq_len")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    model = AetherMindLM(config).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(loader)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels=labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        if step % 20 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:4d}/{args.steps}  loss={loss.item():.4f}  ({elapsed:.1f}s)")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    print(f"Saved checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
