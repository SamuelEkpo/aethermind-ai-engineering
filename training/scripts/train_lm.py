"""
Minimal, reproducible training entrypoint for AetherMindLM.

Intentionally small and explicit so every experiment is easy to understand
and re-run. Not yet a full production trainer (no DDP, no fancy schedulers).
Those come once the core loop is proven on TINY / SMALL.

Usage (after `pip install -e ".[ml]"`):

    python -m training.scripts.train_lm \
        --config tiny \
        --steps 500 \
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


@torch.no_grad()
def evaluate(model: AetherMindLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels=labels)
        total_loss += loss.item()
        n += 1
        if n >= 20:  # cap eval cost on small runs
            break
    model.train()
    return total_loss / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="tiny")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=50)
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

    # Chronological split — last val_ratio held out
    split = int(len(token_ids) * (1.0 - args.val_ratio))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

    seq_len = min(args.seq_len, config.max_seq_len)
    train_ds = TextDataset(train_ids, seq_len=seq_len)
    val_ds = TextDataset(val_ids, seq_len=seq_len)
    if len(train_ds) == 0:
        raise SystemExit("Training split too short for the requested seq_len")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    model = AetherMindLM(config).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Train windows: {len(train_ds):,} | Val windows: {len(val_ds):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    model.train()
    step = 0
    t0 = time.time()
    best_val = float("inf")
    data_iter = iter(train_loader)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
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
            print(f"step {step:4d}/{args.steps}  train_loss={loss.item():.4f}  ({elapsed:.1f}s)")

        if step % args.eval_every == 0 or step == args.steps:
            if len(val_ds) > 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"         val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    args.output.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(args.output)
                    print(f"         saved best -> {args.output}")

    # Always save final weights as well
    final_dir = args.output.parent / (args.output.name + "_final")
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"Saved final checkpoint -> {final_dir}")
    print(f"Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
