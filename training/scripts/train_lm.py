"""
Reproducible training entrypoint for AetherMindLM.

Features:
  - chronological train/val split
  - cosine LR decay with linear warmup
  - deterministic seed control
  - best + last + final checkpointing
  - resumable training state (optimizer, step, RNG, best_val)
  - JSONL metrics logging (train loss, val loss/perplexity, LR, throughput)

Usage:
    python -m training.scripts.train_lm --config tiny --steps 500
    python -m training.scripts.train_lm --config tiny --steps 1000 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from packages.model_core import BASE, SMALL, TINY, AetherMindLM, ModelConfig
from packages.tokenizer import BPETokenizer
from torch.utils.data import DataLoader, Dataset

CONFIGS = {
    "tiny": TINY,
    "small": SMALL,
    "base": BASE,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        p.read_text(encoding="utf-8", errors="ignore") for p in sorted(corpus_dir.rglob("*.txt"))
    ]
    if not texts:
        raise SystemExit(f"No .txt files found under {corpus_dir}")
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
    return ids


def cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup then cosine decay to min_lr_ratio * base_lr."""
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


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
        if n >= 20:
            break
    model.train()
    return total_loss / max(n, 1)


class MetricsLogger:
    """Append-only JSONL metrics log — one line per logged event.

    Kept intentionally lightweight (no external dependency) so training can
    run without a full experiment-tracking stack (e.g. MLflow) configured.
    """

    def __init__(self, path: Path, run_config: dict[str, Any]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Record a header line describing this run's hyperparameters once.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "run_start", **run_config}) + "\n")

    def log(self, **fields: Any) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "step", **fields}) + "\n")


def save_training_state(
    state_dir: Path,
    *,
    step: int,
    best_val: float,
    optimizer: torch.optim.Optimizer,
    seed: int,
) -> None:
    """Persist everything needed to resume training exactly where it left off."""
    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_val": best_val,
            "seed": seed,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        },
        state_dir / "trainer_state.pt",
    )


def load_training_state(state_dir: Path, optimizer: torch.optim.Optimizer) -> tuple[int, float]:
    """Restore optimizer momentum, RNG state, step count, and best_val."""
    state = torch.load(state_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng_state"])
    random.setstate(state["python_rng_state"])
    return state["step"], state["best_val"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="tiny")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Also write a 'last' checkpoint + resumable state every N steps (0 = only at eval-every)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the '<output>_last' checkpoint + trainer state, if present",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("training/checkpoints/tokenizer/tokenizer.json"),
    )
    parser.add_argument("--corpus-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output", type=Path, default=Path("training/checkpoints/lm_tiny"))
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="JSONL metrics log path (default: <output>_logs/metrics.jsonl)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    base_config: ModelConfig = CONFIGS[args.config]

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

    print(
        f"Config: {args.config} | d_model={config.d_model} n_layers={config.n_layers} "
        f"seed={args.seed}"
    )
    device = torch.device(args.device)

    token_ids = load_token_ids(args.tokenizer, args.corpus_dir)
    print(f"Corpus tokens: {len(token_ids):,}")

    split = int(len(token_ids) * (1.0 - args.val_ratio))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

    seq_len = min(args.seq_len, config.max_seq_len)
    train_ds = TextDataset(train_ids, seq_len=seq_len)
    val_ds = TextDataset(val_ids, seq_len=seq_len)
    if len(train_ds) == 0:
        raise SystemExit("Training split too short for the requested seq_len")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    last_dir = args.output.parent / (args.output.name + "_last")
    state_dir = args.output.parent / (args.output.name + "_state")

    resumed = False
    if args.resume and (last_dir / "pytorch_model.bin").exists():
        model = AetherMindLM.from_pretrained(last_dir, map_location=str(device))
        resumed = True
    else:
        if args.resume:
            print(f"--resume given but no checkpoint at {last_dir}; starting fresh")
        model = AetherMindLM(config)
    model.to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print(f"Train windows: {len(train_ds):,} | Val windows: {len(val_ds):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    start_step = 0
    best_val = float("inf")
    if resumed and (state_dir / "trainer_state.pt").exists():
        start_step, best_val = load_training_state(state_dir, optimizer)
        print(f"Resumed from step {start_step} (best_val={best_val:.4f})")

    log_path = args.log_file or (
        args.output.parent / (args.output.name + "_logs") / "metrics.jsonl"
    )
    logger = MetricsLogger(
        log_path,
        {
            "config": args.config,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": seq_len,
            "lr": args.lr,
            "seed": args.seed,
            "resumed": resumed,
            "start_step": start_step,
            "parameters": model.count_parameters(),
        },
    )

    model.train()
    step = start_step
    t0 = time.time()
    data_iter = iter(train_loader)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lr = cosine_lr(step, args.steps, args.lr, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

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
            print(
                f"step {step:4d}/{args.steps}  train_loss={loss.item():.4f}  "
                f"lr={lr:.2e}  ({elapsed:.1f}s)"
            )
            logger.log(step=step, train_loss=loss.item(), lr=lr, elapsed_s=round(elapsed, 1))

        periodic_save = args.save_every > 0 and step % args.save_every == 0
        at_eval = step % args.eval_every == 0 or step == args.steps

        if at_eval and len(val_ds) > 0:
            val_loss = evaluate(model, val_loader, device)
            ppl = math.exp(min(val_loss, 20.0))
            print(f"         val_loss={val_loss:.4f}  ppl={ppl:.2f}")
            logger.log(step=step, val_loss=val_loss, perplexity=ppl)
            if val_loss < best_val:
                best_val = val_loss
                args.output.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(args.output)
                print(f"         saved best -> {args.output}")

        if periodic_save or at_eval:
            model.save_pretrained(last_dir)
            save_training_state(
                state_dir,
                step=step,
                best_val=best_val,
                optimizer=optimizer,
                seed=args.seed,
            )

    final_dir = args.output.parent / (args.output.name + "_final")
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    model.save_pretrained(last_dir)
    save_training_state(
        state_dir, step=step, best_val=best_val, optimizer=optimizer, seed=args.seed
    )
    logger.log(step=step, event_final=True, best_val=best_val)
    print(f"Saved final checkpoint -> {final_dir}")
    print(f"Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
