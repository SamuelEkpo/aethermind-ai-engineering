"""
Evaluate a trained AetherMindLM checkpoint (loss + perplexity).

Usage:
    python -m training.scripts.evaluate_lm \
        --checkpoint training/checkpoints/lm_tiny \
        --tokenizer training/checkpoints/tokenizer/tokenizer.json \
        --corpus-dir datasets/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

from packages.evaluation import compute_perplexity
from packages.model_core import AetherMindLM
from packages.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("training/checkpoints/lm_tiny")
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("training/checkpoints/tokenizer/tokenizer.json"),
    )
    parser.add_argument("--corpus-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = AetherMindLM.from_pretrained(args.checkpoint, map_location=args.device)
    model.to(args.device)

    texts = [
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted(args.corpus_dir.rglob("*.txt"))
    ]
    if not texts:
        raise SystemExit(f"No .txt files under {args.corpus_dir}")

    ids: list[int] = []
    for t in texts:
        ids.extend(tokenizer.encode(t))

    # Use last 15% as a crude held-out slice for reporting
    split = int(len(ids) * 0.85)
    eval_ids = ids[split:] if split < len(ids) else ids

    metrics = compute_perplexity(
        model,
        eval_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
    )
    print(f"tokens_evaluated: {int(metrics['tokens_evaluated']):,}")
    print(f"loss:             {metrics['loss']:.4f}")
    print(f"perplexity:       {metrics['perplexity']:.2f}")


if __name__ == "__main__":
    main()
