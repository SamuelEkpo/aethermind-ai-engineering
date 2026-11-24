"""
Sample text from a trained AetherMindLM checkpoint.

Usage:
    python -m training.scripts.generate \
        --checkpoint training/checkpoints/lm_tiny \
        --tokenizer training/checkpoints/tokenizer/tokenizer.json \
        --prompt "def train" \
        --max-new-tokens 80 \
        --temperature 0.8 \
        --top-k 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from packages.model_core import AetherMindLM
from packages.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("training/checkpoints/lm_tiny"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("training/checkpoints/tokenizer/tokenizer.json"),
    )
    parser.add_argument("--prompt", type=str, default="AetherMind")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = AetherMindLM.from_pretrained(args.checkpoint, map_location=args.device)
    model.to(args.device)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.vocab.get("<bos>", 2)]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    text = tokenizer.decode(output_ids[0].tolist())
    print("─" * 60)
    print(text)
    print("─" * 60)


if __name__ == "__main__":
    main()
