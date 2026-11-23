"""
Quick, human-readable evaluation of a trained BPE tokenizer.

Reports vocabulary size, the compression ratio (bytes per token) on a
held-out sample, a lossless round-trip check, and the most interesting
learned merges. This is a developer-facing sanity check, not a
substitute for the rigorous evaluation in README Section 12.

Usage:
    python -m training.scripts.evaluate_tokenizer \\
        --tokenizer training/checkpoints/tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from packages.tokenizer import BPETokenizer

HELD_OUT_SAMPLES = [
    "AetherMind is an AI research and engineering project focused on "
    "developing an intelligent system for software engineering and programming.",
    "def create_app() -> FastAPI:\n    app = FastAPI(title=settings.app_name)\n    return app",
    "class BPETokenizer:\n    def encode(self, text: str) -> list[int]:\n        ...",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("training/checkpoints/tokenizer/tokenizer.json")
    )
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"Vocab size:    {tokenizer.vocab_size}")
    print(f"Merges learned: {len(tokenizer.merges)}")

    print("\nTop 15 learned merges (by training order, earliest = most frequent):")
    for pair in tokenizer.merges[:15]:
        print(f"  {pair[0]!r} + {pair[1]!r} -> {''.join(pair)!r}")

    print("\nHeld-out samples:")
    total_bytes = 0
    total_tokens = 0
    for text in HELD_OUT_SAMPLES:
        ids = tokenizer.encode(text)
        round_tripped = tokenizer.decode(ids)
        assert round_tripped == text, "lossless round-trip failed!"
        n_bytes = len(text.encode("utf-8"))
        total_bytes += n_bytes
        total_tokens += len(ids)
        print(f"  {n_bytes:4d} bytes -> {len(ids):4d} tokens  | {text[:50]!r}...")

    ratio = total_bytes / total_tokens
    print(f"\nOverall compression: {ratio:.2f} bytes/token over held-out samples")
    print("Lossless round-trip: OK for all samples")


if __name__ == "__main__":
    main()
