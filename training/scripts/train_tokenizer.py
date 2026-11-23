"""
CLI to train AetherMind's BPE tokenizer on a text corpus.

Usage:
    python -m training.scripts.train_tokenizer \\
        --input-dir datasets/raw \\
        --vocab-size 32000 \\
        --output training/checkpoints/tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from packages.tokenizer import BPETokenizer


def iter_corpus(input_dir: Path) -> list[str]:
    """Read every `.txt` file under `input_dir` as one document each."""
    return [path.read_text(encoding="utf-8", errors="ignore") for path in sorted(input_dir.rglob("*.txt"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--output", type=Path, default=Path("training/checkpoints/tokenizer/tokenizer.json")
    )
    args = parser.parse_args()

    documents = iter_corpus(args.input_dir)
    if not documents:
        raise SystemExit(
            f"No .txt files found under {args.input_dir}. Add training text to "
            f"{args.input_dir}/ first (it's gitignored, so this is safe to populate "
            "locally)."
        )

    tokenizer = BPETokenizer.train(documents, vocab_size=args.vocab_size)
    tokenizer.save(args.output)
    print(f"Trained tokenizer with {tokenizer.vocab_size} tokens -> {args.output}")


if __name__ == "__main__":
    main()
