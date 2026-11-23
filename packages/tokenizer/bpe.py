"""
Byte-level Byte Pair Encoding (BPE) tokenizer, trained from scratch.

This is AetherMind's own tokenizer implementation (see README Section 4.3
and the Phase 3 roadmap) rather than a wrapper around a third-party
library: training builds a vocabulary of merge rules directly from a
text corpus, and encoding/decoding apply those rules losslessly over raw
UTF-8 bytes so any input — natural language or source code — can always
be represented, with no "unknown character" failure mode.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from packages.tokenizer.byte_level import bytes_to_unicode, decode_bytes, encode_bytes
from packages.tokenizer.pretokenize import pretokenize

DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")

Pair = tuple[str, str]
Word = tuple[str, ...]


class BPETokenizer:
    """A trained byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[Pair],
        special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        self.vocab = vocab
        self.id_to_token: dict[int, str] = {idx: token for token, idx in vocab.items()}
        self.merges = merges
        self.merge_ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        vocab_size: int,
        special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS,
    ) -> BPETokenizer:
        """Train a new tokenizer from an iterable of text documents.

        `vocab_size` includes the special tokens and the 256 base byte
        symbols, so it must be at least `len(special_tokens) + 256`.
        """
        min_size = len(special_tokens) + 256
        if vocab_size < min_size:
            raise ValueError(
                f"vocab_size must be >= {min_size} (special tokens + 256 base "
                f"bytes), got {vocab_size}"
            )

        word_freqs: Counter[Word] = Counter()
        for document in corpus:
            for piece in pretokenize(document):
                symbols = tuple(encode_bytes(piece))
                word_freqs[symbols] += 1

        # The base vocabulary always contains the full 256-byte alphabet,
        # regardless of what the training corpus happens to contain, so
        # that encoding can never fail on a byte it didn't happen to see
        # during training (e.g. a digit, if the corpus had none).
        vocab: dict[str, int] = {tok: i for i, tok in enumerate(special_tokens)}
        for symbol in bytes_to_unicode().values():
            vocab.setdefault(symbol, len(vocab))

        words: dict[Word, int] = dict(word_freqs)
        merges: list[Pair] = []

        while len(vocab) < vocab_size:
            pair_counts = cls._count_pairs(words)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            words = cls._merge_pair(words, best_pair)
            merges.append(best_pair)
            vocab["".join(best_pair)] = len(vocab)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @staticmethod
    def _count_pairs(words: dict[Word, int]) -> Counter[Pair]:
        pair_counts: Counter[Pair] = Counter()
        for word, freq in words.items():
            for a, b in zip(word, word[1:], strict=False):
                pair_counts[(a, b)] += freq
        return pair_counts

    @staticmethod
    def _merge_pair(words: dict[Word, int], pair: Pair) -> dict[Word, int]:
        merged = "".join(pair)
        new_words: dict[Word, int] = {}
        for word, freq in words.items():
            symbols = list(word)
            i = 0
            out: list[str] = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            key = tuple(out)
            new_words[key] = new_words.get(key, 0) + freq
        return new_words

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        while len(symbols) > 1:
            pairs = list(zip(symbols, symbols[1:], strict=False))
            ranked = [
                (self.merge_ranks[p], i) for i, p in enumerate(pairs) if p in self.merge_ranks
            ]
            if not ranked:
                break
            _, i = min(ranked)
            symbols = symbols[:i] + ["".join(symbols[i : i + 2])] + symbols[i + 2 :]
        return symbols

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids."""
        unk_id = self.vocab.get("<unk>")
        ids: list[int] = []
        for piece in pretokenize(text):
            symbols = self._apply_merges(list(encode_bytes(piece)))
            for symbol in symbols:
                token_id = self.vocab.get(symbol, unk_id)
                if token_id is not None:
                    ids.append(token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into text."""
        symbols = "".join(
            self.id_to_token[i]
            for i in ids
            if i in self.id_to_token and self.id_to_token[i] not in self.special_tokens
        )
        return decode_bytes(symbols)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save the vocabulary and merge rules to a single JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": list(self.special_tokens),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer previously saved with `save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges: list[Pair] = [tuple(pair) for pair in payload["merges"]]
        return cls(
            vocab=payload["vocab"],
            merges=merges,
            special_tokens=tuple(payload["special_tokens"]),
        )
