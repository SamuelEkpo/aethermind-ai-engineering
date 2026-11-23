from pathlib import Path

import pytest
from packages.tokenizer import BPETokenizer

SAMPLE_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks at the fox",
    "def foo(x, y):\n    return x + y\n",
]


def test_train_produces_a_vocab_within_the_requested_bound() -> None:
    tokenizer = BPETokenizer.train(SAMPLE_CORPUS, vocab_size=300)
    # Base vocab is always special_tokens (4) + the full byte alphabet (256).
    # Beyond that, how many merges are actually learned depends on how much
    # repetition the corpus has, so a tiny corpus won't necessarily reach
    # the requested size — it should never exceed it, though.
    assert 260 <= tokenizer.vocab_size <= 300
    assert len(tokenizer.merges) > 0


def test_encode_decode_roundtrip_is_lossless() -> None:
    tokenizer = BPETokenizer.train(SAMPLE_CORPUS, vocab_size=300)
    for text in [*SAMPLE_CORPUS, "unseen word with accents and code x=1+2"]:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text


def test_frequent_substrings_compress_below_byte_length() -> None:
    tokenizer = BPETokenizer.train(SAMPLE_CORPUS, vocab_size=300)
    # "the" and "dog" both repeat often in the corpus, so BPE should have
    # merged them into tokens, giving fewer ids than raw characters.
    ids = tokenizer.encode("the dog")
    assert len(ids) < len("the dog")


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.train(SAMPLE_CORPUS, vocab_size=300)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)
    assert loaded.vocab == tokenizer.vocab
    assert loaded.merges == tokenizer.merges
    assert loaded.encode("the quick fox") == tokenizer.encode("the quick fox")


def test_rejects_vocab_size_smaller_than_base_alphabet() -> None:
    with pytest.raises(ValueError):
        BPETokenizer.train(SAMPLE_CORPUS, vocab_size=10)
