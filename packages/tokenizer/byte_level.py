"""
Byte-level encoding utilities for the AetherMind BPE tokenizer.

Maps every possible byte value (0-255) to a unique, printable Unicode
character so that BPE training and merging can operate on ordinary
strings while still being able to losslessly round-trip arbitrary
UTF-8 text (including source code, emoji, and control characters).
This is the same trick used by GPT-2's tokenizer.
"""

from functools import lru_cache


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    """Return a reversible mapping from byte value (0-255) to a printable char."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


@lru_cache
def unicode_to_bytes() -> dict[str, int]:
    """Inverse of `bytes_to_unicode`."""
    return {ch: b for b, ch in bytes_to_unicode().items()}


def encode_bytes(text: str) -> str:
    """Encode a string's UTF-8 bytes as a string of printable byte-symbols."""
    byte_encoder = bytes_to_unicode()
    return "".join(byte_encoder[b] for b in text.encode("utf-8"))


def decode_bytes(symbols: str) -> str:
    """Decode a string of byte-symbols back into the original UTF-8 string."""
    byte_decoder = unicode_to_bytes()
    raw = bytes(byte_decoder[ch] for ch in symbols)
    return raw.decode("utf-8", errors="replace")
