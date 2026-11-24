"""Code-aware tokenization shared by indexing and querying."""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")

MIN_TOKEN_LEN = 2


def split_identifier(identifier: str) -> list[str]:
    """Split snake_case / camelCase / PascalCase / HTTPServer into lowercase parts."""
    spaced = _ACRONYM_BOUNDARY.sub(r"\1 \2", identifier)
    spaced = _CAMEL_BOUNDARY.sub(r"\1 \2", spaced)
    return [p.lower() for p in re.split(r"[_\s]+", spaced) if p]


def tokenize(text: str) -> list[str]:
    """Tokenize source or a natural-language query.

    Every identifier yields its full lowercase form plus its sub-words, so the
    query "parse request" matches ``parseHttpRequest`` and ``parse_request``.
    Single characters and bare numbers are dropped as noise.
    """
    tokens: list[str] = []
    for match in _IDENT.finditer(text):
        ident = match.group(0)
        parts = split_identifier(ident)
        full = ident.lower().replace("_", "")
        if len(full) >= MIN_TOKEN_LEN:
            tokens.append(full)
        if len(parts) > 1:
            tokens.extend(p for p in parts if len(p) >= MIN_TOKEN_LEN and p != full)
    return tokens
