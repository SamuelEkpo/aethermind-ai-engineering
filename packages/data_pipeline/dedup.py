"""Exact-duplicate detection for corpus ingestion."""

from __future__ import annotations

import hashlib


class ExactDeduper:
    """Tracks content hashes; line endings and surrounding whitespace are ignored."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """Return True if equivalent text was seen before; otherwise remember it."""
        normalised = text.replace("\r\n", "\n").strip()
        digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
        if digest in self._seen:
            return True
        self._seen.add(digest)
        return False
