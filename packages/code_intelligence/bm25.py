"""Okapi BM25 ranking over pre-tokenized documents (pure Python, no dependencies)."""

from __future__ import annotations

import math
from collections import Counter


class BM25Index:
    """In-memory inverted index with BM25 scoring.

    Deterministic: ties are broken by document order, so results are stable
    across runs and platforms.
    """

    def __init__(self, docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("require k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self._doc_lens = [len(d) for d in docs]
        self._avg_len = (sum(self._doc_lens) / self.n_docs) if self.n_docs else 0.0
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, tokens in enumerate(docs):
            for term, freq in Counter(tokens).items():
                self._postings.setdefault(term, []).append((doc_id, freq))

    def _idf(self, term: str) -> float:
        df = len(self._postings[term])
        # The +1 inside the log keeps idf positive for very common terms.
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], k: int = 5) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(doc_id, score)`` pairs, best first; zero scores omitted."""
        if k < 1 or not self.n_docs:
            return []
        scores: dict[int, float] = {}
        for term in set(query_tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_id, freq in postings:
                norm = 1.0 - self.b + self.b * (self._doc_lens[doc_id] / (self._avg_len or 1.0))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (
                    freq * (self.k1 + 1.0) / (freq + self.k1 * norm)
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]
