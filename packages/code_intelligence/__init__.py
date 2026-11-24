"""
code_intelligence — repository indexing and retrieval over a codebase.

Implemented:
    - chunker.py      line-addressed, overlapping chunks
    - code_tokens.py  identifier-aware tokenizer (snake_case / camelCase)
    - bm25.py         dependency-free BM25 ranking
    - index.py        RepoIndex: scan, chunk, search, save/load. Files with
                      detected secrets are never indexed.

Planned (roadmap Phase 7): embedding-based (FAISS) retrieval alongside BM25,
and static structure/dependency analysis.
"""

from .bm25 import BM25Index
from .chunker import Chunk, chunk_text
from .code_tokens import split_identifier, tokenize
from .index import DEFAULT_EXTENSIONS, IndexStats, RepoIndex, SearchHit

__all__ = [
    "DEFAULT_EXTENSIONS",
    "BM25Index",
    "Chunk",
    "IndexStats",
    "RepoIndex",
    "SearchHit",
    "chunk_text",
    "split_identifier",
    "tokenize",
]
