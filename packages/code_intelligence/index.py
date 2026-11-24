"""Repository index: scan a project, chunk it, and answer keyword queries with BM25."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from packages.data_pipeline.filters import SKIP_DIRS, load_text
from packages.data_pipeline.secret_scan import scan_secrets

from .bm25 import BM25Index
from .chunker import Chunk, chunk_text
from .code_tokens import tokenize

INDEX_FORMAT_VERSION = 1

DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".sh",
    ".ps1",
    ".sql",
    ".html",
    ".css",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass
class IndexStats:
    files_indexed: int = 0
    chunks: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def _doc_tokens(chunk: Chunk) -> list[str]:
    # The path is part of the document so queries can match file and module names.
    return tokenize(chunk.path) + tokenize(chunk.text)


def _is_skipped_dir(name: str, skip_dirs: frozenset[str]) -> bool:
    return name in skip_dirs or name.endswith(".egg-info")


class RepoIndex:
    """Searchable chunk index. Build from a directory or load a saved index."""

    def __init__(self, chunks: list[Chunk], stats: IndexStats | None = None) -> None:
        self.chunks = chunks
        self.stats = stats or IndexStats(
            files_indexed=len({c.path for c in chunks}), chunks=len(chunks)
        )
        self._bm25 = BM25Index([_doc_tokens(c) for c in chunks])

    # -- building -----------------------------------------------------------

    @classmethod
    def build(
        cls,
        root: str | Path,
        *,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
        max_file_kb: int = 256,
        max_lines: int = 40,
        overlap: int = 8,
        skip_dirs: frozenset[str] = SKIP_DIRS,
    ) -> RepoIndex:
        """Index every eligible file under ``root``.

        Files containing detected secrets are never indexed (their text could
        otherwise be retrieved into a prompt). Symlinks are not followed.
        """
        root_path = Path(root)
        if not root_path.is_dir():
            raise NotADirectoryError(f"not a directory: {root_path}")

        wanted = {e.lower() for e in extensions}
        max_bytes = max_file_kb * 1024
        stats = IndexStats()
        chunks: list[Chunk] = []

        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not _is_skipped_dir(d, skip_dirs))
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if path.suffix.lower() not in wanted:
                    continue
                if path.is_symlink():
                    stats.skip("symlink")
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        stats.skip("too-large")
                        continue
                    data = path.read_bytes()
                except OSError:
                    stats.skip("unreadable")
                    continue

                text, reason = load_text(data, max_bytes)
                if text is None:
                    stats.skip(reason or "rejected")
                    continue
                if scan_secrets(text):
                    stats.skip("secret")
                    continue

                rel = path.relative_to(root_path).as_posix()
                file_chunks = chunk_text(rel, text, max_lines=max_lines, overlap=overlap)
                if file_chunks:
                    stats.files_indexed += 1
                    chunks.extend(file_chunks)

        stats.chunks = len(chunks)
        return cls(chunks, stats)

    # -- querying -----------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Top-``k`` chunks for a natural-language or identifier query."""
        tokens = tokenize(query)
        if not tokens:
            return []
        return [
            SearchHit(chunk=self.chunks[doc_id], score=score)
            for doc_id, score in self._bm25.search(tokens, k)
        ]

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_FORMAT_VERSION,
            "chunks": [
                {
                    "path": c.path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "text": c.text,
                }
                for c in self.chunks
            ],
        }
        target.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RepoIndex:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload["version"] != INDEX_FORMAT_VERSION:
                raise ValueError(f"unsupported index version: {payload['version']}")
            chunks = [
                Chunk(
                    path=str(item["path"]),
                    start_line=int(item["start_line"]),
                    end_line=int(item["end_line"]),
                    text=str(item["text"]),
                )
                for item in payload["chunks"]
            ]
        except (KeyError, TypeError, ValueError) as exc:  # JSONDecodeError is a ValueError
            raise ValueError(f"malformed index file: {exc}") from exc
        return cls(chunks)
