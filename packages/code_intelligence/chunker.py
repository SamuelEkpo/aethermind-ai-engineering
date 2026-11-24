"""Split source files into overlapping, line-addressed chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A contiguous run of lines from one file. Lines are 1-based and inclusive."""

    path: str
    start_line: int
    end_line: int
    text: str


def chunk_text(path: str, text: str, *, max_lines: int = 40, overlap: int = 8) -> list[Chunk]:
    """Split ``text`` into chunks of at most ``max_lines`` lines.

    Consecutive chunks share ``overlap`` lines so a definition straddling a
    boundary is still retrievable. Whitespace-only chunks are dropped.
    """
    if max_lines < 1:
        raise ValueError("max_lines must be >= 1")
    if not 0 <= overlap < max_lines:
        raise ValueError("overlap must satisfy 0 <= overlap < max_lines")

    lines = text.splitlines()
    step = max_lines - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        body = "\n".join(lines[start:end])
        if body.strip():
            chunks.append(Chunk(path=path, start_line=start + 1, end_line=end, text=body))
        if end >= len(lines):
            break
        start += step
    return chunks
