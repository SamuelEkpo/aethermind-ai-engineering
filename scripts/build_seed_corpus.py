"""
Build / refresh the local seed corpus used for tokenizer + early LM training.

Pulls text from this repository only (safe, no external download required):
  - Python sources under apps/, packages/, tests/, training/, scripts/
  - Markdown docs (README + docs/)
  - Config / yaml files that contain useful natural language

datasets/raw is gitignored, so re-running this is local-only.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_CODE = ROOT / "datasets" / "raw" / "aethermind_code_corpus.txt"
OUT_DOCS = ROOT / "datasets" / "raw" / "aethermind_docs_corpus.txt"

CODE_ROOTS = ["apps", "packages", "tests", "training", "scripts"]
DOC_GLOBS = ["README.md", "docs/**/*.md", "*.md"]


def collect_files(roots: list[str], pattern: str) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for root in roots:
        base = ROOT / root
        if not base.exists():
            continue
        files.extend(sorted(base.rglob(pattern)))
    return files


def write_corpus(paths: list[pathlib.Path], out: pathlib.Path, label: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    kept = 0
    for path in paths:
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if len(text) < 40:
            continue
        chunks.append(f"# --- {path.relative_to(ROOT)} ---\n{text}")
        kept += 1
    out.write_text("\n\n".join(chunks), encoding="utf-8")
    size = out.stat().st_size
    print(f"{label}: {kept} files -> {out.relative_to(ROOT)} ({size:,} bytes)")


def main() -> None:
    py_files = collect_files(CODE_ROOTS, "*.py")
    write_corpus(py_files, OUT_CODE, "code")

    doc_files: list[pathlib.Path] = []
    for pattern in DOC_GLOBS:
        doc_files.extend(sorted(ROOT.glob(pattern)))
    # de-dupe while preserving order
    seen: set[pathlib.Path] = set()
    unique_docs: list[pathlib.Path] = []
    for p in doc_files:
        if p not in seen and p.is_file():
            seen.add(p)
            unique_docs.append(p)
    write_corpus(unique_docs, OUT_DOCS, "docs")


if __name__ == "__main__":
    main()
