"""One-off helper: concatenate the project's own .py sources into a seed
corpus for the first tokenizer training run (datasets/raw is gitignored,
so this script is safe to re-run locally without polluting the repo)."""

from __future__ import annotations

import pathlib

ROOTS = ["apps", "packages", "tests", "training/scripts"]
OUT = pathlib.Path("datasets/raw/aethermind_code_corpus.txt")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    file_count = 0
    for root in ROOTS:
        for path in sorted(pathlib.Path(root).rglob("*.py")):
            chunks.append(path.read_text(encoding="utf-8"))
            file_count += 1
    OUT.write_text("\n\n".join(chunks), encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size} bytes from {file_count} files")


if __name__ == "__main__":
    main()
