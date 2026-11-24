"""Command-line entry point.

Run from the repository root:

    python -m packages.data_pipeline.cli --source C:\\path\\to\\repo --name my-source ...

Exit codes: 0 success, 1 ingestion refused (license/secret/empty), 2 bad arguments.
Nothing is written to the corpus or manifest when the exit code is non-zero.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .errors import IngestError
from .manifest import render_manifest
from .pipeline import SECRET_ACTIONS, ingest_source

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a licensed source tree into a cleaned corpus file plus manifest."
    )
    parser.add_argument("--source", type=Path, required=True, help="Local checkout to ingest")
    parser.add_argument("--name", required=True, help="Slug for outputs (a-z, 0-9, - and _)")
    parser.add_argument("--source-url", required=True, help="Where the source came from")
    parser.add_argument("--commit", required=True, help="Commit hash or snapshot identifier")
    parser.add_argument("--extensions", default=".py,.md", help="Comma-separated file extensions")
    parser.add_argument("--max-file-kb", type=int, default=256)
    parser.add_argument("--secret-action", choices=SECRET_ACTIONS, default="fail")
    parser.add_argument("--out", type=Path, default=None, help="default: datasets/processed/<name>.txt")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="default: datasets/manifests/<name>.md"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not _SLUG_RE.match(args.name):
        print(f"ERROR: invalid --name {args.name!r}; use a-z, 0-9, '-' and '_'", file=sys.stderr)
        return 2
    extensions = tuple(
        ext if ext.startswith(".") else f".{ext}"
        for ext in (part.strip().lower() for part in args.extensions.split(","))
        if ext
    )
    if not extensions:
        print("ERROR: --extensions must name at least one extension", file=sys.stderr)
        return 2

    out_path: Path = args.out or Path("datasets/processed") / f"{args.name}.txt"
    manifest_path: Path = args.manifest or Path("datasets/manifests") / f"{args.name}.md"

    try:
        report = ingest_source(
            args.source,
            name=args.name,
            source_url=args.source_url,
            commit=args.commit,
            extensions=extensions,
            max_file_kb=args.max_file_kb,
            out_path=out_path,
            secret_action=args.secret_action,
        )
    except IngestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(render_manifest(report), encoding="utf-8")
    print(
        f"{out_path}: {report.bytes_out} bytes from {report.files_kept} of "
        f"{report.files_seen} files; manifest: {manifest_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
