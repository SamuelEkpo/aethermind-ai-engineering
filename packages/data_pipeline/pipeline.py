"""Corpus ingestion: license check, filtering, secret scan, PII redaction, dedup."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from .dedup import ExactDeduper
from .errors import IngestError, SecretFoundError
from .filters import SKIP_DIRS, load_text
from .licenses import check_source_license
from .manifest import IngestReport
from .pii import redact_pii
from .secret_scan import scan_secrets

SECRET_ACTIONS = ("fail", "exclude")


def _bump(counts: dict[str, int], key: str, amount: int = 1) -> None:
    counts[key] = counts.get(key, 0) + amount


def _iter_files(
    root: Path, extensions: tuple[str, ...], excluded_dirs: tuple[str, ...]
) -> Iterator[Path]:
    excluded = {root / d for d in excluded_dirs}
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and (current / d) not in excluded
        )
        for name in sorted(filenames):
            path = current / name
            if path.suffix.lower() in extensions and not path.is_symlink():
                yield path


def ingest_source(
    source: Path,
    *,
    name: str,
    source_url: str,
    commit: str,
    extensions: tuple[str, ...],
    max_file_kb: int,
    out_path: Path,
    secret_action: str = "fail",
) -> IngestReport:
    """Ingest one local source tree into a single text corpus file.

    Raises ``IngestError`` (or a subclass) without writing anything if the
    license is not allowed, a secret is found under ``secret_action="fail"``,
    or no files survive filtering.
    """
    if secret_action not in SECRET_ACTIONS:
        raise ValueError(f"secret_action must be one of {SECRET_ACTIONS}, got {secret_action!r}")

    root = source.resolve()
    lic = check_source_license(root)
    report = IngestReport(
        name=name,
        source_url=source_url,
        commit=commit,
        license_ids=lic.spdx_ids,
        license_evidence=lic.evidence,
        license_method=lic.method,
        excluded_dirs=lic.excluded_dirs,
        extensions=extensions,
        max_file_kb=max_file_kb,
        secret_action=secret_action,
    )

    deduper = ExactDeduper()
    chunks: list[str] = []
    max_bytes = max_file_kb * 1024
    for path in _iter_files(root, extensions, lic.excluded_dirs):
        report.files_seen += 1
        rel = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
        except OSError:
            _bump(report.dropped, "unreadable")
            continue

        text, reason = load_text(data, max_bytes)
        if text is None:
            _bump(report.dropped, reason or "rejected")
            continue

        hits = scan_secrets(text)
        if hits:
            if secret_action == "fail":
                raise SecretFoundError(f"Possible secret ({', '.join(hits)}) in {rel}")
            _bump(report.dropped, "secret")
            continue

        if deduper.is_duplicate(text):
            _bump(report.dropped, "duplicate")
            continue

        text, redactions = redact_pii(text)
        for kind, count in redactions.items():
            _bump(report.redactions, kind, count)
        chunks.append(text.rstrip())

    if not chunks:
        raise IngestError("No files were kept after filtering; nothing to write")

    data_out = "\n\n".join(chunks) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data_out)
    tmp_path.replace(out_path)

    report.files_kept = len(chunks)
    report.bytes_out = len(data_out.encode("utf-8"))
    report.output_path = out_path.as_posix()
    return report
