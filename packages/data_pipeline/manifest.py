"""Provenance manifest for an ingested corpus (see licensing-strategy.md section 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class IngestReport:
    """Everything recorded about one ingestion run."""

    name: str
    source_url: str
    commit: str
    license_ids: tuple[str, ...]
    license_evidence: tuple[str, ...]
    license_method: str
    excluded_dirs: tuple[str, ...]
    extensions: tuple[str, ...]
    max_file_kb: int
    secret_action: str
    output_path: str = ""
    files_seen: int = 0
    files_kept: int = 0
    bytes_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    redactions: dict[str, int] = field(default_factory=dict)
    dropped_files: dict[str, list[str]] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(UTC).date().isoformat())


def _counts(title: str, counts: dict[str, int], empty: str) -> list[str]:
    lines = [f"## {title}", ""]
    if not counts:
        lines.append(empty)
    else:
        lines += ["| Reason | Count |", "|---|---|"]
        lines += [f"| {key} | {value} |" for key, value in sorted(counts.items())]
    lines.append("")
    return lines


def read_consumed_by(text: str) -> list[str]:
    """Return the bullet lines under '## Consumed by' in an existing manifest."""
    lines = text.splitlines()
    try:
        start = lines.index("## Consumed by") + 1
    except ValueError:
        return []
    items: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.startswith("- ") and line != "- (none yet)":
            items.append(line)
    return items


def render_manifest(report: IngestReport, consumed_by: list[str] | None = None) -> str:
    """Render the report as a Markdown manifest.

    ``consumed_by`` carries over the 'Consumed by' bullets of a manifest that is
    being regenerated, so re-ingesting a source does not erase its usage record.
    """
    evidence = ", ".join(report.license_evidence)
    lines = [
        f"# Corpus manifest: {report.name}",
        "",
        f"- **Source:** {report.source_url}",
        f"- **Commit / snapshot:** {report.commit}",
        "- **Collection method:** packages.data_pipeline.cli over a local checkout",
        f"- **Created (UTC):** {report.created}",
        f"- **Output file:** {report.output_path}",
        f"- **License(s):** {', '.join(report.license_ids)}",
        f"- **License evidence:** {evidence}",
        f"- **License detection method:** {report.license_method}. Independent "
        "confirmation with scancode-toolkit or licensee is still recommended.",
    ]
    if report.excluded_dirs:
        excluded = ", ".join(report.excluded_dirs)
        lines.append(f"- **Directories excluded (nested license not allowed):** {excluded}")
    lines += [
        "",
        "## Size",
        "",
        f"- Files considered: {report.files_seen}",
        f"- Files kept: {report.files_kept}",
        f"- Output bytes: {report.bytes_out}",
        "- Token count: not yet tokenized",
        "",
        "## Filters",
        "",
        f"- Extensions: {', '.join(report.extensions)}",
        f"- Maximum file size: {report.max_file_kb} KB",
        "- Skipped: binary, non-UTF-8, empty, generated, very long lines, exact duplicates",
        "",
    ]
    lines += _counts("Dropped files", report.dropped, "No files were dropped.")
    if report.dropped_files:
        lines += ["## Dropped file paths", ""]
        for reason, paths in sorted(report.dropped_files.items()):
            extra = report.dropped.get(reason, 0) - len(paths)
            suffix = f" (+{extra} more)" if extra > 0 else ""
            lines.append(f"- {reason}: " + ", ".join(paths) + suffix)
        lines.append("")
    lines += _counts("PII redactions", report.redactions, "No PII patterns were redacted.")
    lines += [
        "## Secret scan",
        "",
        "- Scanner: built-in pattern rules (private keys, cloud/API tokens, credentials)",
        f"- Policy on a hit: {report.secret_action}",
        "- Result: no flagged content in kept files",
        "",
        "## Known limitations",
        "",
        "- License detection reads license files only, not per-file headers.",
        "- Only exact duplicates are removed; near-duplicates are not.",
        "",
        "## Consumed by",
        "",
        *(consumed_by or ["- (none yet)"]),
        "",
    ]
    return "\n".join(lines)
