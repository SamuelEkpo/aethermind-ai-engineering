"""Tests for packages.data_pipeline (license policy, filters, scans, ingestion, CLI)."""

from __future__ import annotations

from pathlib import Path

import pytest
from packages.data_pipeline.cli import main as cli_main
from packages.data_pipeline.dedup import ExactDeduper
from packages.data_pipeline.errors import IngestError, LicenseError, SecretFoundError
from packages.data_pipeline.filters import load_text
from packages.data_pipeline.licenses import ALLOWED_SPDX, check_source_license, detect_license
from packages.data_pipeline.manifest import IngestReport, render_manifest
from packages.data_pipeline.pii import redact_pii
from packages.data_pipeline.pipeline import ingest_source
from packages.data_pipeline.secret_scan import scan_secrets

MIT_TEXT = (
    "MIT License\n\nCopyright (c) 2024 Someone\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    "of this software and associated documentation files.\n"
)
APACHE_TEXT = "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n"
BSD3_TEXT = (
    "Redistribution and use in source and binary forms, with or without modification,\n"
    "are permitted provided that the following conditions are met.\n"
    "Neither the name of the copyright holder nor the names of its contributors may be used.\n"
)
BSD2_TEXT = (
    "Redistribution and use in source and binary forms, with or without modification,\n"
    "are permitted provided that the following conditions are met.\n"
)
BSD4_TEXT = BSD3_TEXT + "All advertising materials mentioning features must display credit.\n"
ISC_TEXT = (
    "Permission to use, copy, modify, and/or distribute this software for any purpose\n"
    "with or without fee is hereby granted.\n"
)
CC0_TEXT = "CC0 1.0 Universal\n\nStatement of Purpose\n"
GPL_TEXT = "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"
LGPL_TEXT = "GNU LESSER GENERAL PUBLIC LICENSE\nVersion 3\n"
AGPL_TEXT = "GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3\n"
MPL_TEXT = "Mozilla Public License Version 2.0\n"

# Fake credentials are assembled from parts so this file is not itself flagged.
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
PRIVATE_KEY_HEADER = "-----BEGIN " + "RSA PRIVATE KEY-----"


def make_repo(root: Path, files: dict[str, str], license_text: str | None = MIT_TEXT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if license_text is not None:
        (root / "LICENSE").write_text(license_text, encoding="utf-8")
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def ingest(repo: Path, out: Path, **overrides: object) -> IngestReport:
    kwargs: dict[str, object] = {
        "name": "demo",
        "source_url": "https://example.test/demo",
        "commit": "abc123",
        "extensions": (".py", ".md"),
        "max_file_kb": 64,
        "out_path": out,
    }
    kwargs.update(overrides)
    return ingest_source(repo, **kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# License detection and policy
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (MIT_TEXT, "MIT"),
        (APACHE_TEXT, "Apache-2.0"),
        (BSD3_TEXT, "BSD-3-Clause"),
        (BSD2_TEXT, "BSD-2-Clause"),
        (BSD4_TEXT, "BSD-4-Clause"),
        (ISC_TEXT, "ISC"),
        (CC0_TEXT, "CC0-1.0"),
        (GPL_TEXT, "GPL"),
        (LGPL_TEXT, "LGPL"),
        (AGPL_TEXT, "AGPL"),
        (MPL_TEXT, "MPL"),
        ("Custom terms: do whatever you like.", None),
    ],
)
def test_detect_license(text: str, expected: str | None) -> None:
    assert detect_license(text) == expected


def test_allowlist_excludes_copyleft_and_bsd4() -> None:
    assert {"GPL", "LGPL", "AGPL", "MPL", "BSD-4-Clause"}.isdisjoint(ALLOWED_SPDX)


def test_check_source_license_accepts_mit(tmp_path: Path) -> None:
    result = check_source_license(make_repo(tmp_path / "r", {}))
    assert result.spdx_ids == ("MIT",)
    assert result.spdx == "MIT"
    assert result.evidence == ("LICENSE",)
    assert result.excluded_dirs == ()


def test_check_source_license_accepts_dual_license_files(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {}, license_text=None)
    (repo / "LICENSE-MIT").write_text(MIT_TEXT, encoding="utf-8")
    (repo / "LICENSE-APACHE").write_text(APACHE_TEXT, encoding="utf-8")
    result = check_source_license(repo)
    assert result.spdx_ids == ("Apache-2.0", "MIT")
    assert set(result.evidence) == {"LICENSE-MIT", "LICENSE-APACHE"}


def test_check_source_license_rejects_missing_license(tmp_path: Path) -> None:
    with pytest.raises(LicenseError, match="No LICENSE"):
        check_source_license(make_repo(tmp_path / "r", {"a.py": "x = 1\n"}, license_text=None))


def test_check_source_license_rejects_copyleft(tmp_path: Path) -> None:
    with pytest.raises(LicenseError, match="GPL"):
        check_source_license(make_repo(tmp_path / "r", {}, license_text=GPL_TEXT))


def test_check_source_license_rejects_unrecognised_text(tmp_path: Path) -> None:
    with pytest.raises(LicenseError, match="Unrecognised"):
        check_source_license(make_repo(tmp_path / "r", {}, license_text="Custom terms."))


def test_check_source_license_rejects_non_directory(tmp_path: Path) -> None:
    with pytest.raises(LicenseError, match="not a directory"):
        check_source_license(tmp_path / "missing")


def test_nested_disallowed_license_excludes_that_directory(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"src/s.py": "s = 1\n", "vendor/v.py": "v = 1\n"})
    (repo / "vendor" / "LICENSE").write_text(GPL_TEXT, encoding="utf-8")
    result = check_source_license(repo)
    assert result.excluded_dirs == ("vendor",)


# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"x" * 10, "too-large"),
        (b"a\x00b", "binary"),
        (b"\xff\xfe", "not-utf8"),
        (b"  \n", "empty"),
        (b"# Auto-generated file\nx = 1\n", "generated"),
        (("x" * 1001).encode(), "long-lines"),
    ],
)
def test_load_text_rejections(data: bytes, reason: str) -> None:
    max_bytes = 5 if reason == "too-large" else 10_000
    assert load_text(data, max_bytes) == (None, reason)


def test_load_text_accepts_and_strips_bom() -> None:
    assert load_text(b"\xef\xbb\xbfx = 1\n", 1000) == ("x = 1\n", None)


def test_load_text_normalises_line_endings() -> None:
    assert load_text(b"a = 1\r\nb = 2\rc = 3\n", 1000) == ("a = 1\nb = 2\nc = 3\n", None)


# ----------------------------------------------------------------------
# Secrets, PII, dedup
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (PRIVATE_KEY_HEADER, "private-key"),
        (f"key = '{AWS_KEY}'", "aws-access-key"),
        ("t = 'ghp_" + "a" * 36 + "'", "github-token"),
        ("t = 'xoxb-" + "1234567890-abcdefghij" + "'", "slack-token"),
        ("k = 'AIza" + "A" * 35 + "'", "google-api-key"),
        ('password = "hunter2hunter2hunter2"', "generic-credential"),
    ],
)
def test_scan_secrets_detects_rule(text: str, rule: str) -> None:
    assert rule in scan_secrets(text)


def test_scan_secrets_ignores_clean_code() -> None:
    assert scan_secrets("def add(a, b):\n    return a + b\n") == []


def test_redact_pii_replaces_emails_and_phones() -> None:
    text, counts = redact_pii("mail a@b.com or c@d.org, call 555-123-4567 or +1 (555) 123-4567")
    assert "a@b.com" not in text
    assert "555-123-4567" not in text
    assert counts == {"email": 2, "phone": 2}
    assert "<EMAIL>" in text
    assert "<PHONE>" in text


def test_redact_pii_leaves_clean_text_alone() -> None:
    assert redact_pii("x = 1") == ("x = 1", {})


def test_exact_deduper_ignores_line_endings_and_edge_whitespace() -> None:
    deduper = ExactDeduper()
    assert deduper.is_duplicate("a = 1\r\n") is False
    assert deduper.is_duplicate("a = 1\n") is True
    assert deduper.is_duplicate("b = 2\n") is False


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------
def test_ingest_happy_path_filters_dedups_and_redacts(tmp_path: Path) -> None:
    repo = make_repo(
        tmp_path / "r",
        {
            "pkg/a.py": "def a():\n    return 1\n",
            "pkg/b.py": "def b():\n    return 2\n",
            "pkg/dup.py": "def a():\n    return 1\n",
            "pkg/email.py": 'AUTHOR = "dev@example.com"\n',
            "pkg/notes.txt": "ignored extension\n",
            "node_modules/x.py": "skipped = True\n",
        },
    )
    (repo / "pkg" / "blob.py").write_bytes(b"\x00\x01\x02")
    out = tmp_path / "out" / "demo.txt"

    report = ingest(repo, out)

    assert report.files_seen == 5
    assert report.files_kept == 3
    assert report.dropped == {"binary": 1, "duplicate": 1}
    assert report.redactions == {"email": 1}
    text = out.read_text(encoding="utf-8")
    assert "def b()" in text
    assert "<EMAIL>" in text
    assert "dev@example.com" not in text
    assert text.endswith("\n")
    assert report.bytes_out == len(text.encode("utf-8"))
    assert report.license_ids == ("MIT",)
    assert not out.with_suffix(".txt.tmp").exists()


def test_ingest_drops_oversized_files(tmp_path: Path) -> None:
    repo = make_repo(
        tmp_path / "r", {"small.py": "x = 1\n", "big.py": "y = 1\n" + "# pad\n" * 400}
    )
    report = ingest(repo, tmp_path / "out.txt", max_file_kb=1)
    assert report.files_kept == 1
    assert report.dropped == {"too-large": 1}


def test_ingest_fails_on_secret_and_writes_nothing(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"ok.py": "x = 1\n", "bad.py": f"KEY = '{AWS_KEY}'\n"})
    out = tmp_path / "out.txt"
    with pytest.raises(SecretFoundError, match="bad.py"):
        ingest(repo, out)
    assert not out.exists()


def test_ingest_can_exclude_files_with_secrets(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"ok.py": "x = 1\n", "bad.py": f"KEY = '{AWS_KEY}'\n"})
    report = ingest(repo, tmp_path / "out.txt", secret_action="exclude")
    assert report.files_kept == 1
    assert report.dropped == {"secret": 1}
    assert AWS_KEY not in (tmp_path / "out.txt").read_text(encoding="utf-8")


def test_ingest_rejects_invalid_secret_action(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"a.py": "x = 1\n"})
    with pytest.raises(ValueError, match="secret_action"):
        ingest(repo, tmp_path / "out.txt", secret_action="ignore")


def test_ingest_refuses_disallowed_license(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"a.py": "x = 1\n"}, license_text=GPL_TEXT)
    out = tmp_path / "out.txt"
    with pytest.raises(LicenseError):
        ingest(repo, out)
    assert not out.exists()


def test_ingest_fails_when_nothing_survives(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"notes.txt": "wrong extension\n"})
    out = tmp_path / "out.txt"
    with pytest.raises(IngestError, match="No files were kept"):
        ingest(repo, out)
    assert not out.exists()


def test_ingest_skips_directories_with_disallowed_nested_license(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"src/s.py": "s = 1\n", "vendor/v.py": "v = 1\n"})
    (repo / "vendor" / "LICENSE").write_text(GPL_TEXT, encoding="utf-8")
    out = tmp_path / "out.txt"
    report = ingest(repo, out)
    assert report.excluded_dirs == ("vendor",)
    text = out.read_text(encoding="utf-8")
    assert "s = 1" in text
    assert "v = 1" not in text


def test_render_manifest_records_provenance_and_counts(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"a.py": "a = 1\n", "b.py": "a = 1\n"})
    report = ingest(repo, tmp_path / "out.txt")
    manifest = render_manifest(report)
    assert "# Corpus manifest: demo" in manifest
    assert "abc123" in manifest
    assert "MIT" in manifest
    assert "| duplicate | 1 |" in manifest
    assert "No PII patterns were redacted." in manifest
    assert "Policy on a hit: fail" in manifest


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _cli_args(repo: Path, out: Path, manifest: Path, *extra: str) -> list[str]:
    return [
        "--source",
        str(repo),
        "--name",
        "demo",
        "--source-url",
        "https://example.test/demo",
        "--commit",
        "abc123",
        "--out",
        str(out),
        "--manifest",
        str(manifest),
        *extra,
    ]


def test_cli_success_writes_corpus_and_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path / "r", {"a.py": "a = 1\n"})
    out = tmp_path / "out" / "demo.txt"
    manifest = tmp_path / "manifests" / "demo.md"
    assert cli_main(_cli_args(repo, out, manifest, "--extensions", "py,.md")) == 0
    assert out.exists()
    assert "abc123" in manifest.read_text(encoding="utf-8")
    assert "1 of 1 files" in capsys.readouterr().out


def test_cli_license_failure_returns_1_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path / "r", {"a.py": "a = 1\n"}, license_text=GPL_TEXT)
    out = tmp_path / "out.txt"
    manifest = tmp_path / "demo.md"
    assert cli_main(_cli_args(repo, out, manifest)) == 1
    assert not out.exists()
    assert not manifest.exists()
    assert "ERROR" in capsys.readouterr().err


def test_cli_rejects_invalid_name(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"a.py": "a = 1\n"})
    args = _cli_args(repo, tmp_path / "o.txt", tmp_path / "m.md")
    args[args.index("demo")] = "Bad Name"
    assert cli_main(args) == 2


def test_cli_rejects_empty_extensions(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", {"a.py": "a = 1\n"})
    args = _cli_args(repo, tmp_path / "o.txt", tmp_path / "m.md", "--extensions", " , ")
    assert cli_main(args) == 2
