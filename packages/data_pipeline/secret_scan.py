"""Secret/credential detection for corpus ingestion.

Pattern-based first pass. It favours catching too much over too little: any
hit aborts the run (or excludes the file, depending on the caller's policy).
A second-opinion scan with detect-secrets/trufflehog is planned, per
docs/data-governance/licensing-strategy.md section 4.
"""

from __future__ import annotations

import re

SECRET_RULES: dict[str, re.Pattern[str]] = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY"),
    "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github-token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack-token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "generic-credential": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|passwd|password|auth[_-]?token"""
        r"""|access[_-]?token)\b\s*[:=]\s*['"][A-Za-z0-9_\-/+=.]{16,}['"]"""
    ),
}


def scan_secrets(text: str) -> list[str]:
    """Return the names of every secret rule that matches ``text``."""
    return [name for name, pattern in SECRET_RULES.items() if pattern.search(text)]
