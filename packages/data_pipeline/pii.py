"""Basic PII redaction for corpus ingestion (emails and phone numbers).

Redaction replaces matches with placeholders instead of dropping the file,
because author emails in source headers are near-universal. Counts are
recorded in the manifest. Known limitation: an expression such as
``x@w.shape`` (matrix multiply on an attribute) looks like an email and is
redacted too.
"""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?![\w.])"
)


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails and phone numbers; return the new text and per-kind counts."""
    counts: dict[str, int] = {}
    text, n_email = EMAIL_RE.subn("<EMAIL>", text)
    if n_email:
        counts["email"] = n_email
    text, n_phone = PHONE_RE.subn("<PHONE>", text)
    if n_phone:
        counts["phone"] = n_phone
    return text, counts
