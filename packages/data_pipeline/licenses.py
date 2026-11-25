"""License detection and policy enforcement for external corpus sources.

Detection is a text heuristic over LICENSE/COPYING files. It is deliberately
conservative: anything it cannot positively identify is rejected. It is not a
substitute for a dedicated scanner (scancode-toolkit, licensee); run one over
any new source as independent evidence, as
docs/data-governance/licensing-strategy.md requires.

Per-file SPDX headers are not inspected yet; only license files are.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import LicenseError
from .filters import SKIP_DIRS

ALLOWED_SPDX: frozenset[str] = frozenset(
    {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "CC0-1.0"}
)

_LICENSE_FILE_RE = re.compile(r"^(?:licen[sc]e|copying)(?:[._-].*)?$", re.IGNORECASE)


@dataclass(frozen=True)
class LicenseResult:
    """Outcome of checking a source tree's license."""

    spdx_ids: tuple[str, ...]
    evidence: tuple[str, ...]
    excluded_dirs: tuple[str, ...]
    method: str = "text-heuristic"

    @property
    def spdx(self) -> str:
        return ", ".join(self.spdx_ids)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def detect_license(text: str) -> str | None:
    """Return an SPDX-style id for known license text, else None.

    Copyleft and other disallowed licenses are returned by name so the caller
    can report them precisely; only ``ALLOWED_SPDX`` ids pass the policy.
    """
    t = _normalise(text)
    if "gnu affero general public license" in t:
        return "AGPL"
    if "gnu lesser general public license" in t:
        return "LGPL"
    if "gnu general public license" in t:
        return "GPL"
    if "mozilla public license" in t:
        return "MPL"
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "cc0 1.0 universal" in t:
        return "CC0-1.0"
    if "permission is hereby granted, free of charge" in t:
        return "MIT"
    if (
        "permission to use, copy, modify, and/or distribute this software" in t
        or "permission to use, copy, modify, and distribute this software" in t
    ):
        return "ISC"
    if "redistribution and use in source and binary forms" in t:
        if "all advertising materials" in t:
            return "BSD-4-Clause"
        if "neither the name of" in t:
            return "BSD-3-Clause"
        return "BSD-2-Clause"
    return None


# Documentation trees often hold pages named license.md/license.rst that merely
# include or describe the real license. An unrecognised file there is a docs page,
# not a competing license. A recognised but disallowed license text still excludes
# its directory, wherever it is.
_DOC_DIRS: frozenset[str] = frozenset({"docs", "doc", "documentation"})


def _in_docs_dir(rel: Path) -> bool:
    return any(part.lower() in _DOC_DIRS for part in rel.parts[:-1])


def _license_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        found.extend(
            Path(dirpath) / name for name in sorted(filenames) if _LICENSE_FILE_RE.match(name)
        )
    return found


def check_source_license(root: Path) -> LicenseResult:
    """Enforce the license allowlist on a source tree.

    Root-level license files must all be recognised and allowed (at least one
    must exist). A nested license file that is not allowed excludes its whole
    directory from ingestion instead of failing the run.
    """
    root = root.resolve()
    if not root.is_dir():
        raise LicenseError(f"Source is not a directory: {root}")

    ids: set[str] = set()
    evidence: list[str] = []
    excluded: set[str] = set()
    for path in _license_files(root):
        rel = path.relative_to(root)
        spdx = detect_license(path.read_text(encoding="utf-8", errors="replace"))
        if path.parent == root:
            if spdx is None:
                raise LicenseError(f"Unrecognised license text in {rel.as_posix()}")
            if spdx not in ALLOWED_SPDX:
                raise LicenseError(
                    f"License {spdx} in {rel.as_posix()} is not on the allowlist "
                    f"({', '.join(sorted(ALLOWED_SPDX))})"
                )
            ids.add(spdx)
            evidence.append(rel.as_posix())
        elif spdx not in ALLOWED_SPDX and not (spdx is None and _in_docs_dir(rel)):
            excluded.add(path.parent.relative_to(root).as_posix())

    if not ids:
        raise LicenseError("No LICENSE/COPYING file found at the source root")
    return LicenseResult(
        spdx_ids=tuple(sorted(ids)),
        evidence=tuple(evidence),
        excluded_dirs=tuple(sorted(excluded)),
    )
