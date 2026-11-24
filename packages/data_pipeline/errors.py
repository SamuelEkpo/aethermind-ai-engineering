"""Exceptions raised by the corpus ingestion pipeline.

Every one of these means the ingestion run was aborted and nothing was
written to the output corpus or manifest.
"""

from __future__ import annotations


class IngestError(Exception):
    """Base class for ingestion failures."""


class LicenseError(IngestError):
    """The source's license is missing, unrecognised, or not on the allowlist."""


class SecretFoundError(IngestError):
    """A possible secret/credential was found in the source."""
