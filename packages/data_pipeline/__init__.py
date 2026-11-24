"""Corpus ingestion pipeline: license policy, secret scan, PII redaction, filtering, dedup."""

from .errors import IngestError, LicenseError, SecretFoundError
from .pipeline import ingest_source

__all__ = ["IngestError", "LicenseError", "SecretFoundError", "ingest_source"]
