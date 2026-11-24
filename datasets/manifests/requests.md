# Corpus manifest: requests

- **Source:** https://github.com/psf/requests
- **Commit / snapshot:** 3c7baaa05ab4aa49cc819ce3a819d260de19c157
- **Collection method:** packages.data_pipeline.cli over a local checkout
- **Created (UTC):** 2026-11-24
- **Output file:** datasets/processed/requests.txt
- **License(s):** Apache-2.0
- **License evidence:** LICENSE
- **License detection method:** text-heuristic. Independent confirmation with scancode-toolkit or licensee is still recommended.
- **Directories excluded (nested license not allowed):** ext

## Size

- Files considered: 50
- Files kept: 48
- Output bytes: 476432
- Token count: not yet tokenized

## Filters

- Extensions: .py, .md
- Maximum file size: 256 KB
- Skipped: binary, non-UTF-8, empty, generated, very long lines, exact duplicates

## Dropped files

| Reason | Count |
|---|---|
| empty | 1 |
| generated | 1 |

## Dropped file paths

- empty: tests/testserver/__init__.py
- generated: docs/conf.py

## PII redactions

| Reason | Count |
|---|---|
| email | 36 |

## Secret scan

- Scanner: built-in pattern rules (private keys, cloud/API tokens, credentials)
- Policy on a hit: exclude
- Result: no flagged content in kept files

## Known limitations

- License detection reads license files only, not per-file headers.
- Only exact duplicates are removed; near-duplicates are not.

## Consumed by

- (none yet)
