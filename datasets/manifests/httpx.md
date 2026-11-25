# Corpus manifest: httpx

- **Source:** https://github.com/encode/httpx
- **Commit / snapshot:** 7cff5f79bc78649bbcd3fd9aa7a74c86e0f63016
- **Collection method:** packages.data_pipeline.cli over a local checkout
- **Created (UTC):** 2026-11-25
- **Output file:** datasets/processed/httpx.txt
- **License(s):** BSD-3-Clause
- **License evidence:** LICENSE.md
- **License detection method:** text-heuristic. Independent confirmation with scancode-toolkit or licensee is still recommended.

## Size

- Files considered: 89
- Files kept: 86
- Output bytes: 768648
- Token count: not yet tokenized

## Filters

- Extensions: .py, .md
- Maximum file size: 256 KB
- Skipped: binary, non-UTF-8, empty, generated, very long lines, exact duplicates

## Dropped files

| Reason | Count |
|---|---|
| empty | 3 |

## Dropped file paths

- empty: tests/__init__.py, tests/client/__init__.py, tests/models/__init__.py

## PII redactions

| Reason | Count |
|---|---|
| email | 42 |

## Secret scan

- Scanner: built-in pattern rules (private keys, cloud/API tokens, credentials)
- Policy on a hit: exclude
- Result: no flagged content in kept files

## Known limitations

- License detection reads license files only, not per-file headers.
- Only exact duplicates are removed; near-duplicates are not.

## Consumed by

- (none yet)
