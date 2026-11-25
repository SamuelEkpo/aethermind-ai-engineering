# Corpus manifest: click

- **Source:** https://github.com/pallets/click
- **Commit / snapshot:** 08b803cf32ceb2ae6a772c62d5da42757a897d44
- **Collection method:** packages.data_pipeline.cli over a local checkout
- **Created (UTC):** 2026-11-25
- **Output file:** datasets/processed/click.txt
- **License(s):** BSD-3-Clause
- **License evidence:** LICENSE.txt
- **License detection method:** text-heuristic. Independent confirmation with scancode-toolkit or licensee is still recommended.

## Size

- Files considered: 134
- Files kept: 128
- Output bytes: 1317903
- Token count: not yet tokenized

## Filters

- Extensions: .py, .md
- Maximum file size: 256 KB
- Skipped: binary, non-UTF-8, empty, generated, very long lines, exact duplicates

## Dropped files

| Reason | Count |
|---|---|
| empty | 5 |
| long-lines | 1 |

## Dropped file paths

- empty: examples/complex/complex/__init__.py, examples/complex/complex/commands/__init__.py, tests/test_exceptions/__init__.py, tests/test_types/__init__.py, tests/test_utils/__init__.py
- long-lines: docs/upgrade-guides.md

## PII redactions

| Reason | Count |
|---|---|
| email | 7 |

## Secret scan

- Scanner: built-in pattern rules (private keys, cloud/API tokens, credentials)
- Policy on a hit: exclude
- Result: no flagged content in kept files

## Known limitations

- License detection reads license files only, not per-file headers.
- Only exact duplicates are removed; near-duplicates are not.

## Consumed by

- (none yet)
