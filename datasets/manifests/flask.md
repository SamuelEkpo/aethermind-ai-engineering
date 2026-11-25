# Corpus manifest: flask

- **Source:** https://github.com/pallets/flask
- **Commit / snapshot:** 49943b5a94413a530761350f04e896f76e8d620e
- **Collection method:** packages.data_pipeline.cli over a local checkout
- **Created (UTC):** 2026-11-25
- **Output file:** datasets/processed/flask.txt
- **License(s):** BSD-3-Clause
- **License evidence:** LICENSE.txt
- **License detection method:** text-heuristic. Independent confirmation with scancode-toolkit or licensee is still recommended.

## Size

- Files considered: 89
- Files kept: 86
- Output bytes: 596011
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

- empty: tests/test_apps/blueprintapp/apps/__init__.py, tests/test_apps/cliapp/__init__.py, tests/test_apps/cliapp/inner1/inner2/__init__.py

## PII redactions

No PII patterns were redacted.

## Secret scan

- Scanner: built-in pattern rules (private keys, cloud/API tokens, credentials)
- Policy on a hit: exclude
- Result: no flagged content in kept files

## Known limitations

- License detection reads license files only, not per-file headers.
- Only exact duplicates are removed; near-duplicates are not.

## Consumed by

- (none yet)
