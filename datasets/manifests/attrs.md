# Corpus manifest: attrs

- **Source:** https://github.com/python-attrs/attrs
- **Commit / snapshot:** 319c53cc178afe980362c1bd59d8109019e344bd
- **Collection method:** packages.data_pipeline.cli over a local checkout
- **Created (UTC):** 2026-11-25
- **Output file:** datasets/processed/attrs.txt
- **License(s):** MIT
- **License evidence:** LICENSE
- **License detection method:** text-heuristic. Independent confirmation with scancode-toolkit or licensee is still recommended.

## Size

- Files considered: 88
- Files kept: 87
- Output bytes: 784579
- Token count: not yet tokenized

## Filters

- Extensions: .py, .md
- Maximum file size: 256 KB
- Skipped: binary, non-UTF-8, empty, generated, very long lines, exact duplicates

## Dropped files

| Reason | Count |
|---|---|
| duplicate | 1 |

## Dropped file paths

- duplicate: typing_tests/__init__.py

## PII redactions

| Reason | Count |
|---|---|
| email | 10 |

## Secret scan

- Scanner: built-in pattern rules (private keys, cloud/API tokens, credentials)
- Policy on a hit: exclude
- Result: no flagged content in kept files

## Known limitations

- License detection reads license files only, not per-file headers.
- Only exact duplicates are removed; near-duplicates are not.

## Consumed by

- (none yet)
