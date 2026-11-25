# Corpus manifest: rich

- **Source:** https://github.com/Textualize/rich
- **Commit / snapshot:** 3a109ccdffe5b2d0e30684a304fbd388e99ce57b
- **Collection method:** packages.data_pipeline.cli over a local checkout
- **Created (UTC):** 2026-11-25
- **Output file:** datasets/processed/rich.txt
- **License(s):** MIT
- **License evidence:** LICENSE
- **License detection method:** text-heuristic. Independent confirmation with scancode-toolkit or licensee is still recommended.

## Size

- Files considered: 257
- Files kept: 221
- Output bytes: 1718543
- Token count: not yet tokenized

## Filters

- Extensions: .py, .md
- Maximum file size: 256 KB
- Skipped: binary, non-UTF-8, empty, generated, very long lines, exact duplicates

## Dropped files

| Reason | Count |
|---|---|
| empty | 2 |
| generated | 22 |
| long-lines | 12 |

## Dropped file paths

- empty: benchmarks/__init__.py, tests/__init__.py
- generated: .faq/FAQ.md, rich/_unicode_data/unicode10-0-0.py, rich/_unicode_data/unicode11-0-0.py, rich/_unicode_data/unicode12-0-0.py, rich/_unicode_data/unicode12-1-0.py, rich/_unicode_data/unicode13-0-0.py, rich/_unicode_data/unicode14-0-0.py, rich/_unicode_data/unicode15-0-0.py, rich/_unicode_data/unicode15-1-0.py, rich/_unicode_data/unicode16-0-0.py, rich/_unicode_data/unicode17-0-0.py, rich/_unicode_data/unicode4-1-0.py, rich/_unicode_data/unicode5-0-0.py, rich/_unicode_data/unicode5-1-0.py, rich/_unicode_data/unicode5-2-0.py, rich/_unicode_data/unicode6-0-0.py, rich/_unicode_data/unicode6-1-0.py, rich/_unicode_data/unicode6-2-0.py, rich/_unicode_data/unicode6-3-0.py, rich/_unicode_data/unicode7-0-0.py, rich/_unicode_data/unicode8-0-0.py, rich/_unicode_data/unicode9-0-0.py
- long-lines: tests/_card_render.py, tests/test_bar.py, tests/test_columns.py, tests/test_columns_align.py, tests/test_console.py, tests/test_live.py, tests/test_log.py, tests/test_markdown.py, tests/test_markdown_no_hyperlinks.py, tests/test_progress.py, tests/test_syntax.py, tests/test_table.py

## PII redactions

| Reason | Count |
|---|---|
| email | 4 |

## Secret scan

- Scanner: built-in pattern rules (private keys, cloud/API tokens, credentials)
- Policy on a hit: exclude
- Result: no flagged content in kept files

## Known limitations

- License detection reads license files only, not per-file headers.
- Only exact duplicates are removed; near-duplicates are not.

## Consumed by

- (none yet)
