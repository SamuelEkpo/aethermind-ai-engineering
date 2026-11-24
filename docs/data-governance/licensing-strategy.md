# Data and Licensing Strategy

Status document for README Section 10 ("Data Governance") and Phase 1
("Finalize initial model specification" / "Document data and licensing
strategy"). This is the policy new corpus additions must satisfy before
they land in `datasets/raw/`; it is deliberately stricter than "whatever is
easiest to scrape," because licensing mistakes made now (training on
incompatible-license code, embedding PII in model weights) are expensive or
impossible to undo later.

## 1. What exists today

| Corpus file | Provenance | License |
|---|---|---|
| `datasets/raw/aethermind_code_corpus.txt` | This repo's own `apps/`, `packages/`, `tests/`, `training/scripts/` (via `scripts/build_code_corpus.py`) | Proprietary (this project — see README Section 17) |
| `datasets/raw/aethermind_docs_corpus.txt` | This repo's own `README.md` | Proprietary (this project) |

Both are self-authored seed material (see `datasets/manifests/tokenizer_v1.md`
for the full provenance record of the tokenizer trained on them). There is
currently **no third-party or externally-sourced text or code** in the
repository, which sidesteps licensing risk entirely for now — but it also
means the tokenizer and any future model trained on this alone will not
generalize beyond this project's own style.

## 2. License policy for future corpus additions

Before any external file is added under `datasets/raw/`, its license must
be identified and must fall into one of the allowed categories below. If a
source's license cannot be determined with confidence, it is excluded —
"probably fine" is not a basis for ingestion.

**Allowed (code):**
- MIT, BSD (2/3-clause), Apache-2.0, ISC, ... (permissive, no copyleft
  obligations, no attribution requirements that conflict with a
  closed/proprietary model output).

**Allowed (text/docs):**
- CC0, CC-BY (with attribution tracked in the manifest), public domain, or
  text the project itself has the rights to (e.g. its own documentation).

**Excluded:**
- GPL / AGPL / LGPL / any copyleft license. Training on copyleft code risks
  the position that model outputs are a derivative work, which is legally
  unsettled and not a risk this project should take on while it is a
  closed, proprietary system (README Section 17).
- "All rights reserved" source with no explicit license grant.
- Any dataset whose license is unknown, unverifiable, or inconsistent
  across files (e.g. an archive with mixed per-file licenses that haven't
  been individually checked).
- Any source whose terms of service prohibit the kind of bulk collection
  used to obtain it, independent of the content's own license (e.g.
  scraping a code-hosting site in violation of its ToS, even if the code
  itself is MIT-licensed).

**Tooling note:** once a real external corpus is being considered, license
detection should not be done by eye — a tool such as `scancode-toolkit` or
`licensee` should be run over candidate sources and its output kept
alongside the manifest as evidence of the determination, not just a one-line
claim in prose.

## 3. Provenance tracking

Every corpus addition — self-authored or external — gets a manifest file
under `datasets/manifests/`, following the format already established by
`tokenizer_v1.md`:

- Source (repo/URL, commit or snapshot date, collection method/script).
- License, with evidence (a `LICENSE` file reference, SPDX identifier, or
  scan tool output).
- Size (bytes / file count / token count once tokenized).
- Known limitations (how representative or skewed the sample is).
- Which pipeline run(s) consumed it (tokenizer version, training run).

This mirrors README Section 10's "Keep dataset manifests and processing
records" and Section 12's requirement that evaluation stay separate from
training data — a manifest is what later lets someone (including future-us)
check whether an eval set accidentally leaked into training.

## 4. PII and secrets handling

Even self-authored or permissively-licensed source can contain things that
should never reach a trained model: embedded credentials, API keys, email
addresses, or other personal data accidentally committed somewhere in
history. Before any new corpus is tokenized or trained on:

1. Run a secret scanner (e.g. `detect-secrets` or `trufflehog`) over the raw
   files and fail the ingestion if anything is flagged, rather than
   manually eyeballing the diff.
2. For natural-language corpora specifically, do a pass for obvious PII
   patterns (emails, phone numbers) in addition to the secret scan, since
   secret scanners are tuned for credentials, not personal data.
3. Record in the manifest that this scan was run and what (if anything) was
   stripped.

This is not yet automated (no corpus has needed it so far, since the seed
corpus is just this repo's own non-secret source) — it is the policy to
apply the first time a real external corpus is brought in, and should
become a check in `scripts/` (parallel to `build_code_corpus.py`) rather
than a manual step once that happens.

## 5. Train / validation / test separation

`training/scripts/train_lm.py` currently does a **chronological** split
(the last `--val-ratio` fraction of the concatenated, tokenized corpus is
held out as validation) — simple and deterministic, but not yet a true
held-out test set distinct from the validation set used to pick "best"
checkpoints. Before Phase 5's "Evaluate model performance" and "Document
experimental results" are meaningful at real scale, this should be extended
to a three-way split (train / val / test) with the test portion never
touched by checkpoint-selection logic, consistent with README Section 12's
requirement that evaluation data be held out to avoid misleading results.

## 6. Open decisions (not yet resolved)

- Where the first real external corpus will come from is still undecided —
  this document defines the *filter* any candidate source must pass, not a
  specific source to use. That decision is Phase 3's "Select initial
  training corpus," still open.
- Deduplication strategy for a larger corpus (exact + near-duplicate removal)
  is not yet designed.
- Whether a dataset "freeze" / versioning scheme (e.g. content-addressed
  snapshots) is needed once corpora stop being small enough to regenerate
  on demand from a single script.
