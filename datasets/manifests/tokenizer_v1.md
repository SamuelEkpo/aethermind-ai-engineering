# Tokenizer artifact manifest - v1 (seed run)

Reproducible record of the first trained tokenizer artifact. The artifact
itself (training/checkpoints/tokenizer/tokenizer.json) is intentionally
gitignored, like other training outputs - this manifest is the durable,
versioned record of how it was produced, per README Section 10 (Data
Governance: "Keep dataset manifests and processing records").

## Corpus

| File | Source | Bytes |
|---|---|---|
| aethermind_code_corpus.txt | This repo's own apps/, packages/, tests/, training/scripts/ (.py files), via scripts/build_code_corpus.py | 22587 |
| aethermind_docs_corpus.txt | This repo's README.md | 14645 |

Total: about 37.2 KB. This is a tiny seed corpus for exercising the
pipeline end-to-end, not a real training corpus. A production tokenizer
needs a much larger, more diverse sample of natural language and source
code before its merges generalize well.

## Training run

- Algorithm: byte-level BPE (packages/tokenizer.BPETokenizer.train)
- Requested vocab_size: 2000 (4 special tokens + 256 base bytes + 1740 learned merges)
- Achieved vocab_size: 2000 (target fully reached - the corpus had enough
  repetition to learn merges all the way to the cap)
- Command: python -m training.scripts.train_tokenizer --input-dir datasets/raw --vocab-size 2000 --output training/checkpoints/tokenizer/tokenizer.json

## Evaluation (python -m training.scripts.evaluate_tokenizer)

- Compression on 3 held-out samples (project description plus two code snippets): 4.49 bytes per token
- Lossless round-trip: verified on all held-out samples
- Top learned merges were dominated by common English bigrams (in, en, er,
  on, st, re, at, or) and repeated whitespace/indentation (the byte-level
  space symbol repeated), which is expected given the small, README-heavy
  corpus.

## Known limitations of this seed artifact

- Corpus is almost entirely this project's own README and source - it will
  not generalize to arbitrary code or prose.
- No held-out validation split was used during training; the evaluation
  samples above were written by hand, not sampled from unseen data.
- Should be retrained once datasets/raw/ has a real, licensed corpus
  (README Phase 3: "Select initial training corpus", still open).
