# Hardware Assessment

Date: 2026-11-23

Measured on the primary local development machine with torch, wmic and Windows, not from vendor specifications.

## Local machine

| Item | Value |
|---|---|
| OS | Windows 10 (10.0.19045) |
| CPU | Intel Core i5-1035G1 @ 1.00 GHz |
| Cores / threads | 4 cores / 8 logical processors |
| RAM | about 7.8 GB (8,361,132,032 bytes) |
| GPU | Intel UHD Graphics (integrated, shared memory); no CUDA device |
| Disk (C:) | about 270 GB free of 390 GB |
| PyTorch | 2.15.0+cpu (cuda_available = False) |
| PyTorch CPU threads (default) | 4 |

## Baseline measurements

| Metric | Value |
|---|---|
| Tokens evaluated | 25,600 |
| Loss | 6.4775 |
| Perplexity | 650.31 |

Measured on the seed corpus with the training.scripts.evaluate_lm module. This is the reference point for later training runs.

## Implications

- Local work is limited to CPU: unit and integration tests, API development, tokenizer work and training of very small models.
- GPU-intensive pretraining and larger fine-tuning runs need cloud GPU resources, as described in the README compute strategy.
- With about 7.8 GB of RAM shared with the OS and editor, dataset loading should be streamed or memory-mapped rather than held in memory.

## Notes

- The repository is located inside a OneDrive folder. Syncing of .venv, caches and checkpoints may slow I/O or lock files. This has not been verified as a problem.
- Cloud provider, GPU type and budget are not yet decided.
