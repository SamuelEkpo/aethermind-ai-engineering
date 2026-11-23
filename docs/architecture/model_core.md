# model_core architecture

## Purpose

`packages/model_core` owns the custom AetherMind decoder-only Transformer.
It is deliberately free of training-loop and HTTP concerns so the architecture
can be unit-tested and benchmarked in isolation.

## Layout

| Module | Responsibility |
|--------|----------------|
| `config.py` | Frozen `ModelConfig` + presets (`TINY`, `SMALL`, `BASE`) |
| `embeddings.py` | Token embedding, learned absolute positions, dropout |
| `attention.py` | Causal multi-head self-attention |
| `transformer.py` | Blocks, full `AetherMindLM`, generate, save/load |

## Design choices

- **Pre-norm** LayerNorm before attention/FFN for stable training from scratch.
- **Weight tying** between token embeddings and the LM head.
- **GPT-2 residual scaling** on projection layers (`std = 0.02 / sqrt(2L)`).
- **Causal mask** registered as a buffer; padding mask optional.
- **Pure PyTorch** first — correctness over kernel fusion.

## Presets

| Preset | ~Params | Intended use |
|--------|---------|--------------|
| `TINY` | ~0.7M | Unit tests, CPU smoke runs |
| `SMALL` | ~15–20M | Laptop GPU experiments |
| `BASE` | ~100M+ | Later production target |

## Training entrypoint

```bash
python -m training.scripts.train_lm --config tiny --steps 500
```

## Inference

```bash
python -m training.scripts.generate --prompt "def " --max-new-tokens 40
```

Or via API after `uvicorn apps.api.main:app --reload`:

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "def ", "max_new_tokens": 40}'
```
