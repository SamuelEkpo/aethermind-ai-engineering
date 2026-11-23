"""
inference — loads trained AetherMind checkpoints and serves generation
requests (used by apps/api, independent of the API framework itself so it
can also be used from training/eval scripts or a CLI).

Planned contents (see roadmap Phase 4-5):
    - engine.py     model loading + generate() entrypoint
    - sampling.py   decoding strategies (greedy, top-k, top-p, temperature)
"""

__all__: list[str] = []
