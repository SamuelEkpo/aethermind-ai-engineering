"""
model_router — decides, per request, whether to serve from the custom
AetherMind model, an integrated open-weight model, or a tool/retrieval-only
path. This is the seam described in the README's "Hybrid Model Integration"
section: it keeps evaluation of the custom model separable from whatever
else is backing a given response.

Planned contents (see roadmap Phase 7):
    - router.py     routing policy / model selection logic
    - registry.py   registered model backends and their capabilities
"""

__all__: list[str] = []
