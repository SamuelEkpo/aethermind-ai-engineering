"""
inference — serving and generation utilities for AetherMind models.

Keeps model loading / generation separate from architecture (model_core)
and from the HTTP layer (apps/api).
"""

from packages.inference.engine import InferenceEngine

__all__ = ["InferenceEngine"]
