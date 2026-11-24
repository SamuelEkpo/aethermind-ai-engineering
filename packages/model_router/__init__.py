"""Model routing across the custom AetherMind model and optional external models."""

from .backends import LocalBackend, OpenAICompatibleBackend
from .base import (
    BackendError,
    BackendKind,
    GenerationBackend,
    NoBackendAvailableError,
    RouterError,
    RoutingPolicy,
    UnknownBackendError,
)
from .router import ModelRouter

__all__ = [
    "BackendError",
    "BackendKind",
    "GenerationBackend",
    "LocalBackend",
    "ModelRouter",
    "NoBackendAvailableError",
    "OpenAICompatibleBackend",
    "RouterError",
    "RoutingPolicy",
    "UnknownBackendError",
]
