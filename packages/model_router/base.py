"""Shared types for the model router."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class BackendKind(StrEnum):
    """Whether a backend is the project's own model or an external integration.

    The README requires the custom model to be evaluable independently of
    external integrations, so every result carries its backend kind.
    """

    CUSTOM = "custom"
    EXTERNAL = "external"


class RouterError(Exception):
    """Base class for router errors."""


class UnknownBackendError(RouterError):
    """Requested a backend name that is not registered."""


class BackendError(RouterError):
    """A backend failed to produce a completion."""


class NoBackendAvailableError(RouterError):
    """Every candidate backend failed (or none are registered)."""

    def __init__(self, message: str, failures: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.failures: dict[str, str] = failures or {}


@runtime_checkable
class GenerationBackend(Protocol):
    """Anything that can turn a prompt into a completion.

    ``InferenceEngine`` satisfies this protocol as-is.
    """

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> dict[str, Any]: ...

    def info(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RoutingPolicy:
    """Declarative routing rules.

    Attributes:
        default: backend used when nothing more specific applies.
        fallbacks: tried in order if the chosen backend fails.
        task_routes: maps a task label (e.g. ``"explain"``) to a backend name.
    """

    default: str
    fallbacks: tuple[str, ...] = ()
    task_routes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisteredBackend:
    name: str
    backend: GenerationBackend
    kind: BackendKind
