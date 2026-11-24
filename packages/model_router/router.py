"""Model router: pick a backend per request and fall back on failure."""

from __future__ import annotations

from typing import Any

from .base import (
    BackendError,
    BackendKind,
    GenerationBackend,
    NoBackendAvailableError,
    RegisteredBackend,
    RoutingPolicy,
    UnknownBackendError,
)


class ModelRouter:
    """Registry of generation backends plus a routing policy.

    Resolution order for a request:
      1. an explicit ``model=`` name (no fallback: the caller asked for it),
      2. the policy's ``task_routes[task]`` if a task is given,
      3. the policy default.
    Cases 2 and 3 then fall through the policy's ``fallbacks`` on BackendError.
    """

    def __init__(self, policy: RoutingPolicy | None = None) -> None:
        self._backends: dict[str, RegisteredBackend] = {}
        self._policy = policy

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        backend: GenerationBackend,
        *,
        kind: BackendKind = BackendKind.CUSTOM,
    ) -> None:
        if not name:
            raise ValueError("backend name must be non-empty")
        if name in self._backends:
            raise ValueError(f"backend already registered: {name}")
        self._backends[name] = RegisteredBackend(name=name, backend=backend, kind=kind)

    def set_policy(self, policy: RoutingPolicy) -> None:
        for name in (policy.default, *policy.fallbacks, *policy.task_routes.values()):
            self._require(name)
        self._policy = policy

    @property
    def names(self) -> list[str]:
        return list(self._backends)

    def _require(self, name: str) -> RegisteredBackend:
        try:
            return self._backends[name]
        except KeyError:
            raise UnknownBackendError(f"unknown backend: {name}") from None

    # -- routing ------------------------------------------------------------

    def candidates(self, *, model: str | None = None, task: str | None = None) -> list[str]:
        """Ordered backend names to try for a request."""
        if model is not None:
            return [self._require(model).name]

        if self._policy is None:
            if not self._backends:
                raise NoBackendAvailableError("no backends registered")
            return [next(iter(self._backends))]

        primary = self._policy.default
        if task is not None and task in self._policy.task_routes:
            primary = self._policy.task_routes[task]
        return list(dict.fromkeys([primary, *self._policy.fallbacks]))

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        task: str | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> dict[str, Any]:
        """Generate via the first healthy candidate; tag the result with provenance."""
        failures: dict[str, str] = {}
        for name in self.candidates(model=model, task=task):
            entry = self._require(name)
            try:
                result = entry.backend.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
            except BackendError as exc:
                failures[name] = str(exc)
                continue
            return {
                **result,
                "model": name,
                "backend_kind": entry.kind.value,
                "fallback_used": bool(failures),
                "failed_backends": list(failures),
            }
        raise NoBackendAvailableError(
            "all candidate backends failed: " + "; ".join(f"{k}: {v}" for k, v in failures.items()),
            failures,
        )

    def describe(self) -> list[dict[str, Any]]:
        """Static description of registered backends (for a /v1/models endpoint)."""
        return [
            # Router-owned fields go last so a backend's own info() can never override them.
            {**e.backend.info(), "name": e.name, "kind": e.kind.value}
            for e in self._backends.values()
        ]
