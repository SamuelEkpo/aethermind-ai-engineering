"""Unit tests for the model router and its backends (no network, no torch)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from packages.model_router import (
    BackendError,
    BackendKind,
    LocalBackend,
    ModelRouter,
    NoBackendAvailableError,
    OpenAICompatibleBackend,
    RoutingPolicy,
    UnknownBackendError,
)


class FakeBackend:
    def __init__(self, label: str, *, fail: Exception | None = None) -> None:
        self.label = label
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "max_new_tokens": max_new_tokens, "top_k": top_k})
        if self.fail is not None:
            raise self.fail
        return {
            "prompt": prompt,
            "text": prompt + self.label,
            "completion": self.label,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }

    def info(self) -> dict[str, Any]:
        return {"label": self.label}


def _router() -> tuple[ModelRouter, FakeBackend, FakeBackend]:
    custom = FakeBackend("custom")
    external = FakeBackend("external")
    router = ModelRouter()
    router.register("aethermind", custom, kind=BackendKind.CUSTOM)
    router.register("open-weight", external, kind=BackendKind.EXTERNAL)
    return router, custom, external


def test_without_policy_uses_first_registered_backend() -> None:
    router, custom, external = _router()
    out = router.generate("hi")
    assert out["model"] == "aethermind"
    assert out["backend_kind"] == "custom"
    assert out["fallback_used"] is False
    assert len(custom.calls) == 1
    assert external.calls == []


def test_empty_router_raises() -> None:
    with pytest.raises(NoBackendAvailableError):
        ModelRouter().generate("hi")


def test_explicit_model_is_used_and_tagged_external() -> None:
    router, _, external = _router()
    out = router.generate("hi", model="open-weight")
    assert out["model"] == "open-weight"
    assert out["backend_kind"] == "external"
    assert len(external.calls) == 1


def test_unknown_explicit_model_raises() -> None:
    router, _, _ = _router()
    with pytest.raises(UnknownBackendError):
        router.generate("hi", model="nope")


def test_explicit_model_does_not_fall_back() -> None:
    router = ModelRouter(RoutingPolicy(default="a", fallbacks=("b",)))
    router.register("a", FakeBackend("a", fail=BackendError("down")))
    router.register("b", FakeBackend("b"))
    with pytest.raises(NoBackendAvailableError) as info:
        router.generate("hi", model="a")
    assert info.value.failures == {"a": "down"}


def test_task_route_overrides_default() -> None:
    router, custom, external = _router()
    router.set_policy(RoutingPolicy(default="aethermind", task_routes={"explain": "open-weight"}))
    assert router.generate("hi", task="explain")["model"] == "open-weight"
    assert router.generate("hi", task="complete")["model"] == "aethermind"
    assert len(custom.calls) == 1
    assert len(external.calls) == 1


def test_fallback_on_backend_error_records_provenance() -> None:
    router = ModelRouter(RoutingPolicy(default="a", fallbacks=("b",)))
    router.register("a", FakeBackend("a", fail=BackendError("boom")))
    router.register("b", FakeBackend("b"), kind=BackendKind.EXTERNAL)
    out = router.generate("hi")
    assert out["model"] == "b"
    assert out["fallback_used"] is True
    assert out["failed_backends"] == ["a"]


def test_all_backends_failing_raises_with_failures() -> None:
    router = ModelRouter(RoutingPolicy(default="a", fallbacks=("b",)))
    router.register("a", FakeBackend("a", fail=BackendError("x")))
    router.register("b", FakeBackend("b", fail=BackendError("y")))
    with pytest.raises(NoBackendAvailableError) as info:
        router.generate("hi")
    assert info.value.failures == {"a": "x", "b": "y"}


def test_non_backend_errors_are_not_swallowed() -> None:
    router = ModelRouter(RoutingPolicy(default="a", fallbacks=("b",)))
    router.register("a", FakeBackend("a", fail=RuntimeError("bug")))
    router.register("b", FakeBackend("b"))
    with pytest.raises(RuntimeError):
        router.generate("hi")


def test_candidates_dedupes_policy_order() -> None:
    router = ModelRouter(RoutingPolicy(default="a", fallbacks=("a", "b")))
    router.register("a", FakeBackend("a"))
    router.register("b", FakeBackend("b"))
    assert router.candidates() == ["a", "b"]


def test_register_validates_names() -> None:
    router = ModelRouter()
    router.register("a", FakeBackend("a"))
    with pytest.raises(ValueError):
        router.register("a", FakeBackend("a"))
    with pytest.raises(ValueError):
        router.register("", FakeBackend("x"))


def test_set_policy_rejects_unknown_backend() -> None:
    router = ModelRouter()
    router.register("a", FakeBackend("a"))
    with pytest.raises(UnknownBackendError):
        router.set_policy(RoutingPolicy(default="a", fallbacks=("ghost",)))


def test_describe_lists_backends_with_kind() -> None:
    router, _, _ = _router()
    described = {d["name"]: d for d in router.describe()}
    assert described["aethermind"]["kind"] == "custom"
    assert described["open-weight"]["kind"] == "external"
    assert described["open-weight"]["label"] == "external"


def test_describe_router_fields_win_over_backend_info() -> None:
    class Colliding(FakeBackend):
        def info(self) -> dict[str, Any]:
            return {"kind": "openai-compatible", "name": "impostor", "model": "m1"}

    router = ModelRouter()
    router.register("ext", Colliding("x"), kind=BackendKind.EXTERNAL)
    (entry,) = router.describe()
    assert entry["kind"] == "external"
    assert entry["name"] == "ext"
    assert entry["model"] == "m1"


# ----------------------------------------------------------------------
# LocalBackend
# ----------------------------------------------------------------------


def test_local_backend_wraps_unexpected_errors() -> None:
    backend = LocalBackend(FakeBackend("x", fail=RuntimeError("cuda oom")))
    with pytest.raises(BackendError, match="cuda oom"):
        backend.generate("hi")


def test_local_backend_passthrough_and_info() -> None:
    inner = FakeBackend("x")
    backend = LocalBackend(inner)
    assert backend.generate("hi", max_new_tokens=5)["completion"] == "x"
    assert inner.calls[0]["max_new_tokens"] == 5
    assert backend.info() == {"label": "x"}


def test_local_backend_preserves_backend_error() -> None:
    backend = LocalBackend(FakeBackend("x", fail=BackendError("already wrapped")))
    with pytest.raises(BackendError, match="already wrapped"):
        backend.generate("hi")


# ----------------------------------------------------------------------
# OpenAICompatibleBackend
# ----------------------------------------------------------------------


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_backend_success_and_request_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["json"] = request.read()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": " world"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    backend = OpenAICompatibleBackend(
        "http://llm.local/v1/", "m1", api_key="k", client=_client(handler)
    )
    out = backend.generate("hello", max_new_tokens=9, temperature=0.3)
    assert seen["url"] == "http://llm.local/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert b'"max_tokens":9' in seen["json"].replace(b" ", b"")
    assert out["completion"] == " world"
    assert out["text"] == "hello world"
    assert (out["prompt_tokens"], out["completion_tokens"], out["total_tokens"]) == (3, 2, 5)
    assert backend.info()["model"] == "m1"


def test_openai_backend_http_error_becomes_backend_error() -> None:
    backend = OpenAICompatibleBackend(
        "http://x", "m", client=_client(lambda r: httpx.Response(500, json={}))
    )
    with pytest.raises(BackendError, match="request failed"):
        backend.generate("hi")


def test_openai_backend_malformed_body_becomes_backend_error() -> None:
    backend = OpenAICompatibleBackend(
        "http://x", "m", client=_client(lambda r: httpx.Response(200, json={"oops": 1}))
    )
    with pytest.raises(BackendError, match="malformed"):
        backend.generate("hi")


def test_openai_backend_without_api_key_sends_no_auth_header() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    out = OpenAICompatibleBackend("http://x", "m", client=_client(handler)).generate("hi")
    assert captured["auth"] is None
    assert out["total_tokens"] == 0
