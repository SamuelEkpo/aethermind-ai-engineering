"""Concrete backends: an OpenAI-compatible HTTP integration for open-weight models.

Works with any server exposing ``POST {base_url}/chat/completions`` (vLLM,
llama.cpp server, Ollama's OpenAI endpoint, hosted providers, ...).
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import BackendError, GenerationBackend


class LocalBackend:
    """Adapts the project's own InferenceEngine (or any GenerationBackend).

    Converts unexpected runtime failures into BackendError so the router can
    fall back instead of surfacing a 500.
    """

    def __init__(self, engine: GenerationBackend) -> None:
        self._engine = engine

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> dict[str, Any]:
        try:
            return self._engine.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"local model failed: {exc}") from exc

    def info(self) -> dict[str, Any]:
        return self._engine.info()


class OpenAICompatibleBackend:
    """Calls an external chat-completions endpoint; never loads weights locally."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,  # noqa: ARG002 - not part of the OpenAI schema
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            )
            response.raise_for_status()
            data = response.json()
            completion = data["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise BackendError(f"external backend request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise BackendError(f"external backend returned a malformed response: {exc}") from exc

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return {
            "prompt": prompt,
            "text": prompt + completion,
            "completion": completion,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
        }

    def info(self) -> dict[str, Any]:
        return {"base_url": self.base_url, "model": self.model, "kind": "openai-compatible"}

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self._client.close()
