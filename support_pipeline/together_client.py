"""Thin client for Together's OpenAI-compatible chat completions API.

Uses only the standard library (urllib), so hosted mode needs no extra install.
The API key is held privately and never appears in repr, exception text, or
exception chains.
"""

from __future__ import annotations

import json
import os
import urllib.request
from types import SimpleNamespace
from typing import Any, Callable

ENV_VAR = "TOGETHER_API_KEY"
DEFAULT_TIMEOUT = 60.0
API_URL = "https://api.together.xyz/v1/chat/completions"


class TransportError(RuntimeError):
    """Provider call failed. Messages never contain credentials or provider text."""


def has_together_key() -> bool:
    return bool(os.environ.get(ENV_VAR, "").strip())


class _HttpCompletions:
    def __init__(self, api_key: str, timeout: float) -> None:
        self._api_key = api_key
        self._timeout = timeout

    def create(self, **payload: Any) -> Any:
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "support-ticket-pipeline",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310 - fixed https URL
            body = json.load(response)
        content = body["choices"][0]["message"]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def __repr__(self) -> str:
        return "_HttpCompletions(api_key=<redacted>)"


def _default_client_factory(api_key: str, timeout: float) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=_HttpCompletions(api_key, timeout)))


class TogetherTransport:
    """Sends one JSON-schema-constrained chat completion and returns the raw content string."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client_factory: Callable[[str, float], Any] | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(ENV_VAR, "")
        if not key.strip():
            raise TransportError(f"{ENV_VAR} is not set")
        self.__api_key = key.strip()
        self.timeout = float(timeout)
        self._client_factory = client_factory or _default_client_factory
        self._client: Any = None

    def __repr__(self) -> str:
        return f"TogetherTransport(timeout={self.timeout}, api_key=<redacted>)"

    __str__ = __repr__

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            if self._client is None:
                self._client = self._client_factory(self.__api_key, self.timeout)
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema},
                },
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
        except TransportError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider errors are opaque by design
            # Only the exception type is kept: provider messages may echo request headers.
            raise TransportError(f"Together request failed ({type(exc).__name__})") from None
        if not isinstance(content, str) or not content.strip():
            raise TransportError("Together returned empty content")
        return content


__all__ = ["API_URL", "ENV_VAR", "DEFAULT_TIMEOUT", "TransportError", "TogetherTransport", "has_together_key"]
