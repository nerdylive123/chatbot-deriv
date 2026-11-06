"""Thin wrapper around the optional Together SDK.

`together` is imported lazily, only when a request is made. The API key is held
privately and never appears in repr, exception text, or exception chains.
"""

from __future__ import annotations

import os
from typing import Any, Callable

ENV_VAR = "TOGETHER_API_KEY"
DEFAULT_TIMEOUT = 30.0


class TransportError(RuntimeError):
    """Provider call failed. Messages never contain credentials or provider text."""


def has_together_key() -> bool:
    return bool(os.environ.get(ENV_VAR, "").strip())


def _default_client_factory(api_key: str, timeout: float) -> Any:
    try:
        import together  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError:
        raise TransportError("together SDK is not installed (pip install -r requirements-together.txt)") from None
    return together.Together(api_key=api_key, timeout=timeout)


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


__all__ = ["ENV_VAR", "DEFAULT_TIMEOUT", "TransportError", "TogetherTransport", "has_together_key"]
