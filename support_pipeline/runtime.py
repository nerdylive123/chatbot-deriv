"""Local model residency: at most one optional local model loaded at a time.

torch is imported lazily and only to pick a device / release CUDA cache; any
failure falls back to "cpu". Nothing here depends on torch being installed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

ModelFactory = Callable[[str], Any]  # factory(device) -> loaded model


def detect_device() -> str:
    """Return "cuda" if torch is importable and reports CUDA, else "cpu"."""
    try:
        import torch  # noqa: PLC0415 - deliberately lazy

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # ImportError, broken drivers, anything else
        return "cpu"


class LocalModelPool:
    """Loads models on demand from injected factories, keeping one resident."""

    def __init__(self, factories: Mapping[str, ModelFactory], device: str | None = None) -> None:
        self._factories = dict(factories)
        self.device = device if device is not None else detect_device()
        self.active: str | None = None
        self.resident: Any = None

    def acquire(self, name: str) -> Any:
        if name not in self._factories:
            raise KeyError(f"no factory registered for model {name!r}")
        if self.active == name and self.resident is not None:
            return self.resident
        self.release()
        self.resident = self._factories[name](self.device)
        self.active = name
        return self.resident

    def release(self) -> None:
        if self.active is None and self.resident is None:
            return
        self.resident = None
        self.active = None
        if self.device == "cuda":
            try:
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()
            except Exception:
                pass


__all__ = ["LocalModelPool", "ModelFactory", "detect_device"]
