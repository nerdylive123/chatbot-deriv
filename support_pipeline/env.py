"""Load KEY=VALUE pairs from a local .env file into os.environ.

Standard library only. Existing environment variables always win, values are
never printed or logged, and a missing file is not an error. Setting
SUPPORT_PIPELINE_NO_DOTENV=1 disables loading (the test suite does this).
"""

from __future__ import annotations

import os
from pathlib import Path

from support_pipeline.data import REPO_ROOT

DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def load_dotenv(path: Path | str = DEFAULT_ENV_FILE) -> list[str]:
    """Set variables from ``path`` that are not already set. Returns the names set (never values)."""
    if os.environ.get("SUPPORT_PIPELINE_NO_DOTENV"):
        return []
    file = Path(path)
    if not file.is_file():
        return []
    loaded: list[str] = []
    for raw in file.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


__all__ = ["DEFAULT_ENV_FILE", "load_dotenv"]
