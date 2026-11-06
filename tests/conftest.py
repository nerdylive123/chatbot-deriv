import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    """Runs before any fixture of any scope: the suite never sees a real key or loads .env."""
    os.environ.pop("TOGETHER_API_KEY", None)
    os.environ["SUPPORT_PIPELINE_NO_DOTENV"] = "1"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _offline_environment(monkeypatch):
    """Tests never use a real key: hide any shell key and stop .env from loading one.

    Subprocess tests inherit os.environ, so this also covers scripts run in a child process.
    """
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("SUPPORT_PIPELINE_NO_DOTENV", "1")

    import urllib.request

    def no_network(*args, **kwargs):
        raise AssertionError("tests must not make network calls")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
