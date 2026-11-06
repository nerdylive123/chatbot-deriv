"""The opt-in smoke scripts must skip cleanly when their prerequisites are absent.

Only the Together script is executed here (in a subprocess, with the key removed
from the environment), so no network call can happen.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_together_smoke_skips_without_key():
    env = {k: v for k, v in os.environ.items() if k != "TOGETHER_API_KEY"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "smoke_together.py")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SKIPPED: TOGETHER_API_KEY not set" in proc.stdout
