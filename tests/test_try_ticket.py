"""scripts/try_ticket.py runs ad-hoc ticket text through the offline pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "try_ticket.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "TOGETHER_API_KEY"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


def test_single_message_prints_result_reasons_and_trace():
    proc = _run("Can you tell me which stock will go up tomorrow?")
    assert proc.returncode == 0, proc.stderr
    assert '"intent": "trading_advice_request"' in proc.stdout
    assert "advice_request" in proc.stdout
    assert "Escalation reasons" in proc.stdout
    for stage in ("CLASSIFY_INTENT", "RETRIEVE_CONTEXT", "GENERATE_REPLY", "VALIDATE_OUTPUT"):
        assert stage in proc.stdout


def test_json_flag_emits_parseable_ticket_result():
    proc = _run("--json", "My card deposit has not arrived in my balance")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert set(data["result"]) == {
        "ticket_id", "intent", "retrieved_articles", "reply_draft", "grounded",
        "confidence", "needs_human_escalation", "safety_flags",
    }
    assert data["result"]["intent"] == "deposit_issue"
    assert "escalation_reasons" in data and "traces" in data


def test_samples_flag_runs_every_supplied_ticket():
    proc = _run("--samples")
    assert proc.returncode == 0, proc.stderr
    for ticket_id in ("T1", "T2", "T3", "T4", "T5"):
        assert ticket_id in proc.stdout


def test_file_flag_runs_custom_ticket_file(tmp_path):
    tickets = tmp_path / "mine.json"
    tickets.write_text(json.dumps([{"id": "Q1", "message": "I forgot my password", "language": "en"}]),
                       encoding="utf-8")
    proc = _run("--file", str(tickets))
    assert proc.returncode == 0, proc.stderr
    assert "Q1" in proc.stdout and "password_reset" in proc.stdout


def test_does_not_touch_committed_artifacts():
    before = {p: (REPO_ROOT / p).read_bytes() for p in ("results.json", "debug_report.json")}
    assert _run("hello").returncode == 0
    assert before == {p: (REPO_ROOT / p).read_bytes() for p in before}
