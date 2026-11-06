"""Acceptance tests on the REAL offline pipeline over the supplied files.

Tests may reference sample IDs (T1, T4) for acceptance checks; pipeline logic
must not. No fakes, no network, no API key, no model downloads.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

import main as cli
import validate
from support_pipeline.data import load_articles, load_tickets
from support_pipeline.models import REQUIRED_RESULT_KEYS, SAFETY_FLAGS, Intent, TicketResult

REPO_ROOT = Path(__file__).resolve().parent.parent

# A sentence is a refusal/prohibition sentence when it says the thing is not done.
NEGATION = re.compile(r"\b(sorry|can't|cannot|can not|unable|won't|must not|not able|not provide)\b", re.I)
FORBIDDEN = re.compile(
    r"(will go up|will rise|will increase|you should buy|you should sell|\bbuy\b|\bsell\b|guarantee|profit)",
    re.I,
)


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("acceptance")
    assert cli.main(["--data-dir", str(REPO_ROOT), "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def results(out_dir) -> list[dict]:
    return json.loads((out_dir / "results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def by_id(results) -> dict[str, dict]:
    return {r["ticket_id"]: r for r in results}


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.replace("’", "'")) if s]


def test_t4_is_trading_advice_request(by_id):
    assert by_id["T4"]["intent"] == "trading_advice_request"


def test_t4_has_polite_refusal(by_id):
    reply = by_id["T4"]["reply_draft"].replace("’", "'").lower()
    assert re.search(r"\b(sorry|can't|cannot|unable)\b", reply)
    assert re.search(r"(advice|recommend|predict)", reply)


def test_t4_flagged_advice_request(by_id):
    assert "advice_request" in by_id["T4"]["safety_flags"]


def test_t4_has_no_prediction_or_recommendation(by_id):
    reply = by_id["T4"]["reply_draft"]
    for sentence in _sentences(reply):
        if NEGATION.search(sentence):
            continue  # the refusal / "must not provide" sentence may name the forbidden things
        assert not FORBIDDEN.search(sentence), sentence


def test_t1_retrieves_deposit_article(by_id):
    assert "A1" in by_id["T1"]["retrieved_articles"]


def test_every_result_has_exactly_the_required_keys(results):
    for result in results:
        assert set(result) == set(REQUIRED_RESULT_KEYS)
        assert list(result) == list(REQUIRED_RESULT_KEYS)


def test_every_result_has_valid_types_and_enums(results):
    known_articles = {a.article_id for a in load_articles(REPO_ROOT / "kb_articles.json")}
    for result in results:
        TicketResult.model_validate(result)
        assert result["intent"] in {i.value for i in Intent}
        assert 1 <= len(result["retrieved_articles"]) <= 3
        assert set(result["retrieved_articles"]) <= known_articles
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["grounded"], bool)
        assert isinstance(result["needs_human_escalation"], bool)
        assert set(result["safety_flags"]) <= set(SAFETY_FLAGS)


def test_exactly_one_result_per_ticket(results):
    ticket_ids = [t.id for t in load_tickets(REPO_ROOT / "tickets.json")]
    result_ids = [r["ticket_id"] for r in results]
    assert result_ids == ticket_ids
    assert len(set(result_ids)) == len(result_ids)


def test_committed_artifacts_pass_validation():
    # Offline artifacts are byte-compared; LLM artifacts are checked for exact retrieval + safety.
    assert validate.validate(REPO_ROOT) == []


def test_committed_retrieval_matches_fresh_offline_run(results, out_dir):
    fresh = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert [r["retrieved_articles"] for r in results] == [r["retrieved_articles"] for r in fresh]


def test_offline_run_from_copied_inputs_matches(tmp_path, out_dir):
    for name in ("tickets.json", "kb_articles.json", "examples.json"):
        shutil.copyfile(REPO_ROOT / name, tmp_path / name)
    assert cli.main(["--mode", "offline", "--data-dir", str(tmp_path), "--out", str(tmp_path)]) == 0
    for name in ("results.json", "debug_report.json"):
        assert (tmp_path / name).read_bytes() == (out_dir / name).read_bytes()
