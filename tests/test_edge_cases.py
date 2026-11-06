"""Edge and adversarial tickets through the REAL offline pipeline (sample KB/examples).

Asserts safe outcomes (schema, known IDs, escalation, refusal), not exact labels
for genuinely ambiguous items. Tests may reference sample IDs; pipeline logic
must not.
"""

import re
from pathlib import Path

import pytest

from support_pipeline.data import load_articles, load_examples, load_tickets
from support_pipeline.models import REQUIRED_RESULT_KEYS, SAFETY_FLAGS, TicketResult
from support_pipeline.pipeline import build_pipeline
from support_pipeline.policy import has_refusal, has_unsafe_content

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "edge_tickets.json"
RENAMED = {"X1": "T1", "X2": "T2", "X3": "T3", "X4": "T4", "X5": "T5"}


def _run(tickets):
    pipeline, _, _ = build_pipeline("offline", "tfidf")
    articles = load_articles(REPO_ROOT / "kb_articles.json")
    examples = load_examples(REPO_ROOT / "examples.json")
    return {run.result.ticket_id: run for run in pipeline.run(tickets, articles, examples)}


@pytest.fixture(scope="module")
def edge_runs():
    return _run(load_tickets(FIXTURE))


@pytest.fixture(scope="module")
def sample_runs():
    return _run(load_tickets(REPO_ROOT / "tickets.json"))


def test_fixture_renamed_copies_match_sample_messages():
    edge = {t.id: t for t in load_tickets(FIXTURE)}
    samples = {t.id: t for t in load_tickets(REPO_ROOT / "tickets.json")}
    for new_id, old_id in RENAMED.items():
        assert edge[new_id].message == samples[old_id].message


def test_all_edge_results_are_valid(edge_runs):
    known_articles = {a.article_id for a in load_articles(REPO_ROOT / "kb_articles.json")}
    assert len(edge_runs) == len(load_tickets(FIXTURE))
    for run in edge_runs.values():
        dumped = run.result.model_dump(mode="json")
        assert list(dumped) == list(REQUIRED_RESULT_KEYS)
        TicketResult.model_validate(dumped)
        assert 1 <= len(dumped["retrieved_articles"]) <= 3
        assert set(dumped["retrieved_articles"]) <= known_articles
        assert 0.0 <= dumped["confidence"] <= 1.0
        assert set(dumped["safety_flags"]) <= set(SAFETY_FLAGS)
        assert dumped["reply_draft"].strip()


@pytest.mark.parametrize("ticket_id", ["E_EMPTY", "E_WHITESPACE", "E_SPANISH", "E_UNKNOWN"])
def test_unanswerable_tickets_escalate(edge_runs, ticket_id):
    result = edge_runs[ticket_id].result
    assert result.needs_human_escalation is True
    assert result.grounded is False
    assert edge_runs[ticket_id].escalation_reasons


@pytest.mark.parametrize("ticket_id", ["E_EMPTY", "E_WHITESPACE"])
def test_blank_tickets_are_other_with_zero_certainty(edge_runs, ticket_id):
    run = edge_runs[ticket_id]
    assert run.result.intent.value == "other"
    assert run.confidence_factors["classifier_certainty"] == 0.0


def test_ambiguous_ticket_is_safe(edge_runs):
    result = edge_runs["E_AMBIGUOUS"].result
    # No exact label: only that the reply makes no promises and stays within the KB.
    assert not has_unsafe_content(result.reply_draft, [])
    if result.intent.value in {"withdrawal_issue", "account_lock"} or "conflicting_signals" in result.safety_flags:
        assert result.needs_human_escalation is True


def test_manipulation_is_refused(edge_runs):
    result = edge_runs["E_MANIPULATION"].result
    assert result.intent.value == "trading_advice_request"
    assert "advice_request" in result.safety_flags
    assert has_refusal(result.reply_draft)
    assert not has_unsafe_content(result.reply_draft, [])


def test_injection_fabricates_nothing_and_escalates(edge_runs):
    result = edge_runs["E_INJECTION"].result
    reply = result.reply_draft.lower()
    assert result.needs_human_escalation is True
    assert "admin mode" not in reply
    assert not re.search(r"\d", reply), "reply must not contain any figures"
    assert not re.search(r"your (balance|account) (is|shows|has)", reply)
    assert not has_unsafe_content(result.reply_draft, [])


@pytest.mark.parametrize("new_id, old_id", sorted(RENAMED.items()))
def test_renamed_tickets_behave_identically(edge_runs, sample_runs, new_id, old_id):
    new = edge_runs[new_id].model_dump(mode="json")
    old = sample_runs[old_id].model_dump(mode="json")
    assert new["result"]["ticket_id"] == new_id
    new["result"]["ticket_id"] = old_id
    for run in (new, old):
        for trace in run["traces"]:
            trace.pop("duration_ms", None)
    # Index counts differ (12 vs 5 tickets in LOAD_DATA); everything else must match.
    for run in (new, old):
        run["traces"][0]["output"].pop("tickets", None)
    assert new == old
