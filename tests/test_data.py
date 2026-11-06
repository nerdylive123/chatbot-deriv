import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from support_pipeline.data import (
    DEFAULT_PATHS,
    DataError,
    load_articles,
    load_examples,
    load_tickets,
)
from support_pipeline.models import (
    REQUIRED_RESULT_KEYS,
    SAFETY_FLAGS,
    STAGES,
    Article,
    Backends,
    Classification,
    Draft,
    Evidence,
    Intent,
    LabeledExample,
    SimilarExample,
    StageTrace,
    Ticket,
    TicketResult,
    TicketRun,
)

SAMPLE_TICKETS = {
    "T1": "I deposited by bank card two hours ago but my balance still shows zero. What should I check?",
    "T2": "I forgot my password and I am not receiving the reset email.",
    "T3": "Why was my withdrawal declined after verification?",
    "T4": "Can you tell me which asset will go up today so I can make profit?",
    "T5": "My account was locked after too many login attempts. What do I do now?",
}


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def valid_result(**overrides):
    data = {
        "ticket_id": "T1",
        "intent": "deposit_issue",
        "retrieved_articles": ["A1"],
        "reply_draft": "Hello",
        "grounded": True,
        "confidence": 0.8,
        "needs_human_escalation": False,
        "safety_flags": [],
    }
    data.update(overrides)
    return data


# ---------- supplied data ----------

def test_supplied_tickets_load_verbatim_in_order(repo_root):
    tickets = load_tickets(repo_root / "tickets.json")
    assert [t.id for t in tickets] == ["T1", "T2", "T3", "T4", "T5"]
    for t in tickets:
        assert t.message == SAMPLE_TICKETS[t.id]
        assert t.language == "en"


def test_supplied_articles_load_in_order(repo_root):
    articles = load_articles(repo_root / "kb_articles.json")
    assert [a.article_id for a in articles] == ["A1", "A2", "A3", "A4", "A5"]
    assert articles[0].title == "Card deposit processing times"
    assert articles[4].title == "No investment or trading advice"
    assert all(a.body.strip() for a in articles)


def test_supplied_examples_cover_every_intent(repo_root):
    examples = load_examples(repo_root / "examples.json")
    assert len(examples) >= 18
    counts = Counter(e.intent for e in examples)
    for intent in Intent:
        assert counts[intent] >= 3, intent


def test_examples_do_not_copy_sample_tickets(repo_root):
    examples = load_examples(repo_root / "examples.json")
    sample = {m.lower() for m in SAMPLE_TICKETS.values()}
    assert not any(e.message.lower() in sample for e in examples)


def test_default_paths_point_at_repo_root(repo_root):
    assert DEFAULT_PATHS["tickets"] == repo_root / "tickets.json"
    assert DEFAULT_PATHS["articles"] == repo_root / "kb_articles.json"
    assert DEFAULT_PATHS["examples"] == repo_root / "examples.json"


def test_json_files_use_lf_and_indent(repo_root):
    for name in ("tickets.json", "kb_articles.json", "examples.json"):
        raw = (repo_root / name).read_bytes()
        assert b"\r\n" not in raw
        assert raw.endswith(b"\n")
        assert raw.startswith(b"[\n  {")


# ---------- loader errors ----------

@pytest.mark.parametrize("loader", [load_tickets, load_articles, load_examples])
def test_missing_file_raises_data_error(tmp_path, loader):
    with pytest.raises(DataError, match="not found"):
        loader(tmp_path / "nope.json")


@pytest.mark.parametrize("loader", [load_tickets, load_articles, load_examples])
def test_invalid_json_raises_data_error(tmp_path, loader):
    p = tmp_path / "bad.json"
    p.write_text("[{not json", encoding="utf-8")
    with pytest.raises(DataError, match="invalid JSON"):
        loader(p)


@pytest.mark.parametrize("loader", [load_tickets, load_articles, load_examples])
def test_wrong_top_level_type_raises(tmp_path, loader):
    p = write_json(tmp_path / "obj.json", {"id": "T1"})
    with pytest.raises(DataError, match="array"):
        loader(p)


def test_data_error_is_value_error():
    assert issubclass(DataError, ValueError)


def test_ticket_schema_error(tmp_path):
    p = write_json(tmp_path / "t.json", [{"id": "T1"}])
    with pytest.raises(DataError, match="message"):
        load_tickets(p)


def test_ticket_non_object_entry(tmp_path):
    p = write_json(tmp_path / "t.json", ["hello"])
    with pytest.raises(DataError):
        load_tickets(p)


def test_article_schema_error(tmp_path):
    p = write_json(tmp_path / "a.json", [{"article_id": "A1", "title": "x"}])
    with pytest.raises(DataError, match="body"):
        load_articles(p)


def test_example_bad_intent(tmp_path):
    p = write_json(tmp_path / "e.json", [{"message": "hi", "intent": "refund"}])
    with pytest.raises(DataError):
        load_examples(p)


def test_duplicate_ticket_ids(tmp_path):
    p = write_json(tmp_path / "t.json", [{"id": "T1", "message": "a"}, {"id": "T1", "message": "b"}])
    with pytest.raises(DataError, match="duplicate"):
        load_tickets(p)


def test_duplicate_article_ids(tmp_path):
    art = {"article_id": "A1", "title": "t", "body": "b"}
    p = write_json(tmp_path / "a.json", [art, art])
    with pytest.raises(DataError, match="duplicate"):
        load_articles(p)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_ticket_id_rejected(tmp_path, blank):
    p = write_json(tmp_path / "t.json", [{"id": blank, "message": "a"}])
    with pytest.raises(DataError, match="blank"):
        load_tickets(p)


def test_blank_article_id_rejected(tmp_path):
    p = write_json(tmp_path / "a.json", [{"article_id": " ", "title": "t", "body": "b"}])
    with pytest.raises(DataError, match="blank"):
        load_articles(p)


def test_blank_message_allowed_and_language_defaults(tmp_path):
    p = write_json(tmp_path / "t.json", [{"id": "X9", "message": ""}, {"id": "X1", "message": "  "}])
    tickets = load_tickets(p)
    assert [t.id for t in tickets] == ["X9", "X1"]
    assert tickets[0].message == "" and tickets[0].language == "en"


def test_load_examples_ignores_optional_id(tmp_path):
    p = write_json(tmp_path / "e.json", [
        {"id": "EX01", "message": "hi", "intent": "other"},
        {"message": "hola", "intent": "other", "language": "es"},
    ])
    examples = load_examples(p)
    assert examples[0] == LabeledExample(message="hi", intent=Intent.other)
    assert "id" not in examples[0].model_dump()
    assert examples[1].language == "es"


# ---------- models ----------

def test_intent_values_and_order():
    assert [i.value for i in Intent] == [
        "deposit_issue", "password_reset", "withdrawal_issue",
        "account_lock", "trading_advice_request", "other",
    ]
    assert issubclass(Intent, str)


def test_constants():
    assert REQUIRED_RESULT_KEYS == (
        "ticket_id", "intent", "retrieved_articles", "reply_draft",
        "grounded", "confidence", "needs_human_escalation", "safety_flags",
    )
    assert STAGES == (
        "LOAD_DATA", "INDEX_KB", "CLASSIFY_INTENT", "RETRIEVE_CONTEXT",
        "GENERATE_REPLY", "VALIDATE_OUTPUT", "WRITE_RESULTS",
    )
    assert SAFETY_FLAGS == (
        "advice_request", "insufficient_grounding", "policy_sensitive",
        "low_confidence", "account_specific_request", "conflicting_signals",
        "ungrounded_claim",
    )


def test_ticket_result_field_order_and_json_dump():
    assert tuple(TicketResult.model_fields) == REQUIRED_RESULT_KEYS
    dumped = TicketResult(**valid_result()).model_dump(mode="json")
    assert tuple(dumped) == REQUIRED_RESULT_KEYS
    assert dumped["intent"] == "deposit_issue"
    assert type(dumped["intent"]) is str
    assert json.loads(json.dumps(dumped))["intent"] == "deposit_issue"


def test_ticket_result_rejects_extra_key():
    with pytest.raises(ValidationError):
        TicketResult(**valid_result(debug="x"))


@pytest.mark.parametrize("missing", REQUIRED_RESULT_KEYS)
def test_ticket_result_rejects_missing_key(missing):
    data = valid_result()
    del data[missing]
    with pytest.raises(ValidationError):
        TicketResult(**data)


def test_ticket_result_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        TicketResult(**valid_result(intent="refund"))


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_confidence_bounds(value):
    with pytest.raises(ValidationError):
        TicketResult(**valid_result(confidence=value))


@pytest.mark.parametrize("value", [0, 1])
def test_confidence_bounds_inclusive(value):
    assert TicketResult(**valid_result(confidence=value)).confidence == value


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_score_bounds(value):
    with pytest.raises(ValidationError):
        SimilarExample(message="m", intent="other", score=value)
    with pytest.raises(ValidationError):
        Classification(intent="other", certainty=value, goal="g", constraints=[],
                       evidence=[], backend="similarity_fallback")
    with pytest.raises(ValidationError):
        Evidence(article_id="A1", title="t", snippet="s", lexical_score=0.5, relevance=value)


def test_evidence_defaults_and_raw_scores():
    ev = Evidence(article_id="A1", title="t", snippet="s", lexical_score=0.4, relevance=0.4)
    assert ev.rerank_score is None
    ev2 = Evidence(article_id="A1", title="t", snippet="s", lexical_score=0.4,
                   rerank_score=-3.2, relevance=0.04)
    assert ev2.rerank_score == -3.2


def test_classification_and_draft_backend_literals():
    c = Classification(intent="other", certainty=0, goal="", constraints=[], evidence=[],
                       backend="deepseek")
    assert c.conflicting_signals is False
    with pytest.raises(ValidationError):
        Classification(intent="other", certainty=0, goal="", constraints=[], evidence=[],
                       backend="gpt")
    Draft(text="x", cited_article_ids=["A1"], sufficient_evidence=True, backend="kimi")
    with pytest.raises(ValidationError):
        Draft(text="x", cited_article_ids=[], sufficient_evidence=True, backend="llm")


def test_backends_literals():
    b = Backends(retrieval="tfidf", classifier="similarity_fallback", generator="template_fallback")
    assert b.model_dump() == {"retrieval": "tfidf", "classifier": "similarity_fallback",
                              "generator": "template_fallback"}
    Backends(retrieval="embedding", classifier="deepseek", generator="kimi")
    for bad in ({"retrieval": "bm25"}, {"classifier": "x"}, {"generator": "y"}):
        data = {"retrieval": "tfidf", "classifier": "deepseek", "generator": "kimi", **bad}
        with pytest.raises(ValidationError):
            Backends(**data)


def test_stage_trace_defaults_and_status():
    t = StageTrace(stage="LOAD_DATA", summary="ok")
    assert (t.input, t.output, t.backend, t.device, t.status, t.duration_ms) == (
        {}, {}, "pending", "none", "pending", 0)
    with pytest.raises(ValidationError):
        StageTrace(stage="X", summary="s", status="done")
    with pytest.raises(ValidationError):
        StageTrace(stage="X", summary="s", duration_ms=-1)


def test_ticket_run_composes():
    run = TicketRun(
        result=TicketResult(**valid_result()),
        traces=[StageTrace(stage=s, summary="") for s in STAGES],
        escalation_reasons=[],
        confidence_factors={"confidence": 0.8},
        backends=Backends(retrieval="tfidf", classifier="similarity_fallback",
                          generator="template_fallback"),
    )
    assert run.result.intent is Intent.deposit_issue


def test_simple_models():
    assert Ticket(id="T1", message="m").language == "en"
    assert Article(article_id="A1", title="t", body="b").title == "t"
    assert LabeledExample(message="m", intent="password_reset").intent is Intent.password_reset
