import json
import re

import pytest

import main as cli
from support_pipeline.agents import (
    DEEPSEEK_SYSTEM_PROMPT,
    KIMI_SYSTEM_PROMPT,
    DeepSeekClassifier,
    KimiGenerator,
)
from support_pipeline.data import load_articles, load_examples, load_tickets
from support_pipeline.fallback import SimilarityClassifier, TemplateGenerator
from support_pipeline.models import (
    REQUIRED_RESULT_KEYS,
    STAGES,
    Classification,
    Draft,
    Evidence,
    Intent,
    SimilarExample,
    Ticket,
)
from support_pipeline.pipeline import ConfigError, Pipeline, build_pipeline
from support_pipeline.retrieval import TfidfRetriever
from support_pipeline.serialize import build_debug_report, build_results

TRACE_KEYS = ["stage", "summary", "input", "output", "backend", "device", "status"]
TICKET_ENTRY_KEYS = [
    "ticket_id",
    "predicted_intent",
    "retrieved_article_titles",
    "escalated",
    "escalation_reasons",
    "confidence_factors",
    "backends",
]


# ------------------------------------------------------------------ fakes


SELECTED = [
    Evidence(article_id="A1", title="Card deposits", snippet="Card deposits are usually instant.",
             lexical_score=0.8, relevance=0.8),
    Evidence(article_id="A2", title="Password reset", snippet="Check spam folders.",
             lexical_score=0.5, relevance=0.5),
]


class FakeRetriever:
    name = "tfidf"

    def __init__(self, log):
        self.log = log

    def index(self, articles, examples):
        self.log.append("index")

    def similar_examples(self, ticket):
        self.log.append("similar")
        return [SimilarExample(message="my card deposit is missing", intent=Intent.deposit_issue, score=0.9)]

    def retrieve(self, ticket):
        self.log.append("retrieve")
        return list(SELECTED)


class FakeClassifier:
    name = "similarity_fallback"

    def __init__(self, log):
        self.log = log

    def classify(self, ticket, similar):
        self.log.append("classify")
        return Classification(intent=Intent.deposit_issue, certainty=0.9, goal="Resolve a deposit",
                              constraints=["no account facts"], evidence=["deposit"],
                              backend="similarity_fallback")


class RecordingGenerator:
    name = "template_fallback"

    def __init__(self, log):
        self.log = log
        self.calls = []

    def draft(self, *args, **kwargs):
        self.log.append("draft")
        self.calls.append((args, kwargs))
        ticket, classification, evidence = args
        return Draft(text="Card deposits are usually instant.", cited_article_ids=["A1"],
                     sufficient_evidence=True, backend="template_fallback")


def fake_pipeline():
    log = []
    generator = RecordingGenerator(log)
    pipe = Pipeline(FakeRetriever(log), FakeClassifier(log), generator)
    return pipe, generator, log


TICKET = Ticket(id="X1", message="My card deposit has not arrived.")


class FailingTransport:
    def chat_json(self, model, messages, schema, max_tokens, temperature):
        raise RuntimeError("boom")


class ScriptedTransport:
    """Valid DeepSeek output; Kimi cites one selected article and one unknown article."""

    def __init__(self):
        self.calls = []

    def chat_json(self, model, messages, schema, max_tokens, temperature):
        self.calls.append(model)
        if model == DeepSeekClassifier.MODEL:
            return json.dumps({"intent": "deposit_issue", "certainty": 0.9, "goal": "Resolve a deposit",
                               "constraints": [], "evidence": ["deposit"], "conflicting_signals": False})
        return json.dumps({"text": "Please confirm the payment and check your transaction history.",
                           "cited_article_ids": ["A1", "A99"], "sufficient_evidence": True})


@pytest.fixture
def sample_data(repo_root):
    return (
        load_tickets(repo_root / "tickets.json"),
        load_articles(repo_root / "kb_articles.json"),
        load_examples(repo_root / "examples.json"),
    )


def offline_runs(tickets, articles, examples):
    pipe = Pipeline(TfidfRetriever(), SimilarityClassifier(), TemplateGenerator())
    return pipe.run(tickets, articles, examples)


# ------------------------------------------------------------------ fakes: orchestration


def test_stage_order_and_names_exact():
    pipe, _, _ = fake_pipeline()
    (run,) = pipe.run([TICKET], [], [])
    assert [t.stage for t in run.traces] == list(STAGES)
    assert all(t.status in {"completed", "fallback"} for t in run.traces)


def test_work_order():
    pipe, _, log = fake_pipeline()
    pipe.run([TICKET, Ticket(id="X2", message="again")], [], [])
    assert log == ["index", "similar", "classify", "retrieve", "draft", "similar", "classify", "retrieve", "draft"]


def test_generator_receives_only_selected_evidence():
    pipe, generator, _ = fake_pipeline()
    pipe.run([TICKET], [], [])
    ((args, kwargs),) = generator.calls
    assert kwargs == {}
    ticket, classification, evidence = args
    assert ticket == TICKET
    assert isinstance(classification, Classification)
    assert evidence == SELECTED


def test_generate_reply_trace_is_compact():
    pipe, _, _ = fake_pipeline()
    (run,) = pipe.run([TICKET], [], [])
    gen = next(t for t in run.traces if t.stage == "GENERATE_REPLY")
    assert list(gen.input) == ["message", "intent", "goal", "constraints", "evidence"]
    assert gen.input["message"] == TICKET.message
    for item in gen.input["evidence"]:
        assert list(item) == ["article_id", "title", "snippet", "relevance"]
    assert [e["article_id"] for e in gen.input["evidence"]] == ["A1", "A2"]
    dumped = json.dumps(gen.model_dump())
    assert "my card deposit is missing" not in dumped  # similar-example history stays out


def test_classify_trace_shows_similar_examples():
    pipe, _, _ = fake_pipeline()
    (run,) = pipe.run([TICKET], [], [])
    cls = next(t for t in run.traces if t.stage == "CLASSIFY_INTENT")
    assert cls.input["similar_examples"] == [
        {"intent": "deposit_issue", "score": 0.9, "message": "my card deposit is missing"}
    ]
    assert cls.output["intent"] == "deposit_issue"


def test_one_result_per_ticket_in_input_order():
    pipe, _, _ = fake_pipeline()
    tickets = [Ticket(id=i, message="deposit") for i in ["B", "A", "C"]]
    runs = pipe.run(tickets, [], [])
    assert [r.result.ticket_id for r in runs] == ["B", "A", "C"]


def test_duplicate_ticket_ids_rejected():
    pipe, _, _ = fake_pipeline()
    with pytest.raises(ValueError, match="duplicate"):
        pipe.run([TICKET, TICKET], [], [])


def test_run_ticket_requires_index():
    pipe, _, _ = fake_pipeline()
    with pytest.raises(RuntimeError, match="index"):
        pipe.run_ticket(TICKET)


def test_offline_trace_backends_and_devices(sample_data):
    runs = offline_runs(*sample_data)
    by_stage = {t.stage: t for t in runs[0].traces}
    assert by_stage["CLASSIFY_INTENT"].backend == "similarity_fallback"
    assert by_stage["GENERATE_REPLY"].backend == "template_fallback"
    assert by_stage["RETRIEVE_CONTEXT"].backend == "tfidf"
    assert all(t.device == "cpu" for t in runs[0].traces)
    assert by_stage["WRITE_RESULTS"].output == {"results_file": "results.json", "debug_file": "debug_report.json"}
    assert all(t.status == "completed" for t in runs[0].traces)


def test_empty_message_is_safe(sample_data):
    _, articles, examples = sample_data
    known = {a.article_id for a in articles}
    runs = offline_runs([Ticket(id="E1", message=""), Ticket(id="E2", message="   \n\t")], articles, examples)
    for run in runs:
        assert run.result.intent == Intent.other
        assert run.result.needs_human_escalation is True
        assert run.result.grounded is False
        assert set(run.result.retrieved_articles) <= known


def test_renamed_ids_behave_identically(sample_data):
    tickets, articles, examples = sample_data
    renamed = [t.model_copy(update={"id": f"ZZ{9 - i}"}) for i, t in enumerate(tickets)]
    base = offline_runs(tickets, articles, examples)
    other = offline_runs(renamed, articles, examples)
    for a, b, new in zip(base, other, renamed):
        assert b.result.ticket_id == new.id
        assert a.result.model_dump(exclude={"ticket_id"}) == b.result.model_dump(exclude={"ticket_id"})
        assert [t.model_dump() for t in a.traces] == [t.model_dump() for t in b.traces]
        assert a.escalation_reasons == b.escalation_reasons


# ------------------------------------------------------------------ Together components


def test_together_failure_falls_back_visibly(sample_data):
    tickets, articles, examples = sample_data
    pipe = Pipeline(TfidfRetriever(), DeepSeekClassifier(FailingTransport()), KimiGenerator(FailingTransport()),
                    mode="together")
    runs = pipe.run(tickets, articles, examples)
    offline = offline_runs(tickets, articles, examples)
    for run, base in zip(runs, offline):
        assert run.backends.model_dump() == {"retrieval": "tfidf", "classifier": "similarity_fallback",
                                             "generator": "template_fallback"}
        by_stage = {t.stage: t for t in run.traces}
        for stage in ("CLASSIFY_INTENT", "GENERATE_REPLY"):
            assert by_stage[stage].status == "fallback"
            assert by_stage[stage].device == "cpu"
            assert by_stage[stage].output["fallback_reason"] == "provider error (RuntimeError)"
        assert run.result == base.result


def test_together_success_records_remote_and_dropped_citation(sample_data):
    _, articles, examples = sample_data
    transport = ScriptedTransport()
    pipe = Pipeline(TfidfRetriever(), DeepSeekClassifier(transport), KimiGenerator(transport), mode="together")
    (run,) = pipe.run([Ticket(id="Q1", message="My card deposit has not arrived in my balance.")],
                      articles, examples)
    assert run.backends.classifier == "deepseek"
    assert run.backends.generator == "kimi"
    by_stage = {t.stage: t for t in run.traces}
    for stage, backend in (("CLASSIFY_INTENT", "deepseek"), ("GENERATE_REPLY", "kimi")):
        assert by_stage[stage].backend == backend
        assert by_stage[stage].device == "remote"
        assert by_stage[stage].status == "completed"
    assert by_stage["GENERATE_REPLY"].output["dropped_citations"] == ["A99"]
    # The dropped citation reaches policy: ungrounded, flagged, escalated.
    assert "ungrounded_claim" in run.result.safety_flags
    assert run.result.grounded is False
    assert run.result.needs_human_escalation is True
    assert "A99" not in run.result.retrieved_articles
    dumped = json.dumps([t.model_dump() for t in run.traces])
    assert DEEPSEEK_SYSTEM_PROMPT not in dumped and KIMI_SYSTEM_PROMPT not in dumped


def test_debug_report_top_level_backends_are_configured(monkeypatch, sample_data):
    tickets, articles, examples = sample_data
    monkeypatch.setenv("TOGETHER_API_KEY", "dummy")
    pipe, backends, notes = build_pipeline("together", transport=FailingTransport())
    runs = pipe.run(tickets, articles, examples)
    report = build_debug_report(runs, "together", backends)
    assert report["backends"] == {"retrieval": "tfidf", "classifier": "deepseek", "generator": "kimi"}
    assert report["tickets"][0]["backends"]["classifier"] == "similarity_fallback"


# ------------------------------------------------------------------ build_pipeline


def test_build_pipeline_offline_default():
    pipe, backends, notes = build_pipeline()
    assert pipe.mode == "offline"
    assert backends.model_dump() == {"retrieval": "tfidf", "classifier": "similarity_fallback",
                                     "generator": "template_fallback"}
    assert notes == []


def test_build_pipeline_embedding_falls_back_with_note():
    pipe, backends, notes = build_pipeline(retrieval="embedding")
    if backends.retrieval == "tfidf":  # sentence-transformers absent or models not cached
        assert notes and "TF-IDF" in notes[0]
    else:
        assert notes == []


def test_build_pipeline_together_requires_key(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="TOGETHER_API_KEY"):
        build_pipeline("together")


def test_build_pipeline_together_with_key_never_exposes_it(monkeypatch, sample_data):
    secret = "tg-test-secret-123456"
    monkeypatch.setenv("TOGETHER_API_KEY", secret)
    pipe, backends, _ = build_pipeline("together", transport=FailingTransport())
    assert backends.classifier == "deepseek" and backends.generator == "kimi"
    runs = pipe.run(*sample_data)
    blob = json.dumps(build_results(runs)) + json.dumps(build_debug_report(runs, "together", backends))
    assert secret not in blob


def test_build_pipeline_unknown_mode():
    with pytest.raises(ConfigError, match="mode"):
        build_pipeline("cloud")


# ------------------------------------------------------------------ real offline pipeline


def _floats_ok(path):
    seen = []
    json.loads(path.read_text(encoding="utf-8"), parse_float=lambda s: seen.append(s) or float(s))
    assert seen, "expected some floats"
    for literal in seen:
        assert re.fullmatch(r"-?\d+(\.\d{1,3})?", literal), literal


def test_real_offline_runs_are_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert cli.main(["--out", str(a)]) == 0
    assert cli.main(["--out", str(b)]) == 0
    for name in ("results.json", "debug_report.json"):
        raw = (a / name).read_bytes()
        assert raw == (b / name).read_bytes()
        assert raw.endswith(b"\n") and b"\r\n" not in raw
        _floats_ok(a / name)


def test_real_offline_artifact_shapes(tmp_path, sample_data):
    tickets = sample_data[0]
    assert cli.main(["--out", str(tmp_path)]) == 0
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [r["ticket_id"] for r in results] == [t.id for t in tickets]
    for r in results:
        assert list(r) == list(REQUIRED_RESULT_KEYS)
    by_id = {r["ticket_id"]: r for r in results}
    assert "A1" in by_id["T1"]["retrieved_articles"]
    assert by_id["T4"]["intent"] == "trading_advice_request"
    assert "advice_request" in by_id["T4"]["safety_flags"]

    report = json.loads((tmp_path / "debug_report.json").read_text(encoding="utf-8"))
    assert list(report) == ["mode", "backends", "tickets", "traces"]
    assert report["mode"] == "offline"
    assert report["backends"] == {"retrieval": "tfidf", "classifier": "similarity_fallback",
                                  "generator": "template_fallback"}
    assert [e["ticket_id"] for e in report["tickets"]] == [t.id for t in tickets]
    for entry in report["tickets"]:
        assert list(entry) == TICKET_ENTRY_KEYS
        assert list(entry["confidence_factors"]) == [
            "classifier_certainty", "best_relevance", "penalty_insufficient_evidence",
            "penalty_conflicting_signals", "penalty_ungrounded_claim", "confidence",
        ]
        assert all(isinstance(t, str) for t in entry["retrieved_article_titles"])
        assert len(entry["retrieved_article_titles"]) == len(by_id[entry["ticket_id"]]["retrieved_articles"])
    assert list(report["traces"]) == [t.id for t in tickets]
    for trace_list in report["traces"].values():
        assert [t["stage"] for t in trace_list] == list(STAGES)
        for trace in trace_list:
            assert list(trace) == TRACE_KEYS


# ------------------------------------------------------------------ main.py


def test_main_together_without_key_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    assert cli.main(["--mode", "together", "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "TOGETHER_API_KEY" in err
    assert not (tmp_path / "results.json").exists()


def test_main_data_error_exits_1(tmp_path, capsys):
    assert cli.main(["--data-dir", str(tmp_path), "--out", str(tmp_path)]) == 1
    assert "tickets.json" in capsys.readouterr().err


def test_main_prints_summary(tmp_path, capsys):
    assert cli.main(["--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "T4" in out and "trading_advice_request" in out
