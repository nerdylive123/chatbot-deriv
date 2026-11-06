import json

import pytest

from support_pipeline.models import (
    REQUIRED_RESULT_KEYS,
    Backends,
    StageTrace,
    TicketResult,
    TicketRun,
)
from support_pipeline.serialize import (
    build_debug_report,
    build_results,
    dump_json,
    round_floats,
    write_artifacts,
)

BACKENDS = Backends(retrieval="tfidf", classifier="similarity_fallback", generator="template_fallback")


def make_run(ticket_id: str = "T1", confidence: float = 0.123456) -> TicketRun:
    return TicketRun(
        result=TicketResult(
            ticket_id=ticket_id,
            intent="deposit_issue",
            retrieved_articles=["A1"],
            reply_draft="Hello",
            grounded=True,
            confidence=confidence,
            needs_human_escalation=False,
            safety_flags=[],
        ),
        traces=[
            StageTrace(
                stage="LOAD_DATA",
                summary="loaded",
                output={"score": 0.98765},
                backend="json",
                device="cpu",
                status="completed",
                duration_ms=12.3456,
            )
        ],
        escalation_reasons=[],
        confidence_factors={"classifier_certainty": 0.33333, "confidence": confidence},
        backends=BACKENDS,
    )


def test_round_floats_recurses_and_leaves_other_types():
    data = {"a": 0.12345, "b": [1.98765, {"c": 2.0004}], "d": (0.5556,), "e": True, "f": 3, "g": "x"}
    assert round_floats(data) == {"a": 0.123, "b": [1.988, {"c": 2.0}], "d": [0.556], "e": True, "f": 3, "g": "x"}
    assert round_floats(0.00001) == 0.0
    assert round_floats(1.23456, ndigits=1) == 1.2


def test_dump_json_format(tmp_path):
    path = tmp_path / "out.json"
    dump_json({"b": 0.12345, "a": "é"}, path)
    raw = path.read_bytes()
    assert raw == '{\n  "b": 0.123,\n  "a": "é"\n}\n'.encode("utf-8")
    assert b"\r\n" not in raw


def test_build_results_exact_keys_in_order():
    results = build_results([make_run("T1"), make_run("T2")])
    assert [r["ticket_id"] for r in results] == ["T1", "T2"]
    for r in results:
        assert list(r) == list(REQUIRED_RESULT_KEYS)
        assert r["intent"] == "deposit_issue"


def test_build_debug_report_shape():
    report = build_debug_report([make_run("T1")], "offline", BACKENDS)
    assert list(report) == ["mode", "backends", "tickets", "traces"]
    assert report["mode"] == "offline"
    assert report["backends"] == BACKENDS.model_dump()
    (entry,) = report["tickets"]
    assert list(entry) == [
        "ticket_id",
        "predicted_intent",
        "retrieved_article_titles",
        "escalated",
        "escalation_reasons",
        "confidence_factors",
        "backends",
    ]
    assert entry["predicted_intent"] == "deposit_issue"
    assert entry["retrieved_article_titles"] == []  # no RETRIEVE_CONTEXT trace in this run
    assert entry["escalated"] is False
    assert list(report["traces"]) == ["T1"]
    (trace,) = report["traces"]["T1"]
    assert list(trace) == ["stage", "summary", "input", "output", "backend", "device", "status"]
    assert "duration_ms" not in trace


def test_debug_report_titles_come_from_retrieve_trace():
    run = make_run("T1")
    run.traces.append(
        StageTrace(
            stage="RETRIEVE_CONTEXT",
            summary="retrieved",
            output={"evidence": [{"article_id": "A1", "title": "Card deposit processing times", "relevance": 0.5}]},
            status="completed",
        )
    )
    report = build_debug_report([run], "offline", BACKENDS)
    assert report["tickets"][0]["retrieved_article_titles"] == ["Card deposit processing times"]


def test_build_debug_report_rejects_duplicate_ticket_ids():
    with pytest.raises(ValueError, match="duplicate"):
        build_debug_report([make_run("T1"), make_run("T1")], "offline", BACKENDS)


def test_write_artifacts_writes_both_files_rounded(tmp_path):
    out = tmp_path / "nested"
    paths = write_artifacts([make_run("T1")], out, "offline", BACKENDS)
    assert sorted(p.name for p in paths) == ["debug_report.json", "results.json"]
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results[0]["confidence"] == 0.123
    report = json.loads((out / "debug_report.json").read_text(encoding="utf-8"))
    assert report["traces"]["T1"][0]["output"]["score"] == 0.988
    assert report["tickets"][0]["confidence_factors"]["classifier_certainty"] == 0.333
