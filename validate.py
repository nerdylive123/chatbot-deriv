"""Validate the committed artifacts against the inputs and the required contract.

    python validate.py [--root DIR]

Checks: required files exist and are valid JSON; tickets and KB load; exactly one
result per ticket; exact result schema, allowed intents, 1-3 known article IDs,
confidence in [0, 1], boolean fields, known safety flags; sample acceptance
checks (T4 refused, T1 retrieves A1); and reproducibility (rerun the pipeline in
the recorded mode into a temp dir and byte-compare both artifacts).

Exit code 0 when every check passes, 1 otherwise. Secrets are never printed.
Sample IDs (T1, T4) appear here only as acceptance checks; pipeline logic never
depends on ticket IDs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

import main as cli
from support_pipeline.data import DataError, load_articles, load_examples, load_tickets
from support_pipeline.models import REQUIRED_RESULT_KEYS, SAFETY_FLAGS, Classification, Draft, Intent, TicketResult
from support_pipeline.pipeline import MODES
from support_pipeline.policy import evaluate, has_refusal, has_unsafe_content
from support_pipeline.retrieval import _embedding_unavailable_reason, make_retriever
from support_pipeline.serialize import DEBUG_FILE, RESULTS_FILE, round_floats

REPO_ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = ("tickets.json", "kb_articles.json", RESULTS_FILE, DEBUG_FILE, "README.md")
JSON_FILES = ("tickets.json", "kb_articles.json", RESULTS_FILE, DEBUG_FILE)
INTENTS = {intent.value for intent in Intent}
BOOL_FIELDS = ("grounded", "needs_human_escalation")

# Seam for tests: returns "" when the embedding backend can run offline.
embedding_unavailable_reason: Callable[[], str] = _embedding_unavailable_reason


def _first_difference(expected: bytes, actual: bytes) -> str:
    exp_lines = expected.decode("utf-8", "replace").split("\n")
    act_lines = actual.decode("utf-8", "replace").split("\n")
    for number, (exp, act) in enumerate(zip(exp_lines, act_lines), start=1):
        if exp != act:
            return f"first difference at line {number}: committed {exp!r}, rerun {act!r}"
    number = min(len(exp_lines), len(act_lines)) + 1
    return f"first difference at line {number}: files have different lengths"


class _Check:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.passed: list[str] = []

    def ok(self, message: str) -> None:
        self.passed.append(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)


def _check_result(index: int, item: Any, known_articles: set[str], check: _Check) -> None:
    label = f"results.json entry {index}"
    if not isinstance(item, dict):
        check.fail(f"{label}: must be a JSON object")
        return
    label = f"results.json entry {index} ({item.get('ticket_id', '?')!s})"
    keys = set(item)
    missing = [k for k in REQUIRED_RESULT_KEYS if k not in keys]
    extra = sorted(keys - set(REQUIRED_RESULT_KEYS))
    if missing:
        check.fail(f"{label}: missing keys {missing}")
    if extra:
        check.fail(f"{label}: extra keys {extra}")

    if "ticket_id" in item and not isinstance(item["ticket_id"], str):
        check.fail(f"{label}: ticket_id must be a string, got {item['ticket_id']!r}")

    intent = item.get("intent")
    if "intent" in item and (not isinstance(intent, str) or intent not in INTENTS):
        check.fail(f"{label}: intent {intent!r} is not one of {sorted(INTENTS)}")

    articles = item.get("retrieved_articles")
    if "retrieved_articles" in item:
        if not isinstance(articles, list) or not all(isinstance(a, str) for a in articles):
            check.fail(f"{label}: retrieved_articles must be a list of article IDs")
        else:
            if not 1 <= len(articles) <= 3:
                check.fail(f"{label}: retrieved_articles must have 1-3 IDs, got {len(articles)}")
            unknown = [a for a in articles if a not in known_articles]
            if unknown:
                check.fail(f"{label}: unknown article IDs {unknown}")

    confidence = item.get("confidence")
    if "confidence" in item:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            check.fail(f"{label}: confidence must be a number, got {confidence!r}")
        elif not 0.0 <= confidence <= 1.0:
            check.fail(f"{label}: confidence {confidence} is outside [0, 1]")

    for name in BOOL_FIELDS:
        if name in item and not isinstance(item[name], bool):
            check.fail(f"{label}: {name} must be a bool, got {item[name]!r}")

    flags = item.get("safety_flags")
    if "safety_flags" in item:
        if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
            check.fail(f"{label}: safety_flags must be a list of strings")
        else:
            unknown_flags = [f for f in flags if f not in SAFETY_FLAGS]
            if unknown_flags:
                check.fail(f"{label}: unknown safety flags {unknown_flags}")

    if "reply_draft" in item and not isinstance(item["reply_draft"], str):
        check.fail(f"{label}: reply_draft must be a string")

    if not missing and not extra:
        try:
            TicketResult.model_validate(item, strict=False)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            check.fail(f"{label}: does not validate as TicketResult ({problems})")


def _check_acceptance(by_id: dict[str, dict[str, Any]], check: _Check) -> None:
    t4 = by_id.get("T4")
    if isinstance(t4, dict):
        before = len(check.errors)
        if t4.get("intent") != Intent.trading_advice_request.value:
            check.fail(f"T4: intent must be trading_advice_request, got {t4.get('intent')!r}")
        if "advice_request" not in (t4.get("safety_flags") or []):
            check.fail("T4: safety_flags must include advice_request")
        reply = t4.get("reply_draft")
        reply = reply if isinstance(reply, str) else ""
        if not has_refusal(reply):
            check.fail("T4: reply_draft must contain a polite refusal of trading advice")
        if has_unsafe_content(reply, []):
            check.fail("T4: reply_draft contains prediction/recommendation text outside the refusal")
        if len(check.errors) == before:
            check.ok("T4 is classified trading_advice_request, refused politely, flagged advice_request")
    t1 = by_id.get("T1")
    if isinstance(t1, dict):
        if "A1" in (t1.get("retrieved_articles") or []):
            check.ok("T1 retrieves A1")
        else:
            check.fail("T1: retrieved_articles must include A1 (card deposit article)")


def _check_reproducibility(root: Path, report: Any, check: _Check) -> None:
    if not isinstance(report, dict):
        check.fail("debug_report.json: expected a JSON object with mode and backends")
        return
    mode = report.get("mode")
    backends = report.get("backends")
    retrieval = backends.get("retrieval") if isinstance(backends, dict) else None
    if mode not in MODES:
        check.fail(f"debug_report.json: unknown mode {mode!r}; expected one of {', '.join(MODES)}")
        return
    if retrieval not in ("tfidf", "embedding"):
        check.fail(f"debug_report.json: unknown retrieval backend {retrieval!r}")
        return
    if retrieval == "embedding":
        reason = embedding_unavailable_reason()
        if reason:
            check.fail(
                "results.json was generated with the embedding retrieval backend, which is not "
                f"available here ({reason}); rerun `python main.py --retrieval tfidf` or install and "
                "cache the local models (requirements-local-models.txt)"
            )
            return

    # Reruns are always offline: no key, no network, no cost. Retrieval is local and
    # deterministic in every mode, so it is compared exactly; LLM wording is not
    # byte-reproducible, so together-mode reply text is checked by schema and safety only.
    with tempfile.TemporaryDirectory(prefix="validate-rerun-") as tmp:
        first, second = Path(tmp) / "a", Path(tmp) / "b"
        for out in (first, second):
            argv = ["--mode", "offline", "--retrieval", retrieval, "--data-dir", str(root), "--out", str(out)]
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                code = cli.main(argv)
            if code != 0:
                detail = sink.getvalue().strip().splitlines()
                check.fail(f"reproducibility: rerun `python main.py --mode offline` failed (exit {code}): "
                           f"{detail[-1] if detail else 'no output'}")
                return

        if mode == "offline":
            mismatched = False
            for name in (RESULTS_FILE, DEBUG_FILE):
                committed = (root / name).read_bytes()
                rerun = (first / name).read_bytes()
                if committed != rerun:
                    mismatched = True
                    check.fail(f"{name}: not reproducible by an offline/{retrieval} rerun; "
                               f"{_first_difference(committed, rerun)}")
            if not mismatched:
                check.ok(f"rerun in offline/{retrieval} mode reproduces results.json and "
                         "debug_report.json byte-for-byte")
            return

        for name in (RESULTS_FILE, DEBUG_FILE):
            if (first / name).read_bytes() != (second / name).read_bytes():
                check.fail(f"{name}: two offline reruns differ; the offline path is not deterministic")
                return
        committed = json.loads((root / RESULTS_FILE).read_text(encoding="utf-8"))
        rerun = json.loads((first / RESULTS_FILE).read_text(encoding="utf-8"))
        expected = {r["ticket_id"]: r["retrieved_articles"] for r in rerun}
        actual = {r.get("ticket_id"): r.get("retrieved_articles") for r in committed if isinstance(r, dict)}
        diffs = sorted(t for t in expected if actual.get(t) != expected[t])
        if diffs:
            check.fail("results.json: retrieved_articles differ from a fresh deterministic retrieval for "
                       + ", ".join(f"{t} (committed {actual.get(t)}, rerun {expected[t]})" for t in diffs))
            return
        check.ok(f"{mode} artifacts: retrieval matches a fresh deterministic run for every ticket; "
                 "offline path reproduces byte-for-byte (LLM wording is not byte-reproducible by design)")


REPLAYED_FIELDS = ("reply_draft", "grounded", "confidence", "needs_human_escalation", "safety_flags")


def _stage(traces: list[Any], name: str) -> dict[str, Any]:
    for trace in traces:
        if isinstance(trace, dict) and trace.get("stage") == name:
            return trace
    raise KeyError(name)


def _check_policy_replay(root: Path, results: list[Any], report: Any, check: _Check) -> None:
    """Re-run the deterministic policy on every saved result, in any mode.

    Inputs: the saved classification and the generator's draft (from the debug traces)
    plus evidence from a fresh local retrieval. The saved decision fields must equal
    the replayed ones, so a hand-edited escalation flag or an unsafe sentence added to
    any reply fails even though LLM wording itself is not reproducible.
    """
    if not isinstance(report, dict) or not isinstance(report.get("traces"), dict):
        check.fail("debug_report.json: expected a traces object to replay the policy")
        return
    backends = report.get("backends") if isinstance(report.get("backends"), dict) else {}
    retriever, note = make_retriever(backends.get("retrieval", "tfidf"))
    if note:
        retriever, _ = make_retriever("tfidf")
    retriever.index(load_articles(root / "kb_articles.json"), load_examples(root / "examples.json"))
    tickets = {t.id: t for t in load_tickets(root / "tickets.json")}
    before = len(check.errors)
    for item in results:
        tid = item["ticket_id"]
        label = f"results.json {tid}"
        try:
            traces = report["traces"][tid]
            cls_out = _stage(traces, "CLASSIFY_INTENT")["output"]
            gen = _stage(traces, "GENERATE_REPLY")
            val_in = _stage(traces, "VALIDATE_OUTPUT")["input"]
            ticket_backends = next(
                (t.get("backends", {}) for t in report.get("tickets", []) if isinstance(t, dict) and t.get("ticket_id") == tid),
                {},
            )
            classification = Classification(
                intent=cls_out["intent"], certainty=cls_out["certainty"], goal=cls_out.get("goal", ""),
                constraints=gen["input"].get("constraints", []), evidence=[],
                conflicting_signals=cls_out.get("conflicting_signals", False),
                backend=ticket_backends.get("classifier", "similarity_fallback"),
            )
            draft = Draft(
                text=gen["output"]["reply_draft"], cited_article_ids=val_in["cited_article_ids"],
                sufficient_evidence=val_in["sufficient_evidence"],
                backend=ticket_backends.get("generator", "template_fallback"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            check.fail(f"{label}: debug_report.json traces are incomplete for a policy replay ({type(exc).__name__}: {exc})")
            continue
        if item["intent"] != classification.intent.value:
            check.fail(f"{label}: intent {item['intent']!r} differs from the saved classification "
                       f"{classification.intent.value!r}")
        outcome = evaluate(tickets[tid], classification, retriever.retrieve(tickets[tid]), draft)
        expected = round_floats(outcome.model_dump(mode="json"))
        for field in REPLAYED_FIELDS:
            if item[field] != expected[field]:
                check.fail(f"{label}: {field} is {item[field]!r} but the deterministic policy gives "
                           f"{expected[field]!r}")
        if has_unsafe_content(item["reply_draft"], []) and "ungrounded_claim" not in item["safety_flags"]:
            check.fail(f"{label}: reply_draft contains an unsafe or unsupported claim without ungrounded_claim")
    if len(check.errors) == before:
        check.ok(f"policy replay: every saved reply, grounding, confidence, escalation and flag set "
                 f"matches the deterministic policy ({len(results)} tickets)")


def _run_checks(root: Path) -> _Check:
    check = _Check()
    root = Path(root)

    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    for name in missing:
        check.fail(f"{name}: required file is missing from {root}")
    if not missing:
        check.ok("required files exist: " + ", ".join(REQUIRED_FILES))

    parsed: dict[str, Any] = {}
    for name in JSON_FILES:
        if name in missing:
            continue
        try:
            parsed[name] = json.loads((root / name).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            check.fail(f"{name}: invalid JSON ({exc})")
    if len(parsed) == len(JSON_FILES):
        check.ok("all JSON files parse")

    tickets = articles = None
    if "tickets.json" in parsed:
        try:
            tickets = load_tickets(root / "tickets.json")
        except DataError as exc:
            check.fail(str(exc))
    if "kb_articles.json" in parsed:
        try:
            articles = load_articles(root / "kb_articles.json")
        except DataError as exc:
            check.fail(str(exc))
    if tickets is not None and articles is not None:
        check.ok(f"loaded {len(tickets)} tickets and {len(articles)} KB articles")

    results = parsed.get(RESULTS_FILE)
    if RESULTS_FILE in parsed and not isinstance(results, list):
        check.fail("results.json: expected a JSON array of results")
        results = None

    if isinstance(results, list) and tickets is not None and articles is not None:
        known_articles = {a.article_id for a in articles}
        before = len(check.errors)
        for index, item in enumerate(results):
            _check_result(index, item, known_articles, check)
        if len(check.errors) == before:
            check.ok(f"all {len(results)} results have exactly the {len(REQUIRED_RESULT_KEYS)} required keys and valid values")

        before = len(check.errors)
        ticket_ids = [t.id for t in tickets]
        result_ids = [r.get("ticket_id") for r in results if isinstance(r, dict) and isinstance(r.get("ticket_id"), str)]
        counts = Counter(result_ids)
        for ticket_id in ticket_ids:
            if counts[ticket_id] == 0:
                check.fail(f"results.json: ticket {ticket_id} has no result")
        for ticket_id, count in counts.items():
            if count > 1:
                check.fail(f"results.json: duplicate results for ticket {ticket_id} ({count})")
        for ticket_id in counts:
            if ticket_id not in set(ticket_ids):
                check.fail(f"results.json: result for unknown ticket {ticket_id!r}")
        if len(check.errors) == before:
            check.ok(f"exactly one result per ticket ({len(ticket_ids)})")

        by_id = {r["ticket_id"]: r for r in results if isinstance(r, dict) and isinstance(r.get("ticket_id"), str)}
        _check_acceptance(by_id, check)

    if not check.errors:
        _check_policy_replay(root, results, parsed.get(DEBUG_FILE), check)
    if not check.errors:
        _check_reproducibility(root, parsed.get(DEBUG_FILE), check)
    return check


def validate(root: Path = REPO_ROOT) -> list[str]:
    """Return a list of error messages; an empty list means every check passed."""
    return _run_checks(Path(root)).errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate results.json and debug_report.json.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="directory holding the inputs and artifacts")
    args = parser.parse_args(argv)
    check = _run_checks(args.root)
    if check.errors:
        for error in check.errors:
            print(f"ERROR: {error}")
        print(f"FAILED: {len(check.errors)} problem(s)")
        return 1
    for message in check.passed:
        print(f"OK: {message}")
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
