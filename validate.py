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
from support_pipeline.data import DataError, load_articles, load_tickets
from support_pipeline.models import REQUIRED_RESULT_KEYS, SAFETY_FLAGS, Intent, TicketResult
from support_pipeline.pipeline import MODES
from support_pipeline.policy import has_refusal, has_unsafe_content
from support_pipeline.retrieval import _embedding_unavailable_reason
from support_pipeline.serialize import DEBUG_FILE, RESULTS_FILE
from support_pipeline.together_client import ENV_VAR, has_together_key

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
    label = f"results.json entry {index} ({item.get('ticket_id', '?')})"
    keys = set(item)
    missing = [k for k in REQUIRED_RESULT_KEYS if k not in keys]
    extra = sorted(keys - set(REQUIRED_RESULT_KEYS))
    if missing:
        check.fail(f"{label}: missing keys {missing}")
    if extra:
        check.fail(f"{label}: extra keys {extra}")

    intent = item.get("intent")
    if "intent" in item and intent not in INTENTS:
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
    if mode == "together" and not has_together_key():
        check.fail(
            f"results.json was generated in together mode; rerun `python main.py` (offline) "
            f"or provide {ENV_VAR}"
        )
        return
    if retrieval == "embedding":
        reason = embedding_unavailable_reason()
        if reason:
            check.fail(
                "results.json was generated with the embedding retrieval backend, which is not "
                f"available here ({reason}); rerun `python main.py` (tfidf) or install and cache "
                "the local models (requirements-local-models.txt)"
            )
            return

    with tempfile.TemporaryDirectory(prefix="validate-rerun-") as tmp:
        argv = ["--mode", mode, "--retrieval", retrieval, "--data-dir", str(root), "--out", tmp]
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            code = cli.main(argv)
        if code != 0:
            detail = sink.getvalue().strip().splitlines()
            check.fail(f"reproducibility: rerun `python main.py` failed (exit {code}): "
                       f"{detail[-1] if detail else 'no output'}")
            return
        mismatched = False
        for name in (RESULTS_FILE, DEBUG_FILE):
            committed = (root / name).read_bytes()
            rerun = (Path(tmp) / name).read_bytes()
            if committed != rerun:
                mismatched = True
                check.fail(f"{name}: not reproducible by a {mode}/{retrieval} rerun; "
                           f"{_first_difference(committed, rerun)}")
        if not mismatched:
            check.ok(f"rerun in {mode}/{retrieval} mode reproduces results.json and debug_report.json byte-for-byte")


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
        result_ids = [r.get("ticket_id") for r in results if isinstance(r, dict)]
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

        by_id = {r["ticket_id"]: r for r in results if isinstance(r, dict) and "ticket_id" in r}
        _check_acceptance(by_id, check)

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
