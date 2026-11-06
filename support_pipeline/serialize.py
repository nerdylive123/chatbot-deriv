"""Deterministic artifact writers for results.json and debug_report.json.

All floats are rounded to 3 decimals, keys keep model order (no sorting), and
files are UTF-8 with LF line endings and a trailing newline.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import Backends, TicketRun

RESULTS_FILE = "results.json"
DEBUG_FILE = "debug_report.json"


def round_floats(obj: Any, ndigits: int = 3) -> Any:
    """Recursively round floats; tuples become lists (JSON arrays)."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits) + 0.0  # + 0.0 turns -0.0 into 0.0
    if isinstance(obj, dict):
        return {key: round_floats(value, ndigits) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(value, ndigits) for value in obj]
    return obj


def dump_json(obj: Any, path: Path | str) -> Path:
    path = Path(path)
    text = json.dumps(round_floats(obj), indent=2, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def _check_unique(runs: Sequence[TicketRun]) -> None:
    seen: set[str] = set()
    for run in runs:
        ticket_id = run.result.ticket_id
        if ticket_id in seen:
            raise ValueError(f"duplicate ticket id {ticket_id!r} in pipeline output")
        seen.add(ticket_id)


def build_results(runs: Sequence[TicketRun]) -> list[dict[str, Any]]:
    _check_unique(runs)
    return [run.result.model_dump(mode="json") for run in runs]


def _retrieved_titles(run: TicketRun) -> list[str]:
    for trace in run.traces:
        if trace.stage == "RETRIEVE_CONTEXT":
            return [item["title"] for item in trace.output.get("evidence", [])]
    return []


def _backends_dict(backends: Backends | dict[str, Any]) -> dict[str, Any]:
    if isinstance(backends, Backends):
        return backends.model_dump(mode="json")
    return Backends.model_validate(backends).model_dump(mode="json")


def build_debug_report(
    runs: Sequence[TicketRun], mode: str, backends: Backends | dict[str, Any]
) -> dict[str, Any]:
    _check_unique(runs)
    tickets = [
        {
            "ticket_id": run.result.ticket_id,
            "predicted_intent": run.result.intent.value,
            "retrieved_article_titles": _retrieved_titles(run),
            "escalated": run.result.needs_human_escalation,
            "escalation_reasons": list(run.escalation_reasons),
            "confidence_factors": dict(run.confidence_factors),
            "backends": run.backends.model_dump(mode="json"),
        }
        for run in runs
    ]
    traces = {
        run.result.ticket_id: [
            trace.model_dump(mode="json", exclude={"duration_ms"}) for trace in run.traces
        ]
        for run in runs
    }
    return {"mode": mode, "backends": _backends_dict(backends), "tickets": tickets, "traces": traces}


def write_artifacts(
    runs: Sequence[TicketRun],
    out_dir: Path | str,
    mode: str,
    backends: Backends | dict[str, Any],
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = build_results(runs)
    report = build_debug_report(runs, mode, backends)
    return [dump_json(results, out / RESULTS_FILE), dump_json(report, out / DEBUG_FILE)]


__all__ = [
    "DEBUG_FILE",
    "RESULTS_FILE",
    "build_debug_report",
    "build_results",
    "dump_json",
    "round_floats",
    "write_artifacts",
]
