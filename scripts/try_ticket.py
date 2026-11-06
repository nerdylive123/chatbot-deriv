"""Try ticket text against the pipeline from the terminal.

    python scripts/try_ticket.py "My withdrawal was declined"   # one ticket
    python scripts/try_ticket.py                                 # interactive prompt
    python scripts/try_ticket.py --samples                       # the supplied tickets.json
    python scripts/try_ticket.py --file my_tickets.json          # your own ticket file
    python scripts/try_ticket.py --json "..."                    # machine-readable output

Uses the supplied kb_articles.json and examples.json. Offline by default (no key,
no downloads); pass --mode together to use DeepSeek/Kimi if TOGETHER_API_KEY is set.
Never writes results.json or debug_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from support_pipeline.data import DataError, load_articles, load_examples, load_tickets  # noqa: E402
from support_pipeline.models import Ticket, TicketRun  # noqa: E402
from support_pipeline.pipeline import MODES, ConfigError, build_pipeline  # noqa: E402
from support_pipeline.serialize import round_floats  # noqa: E402


def _as_dict(run: TicketRun) -> dict:
    return round_floats({
        "result": run.result.model_dump(mode="json"),
        "escalation_reasons": run.escalation_reasons,
        "confidence_factors": run.confidence_factors,
        "backends": run.backends.model_dump(mode="json"),
        "traces": [t.model_dump(mode="json", exclude={"duration_ms"}) for t in run.traces],
    })


def _print_human(run: TicketRun) -> None:
    data = _as_dict(run)
    print("=" * 72)
    print(json.dumps(data["result"], indent=2, ensure_ascii=False))
    print("\nEscalation reasons:")
    for reason in data["escalation_reasons"] or ["(none, not escalated)"]:
        print(f"  - {reason}")
    print("\nConfidence factors:", json.dumps(data["confidence_factors"]))
    print("Backends:", json.dumps(data["backends"]))
    print("\nStage trace:")
    for trace in data["traces"]:
        print(f"  {trace['stage']:<17} {trace['status']:<9} {trace['backend']:<20} {trace['summary']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ad-hoc tickets through the pipeline.")
    parser.add_argument("message", nargs="*", help="ticket text (omit for an interactive prompt)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--samples", action="store_true", help="run the supplied tickets.json")
    source.add_argument("--file", type=Path, help="run a tickets JSON file (same schema as tickets.json)")
    parser.add_argument("--mode", choices=MODES, default="offline")
    parser.add_argument("--retrieval", choices=("tfidf", "embedding"), default="tfidf")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a readable summary")
    args = parser.parse_args(argv)

    try:
        pipeline, _backends, notes = build_pipeline(args.mode, args.retrieval)
        pipeline.index(load_articles(), load_examples())
        if args.samples:
            tickets = load_tickets()
        elif args.file:
            tickets = load_tickets(args.file)
        else:
            tickets = []
    except (ConfigError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, ConfigError) else 1
    for note in notes:
        print(f"note: {note}", file=sys.stderr)

    def emit(ticket: Ticket) -> None:
        run = pipeline.run_ticket(ticket)
        if args.json:
            print(json.dumps(_as_dict(run), indent=2, ensure_ascii=False))
        else:
            _print_human(run)

    if tickets:
        for ticket in tickets:
            emit(ticket)
        return 0
    if args.message:
        emit(Ticket(id="CLI1", message=" ".join(args.message)))
        return 0

    print("Type a ticket message and press Enter (empty line or Ctrl+C to quit).")
    count = 0
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line.strip():
            return 0
        count += 1
        emit(Ticket(id=f"CLI{count}", message=line))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
