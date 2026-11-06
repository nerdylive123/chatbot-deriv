"""Run the support-ticket pipeline and write results.json + debug_report.json.

    python main.py [--mode auto|offline|together] [--retrieval tfidf|embedding]
                   [--data-dir DIR] [--out DIR]

Default mode "auto" uses DeepSeek + Kimi whenever TOGETHER_API_KEY is set (in the
shell or a local .env file) and falls back to the offline components otherwise.
Exit codes: 0 success, 1 invalid input data, 2 configuration error (for
example --mode together without TOGETHER_API_KEY). Secrets are never printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from support_pipeline.env import load_dotenv
from support_pipeline.data import REPO_ROOT, DataError, load_articles, load_examples, load_tickets
from support_pipeline.pipeline import MODES, ConfigError, build_pipeline
from support_pipeline.serialize import write_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify, ground and draft replies for support tickets.")
    parser.add_argument("--mode", choices=MODES, default="auto",
                        help="auto (default: LLMs when TOGETHER_API_KEY is set, else offline), "
                             "offline (no key), or together (requires the key)")
    parser.add_argument("--retrieval", choices=("tfidf", "embedding"), default="tfidf",
                        help="tfidf (default) or embedding (only if installed and cached)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT,
                        help="directory with tickets.json, kb_articles.json, examples.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT, help="output directory for the artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()
    try:
        pipeline, backends, notes = build_pipeline(args.mode, args.retrieval)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for note in notes:
        print(f"note: {note}")

    data_dir: Path = args.data_dir
    try:
        tickets = load_tickets(data_dir / "tickets.json")
        articles = load_articles(data_dir / "kb_articles.json")
        examples = load_examples(data_dir / "examples.json")
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    runs = pipeline.run(tickets, articles, examples)
    paths = write_artifacts(runs, args.out, pipeline.mode, backends)

    print(f"mode={pipeline.mode} retrieval={backends.retrieval} classifier={backends.classifier} "
          f"generator={backends.generator}")
    for run in runs:
        r = run.result
        print(f"{r.ticket_id}\t{r.intent.value}\tescalated={str(r.needs_human_escalation).lower()}")
    print("wrote " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
