"""Opt-in smoke test: one real DeepSeek classify and one real Kimi draft on T1.

    python scripts/smoke_together.py

Needs TOGETHER_API_KEY (shell or .env); no extra packages. Without
the key it prints "SKIPPED: TOGETHER_API_KEY not set" and exits 0. It makes at
most two bounded requests and never prints the key. Not run by pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from support_pipeline.env import load_dotenv  # noqa: E402
from support_pipeline.together_client import ENV_VAR, has_together_key  # noqa: E402


def main() -> int:
    load_dotenv()
    if not has_together_key():
        print(f"SKIPPED: {ENV_VAR} not set")
        return 0

    from support_pipeline.agents import DeepSeekClassifier, KimiGenerator  # noqa: PLC0415
    from support_pipeline.data import load_articles, load_examples, load_tickets  # noqa: PLC0415
    from support_pipeline.models import Classification, Draft  # noqa: PLC0415
    from support_pipeline.retrieval import TfidfRetriever  # noqa: PLC0415
    from support_pipeline.together_client import TogetherTransport  # noqa: PLC0415

    tickets = load_tickets(REPO_ROOT / "tickets.json")
    ticket = next((t for t in tickets if t.id == "T1"), tickets[0])
    retriever = TfidfRetriever()
    retriever.index(load_articles(REPO_ROOT / "kb_articles.json"), load_examples(REPO_ROOT / "examples.json"))
    similar = retriever.similar_examples(ticket)
    evidence = retriever.retrieve(ticket)

    transport = TogetherTransport()
    ok = True

    classifier = DeepSeekClassifier(transport)
    classification = classifier.classify(ticket, similar)
    used = classification.backend
    print(f"classify: model={classifier.MODEL} backend_used={used}")
    if classifier.last_fallback_reason:
        print(f"  fallback_reason={classifier.last_fallback_reason}")
    valid = used == "deepseek" and isinstance(classification, Classification)
    print(f"  validated={str(valid).lower()} intent={classification.intent.value} certainty={classification.certainty}")
    ok &= valid

    generator = KimiGenerator(transport)
    draft = generator.draft(ticket, classification, evidence)
    used = draft.backend
    print(f"draft: model={generator.MODEL} backend_used={used}")
    if generator.last_fallback_reason:
        print(f"  fallback_reason={generator.last_fallback_reason}")
    if generator.last_dropped_citations:
        print(f"  dropped_citations={generator.last_dropped_citations}")
    valid = used == "kimi" and isinstance(draft, Draft)
    print(f"  validated={str(valid).lower()} cited={draft.cited_article_ids} "
          f"sufficient_evidence={str(draft.sufficient_evidence).lower()}")
    print(f"  text={draft.text!r}")
    ok &= valid

    print("PASSED" if ok else "FAILED: at least one call fell back to the offline component")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
