"""Pipeline orchestration: seven named stages per ticket, deterministic traces.

Work order per run: index articles + examples once, then for each ticket
similar_examples -> classify -> retrieve -> draft (selected evidence only) ->
policy.evaluate -> TicketResult validation. Traces are compact: no prompts, no
candidate lists beyond the selected evidence, no credentials, no timings that
leak into artifacts (duration_ms is excluded on serialization).

Pipeline logic never looks at ticket IDs except to copy them into the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .agents import Classifier, DeepSeekClassifier, Generator, KimiGenerator, Transport
from .fallback import SimilarityClassifier, TemplateGenerator
from .models import (
    STAGES,
    Article,
    Backends,
    Classification,
    Draft,
    Evidence,
    LabeledExample,
    SimilarExample,
    StageTrace,
    Ticket,
    TicketResult,
    TicketRun,
)
from .policy import evaluate
from .retrieval import Retriever, make_retriever
from .serialize import DEBUG_FILE, RESULTS_FILE
from .together_client import ENV_VAR, TogetherTransport, has_together_key

MODES = ("auto", "offline", "together")
REMOTE_BACKENDS = frozenset({"deepseek", "kimi"})


class ConfigError(ValueError):
    """Raised when the requested mode/backends cannot be configured."""


def _retriever_device(retriever: Retriever) -> str:
    pool = getattr(retriever, "pool", None)
    return str(getattr(pool, "device", "cpu")) if pool is not None else "cpu"


def _component_device(actual_backend: str) -> str:
    return "remote" if actual_backend in REMOTE_BACKENDS else "cpu"


def _fallback_reason(component: Any) -> str:
    return str(getattr(component, "last_fallback_reason", "") or "")


def _evidence_view(evidence: Sequence[Evidence], snippet: bool) -> list[dict[str, Any]]:
    items = []
    for ev in evidence:
        item: dict[str, Any] = {"article_id": ev.article_id, "title": ev.title}
        if snippet:
            item["snippet"] = ev.snippet
        item["relevance"] = ev.relevance
        items.append(item)
    return items


class Pipeline:
    def __init__(
        self,
        retriever: Retriever,
        classifier: Classifier,
        generator: Generator,
        mode: str = "offline",
    ) -> None:
        self.retriever = retriever
        self.classifier = classifier
        self.generator = generator
        self.mode = mode
        self._counts: dict[str, int] | None = None

    # ------------------------------------------------------------ run level

    def index(
        self,
        articles: Sequence[Article],
        examples: Sequence[LabeledExample],
        ticket_count: int | None = None,
    ) -> None:
        self.retriever.index(list(articles), list(examples))
        counts = {"articles": len(articles), "examples": len(examples)}
        if ticket_count is not None:
            counts = {"tickets": ticket_count, **counts}
        self._counts = counts

    def run(
        self,
        tickets: Sequence[Ticket],
        articles: Sequence[Article],
        examples: Sequence[LabeledExample],
    ) -> list[TicketRun]:
        seen: set[str] = set()
        for ticket in tickets:
            if ticket.id in seen:
                raise ValueError(f"duplicate ticket id {ticket.id!r}; each ticket needs a unique id")
            seen.add(ticket.id)
        self.index(articles, examples, ticket_count=len(tickets))
        return [self.run_ticket(ticket) for ticket in tickets]

    # ------------------------------------------------------------ ticket level

    def run_ticket(self, ticket: Ticket) -> TicketRun:
        if self._counts is None:
            raise RuntimeError("Pipeline.index() (or run()) must be called before run_ticket()")
        traces: list[StageTrace] = [self._load_trace(), self._index_trace()]

        similar = self.retriever.similar_examples(ticket)
        classification = self.classifier.classify(ticket, similar)
        traces.append(self._classify_trace(ticket, similar, classification))

        evidence = self.retriever.retrieve(ticket)
        traces.append(self._retrieve_trace(ticket, evidence))

        draft = self.generator.draft(ticket, classification, evidence)
        dropped = list(getattr(self.generator, "last_dropped_citations", None) or [])
        traces.append(self._generate_trace(ticket, classification, evidence, draft, dropped))

        # Citations the generator dropped (unknown IDs) still count against
        # grounding: policy sees them and flags ungrounded_claim.
        policy_draft = draft
        if dropped:
            policy_draft = draft.model_copy(
                update={"cited_article_ids": [*draft.cited_article_ids, *dropped]}
            )
        outcome = evaluate(ticket, classification, evidence, policy_draft)
        result = TicketResult(
            ticket_id=ticket.id,
            intent=classification.intent,
            retrieved_articles=[ev.article_id for ev in evidence],
            reply_draft=outcome.reply_draft,
            grounded=outcome.grounded,
            confidence=outcome.confidence,
            needs_human_escalation=outcome.needs_human_escalation,
            safety_flags=list(outcome.safety_flags),
        )
        traces.append(
            StageTrace(
                stage="VALIDATE_OUTPUT",
                summary="Applied deterministic policy and validated the result schema.",
                input={
                    "cited_article_ids": list(policy_draft.cited_article_ids),
                    "sufficient_evidence": draft.sufficient_evidence,
                },
                output={
                    "grounded": result.grounded,
                    "confidence": result.confidence,
                    "needs_human_escalation": result.needs_human_escalation,
                    "safety_flags": list(result.safety_flags),
                    "escalation_reasons": list(outcome.escalation_reasons),
                    "reply_replaced": outcome.reply_draft != draft.text,
                },
                backend="policy",
                device="cpu",
                status="completed",
            )
        )
        traces.append(
            StageTrace(
                stage="WRITE_RESULTS",
                summary="Result queued for the run's JSON artifacts.",
                output={"results_file": RESULTS_FILE, "debug_file": DEBUG_FILE},
                backend="json",
                device="cpu",
                status="completed",
            )
        )
        assert [t.stage for t in traces] == list(STAGES)

        backends = Backends(
            retrieval=self.retriever.name,  # type: ignore[arg-type]
            classifier=classification.backend,
            generator=draft.backend,
        )
        return TicketRun(
            result=result,
            traces=traces,
            escalation_reasons=list(outcome.escalation_reasons),
            confidence_factors=dict(outcome.confidence_factors),
            backends=backends,
        )

    # ------------------------------------------------------------ traces

    def _load_trace(self) -> StageTrace:
        assert self._counts is not None
        return StageTrace(
            stage="LOAD_DATA",
            summary="Loaded tickets, KB articles and labeled examples.",
            output=dict(self._counts),
            backend="json",
            device="cpu",
            status="completed",
        )

    def _index_trace(self) -> StageTrace:
        assert self._counts is not None
        return StageTrace(
            stage="INDEX_KB",
            summary="Indexed KB articles and labeled examples.",
            output={"articles": self._counts["articles"], "examples": self._counts["examples"]},
            backend=self.retriever.name,
            device=_retriever_device(self.retriever),
            status="completed",
        )

    def _classify_trace(
        self, ticket: Ticket, similar: list[SimilarExample], classification: Classification
    ) -> StageTrace:
        reason = _fallback_reason(self.classifier)
        fell_back = bool(reason) or classification.backend != self.classifier.name
        output: dict[str, Any] = {
            "intent": classification.intent.value,
            "certainty": classification.certainty,
            "conflicting_signals": classification.conflicting_signals,
            "goal": classification.goal,
        }
        if fell_back:
            output["fallback_reason"] = reason or f"{self.classifier.name} fell back"
        return StageTrace(
            stage="CLASSIFY_INTENT",
            summary=f"Classified intent as {classification.intent.value}.",
            input={
                "message": ticket.message,
                "similar_examples": [
                    {"intent": ex.intent.value, "score": ex.score, "message": ex.message} for ex in similar
                ],
            },
            output=output,
            backend=classification.backend,
            device=_component_device(classification.backend),
            status="fallback" if fell_back else "completed",
        )

    def _retrieve_trace(self, ticket: Ticket, evidence: list[Evidence]) -> StageTrace:
        return StageTrace(
            stage="RETRIEVE_CONTEXT",
            summary=f"Selected {len(evidence)} help article(s).",
            input={"query": ticket.message},
            output={"evidence": _evidence_view(evidence, snippet=False)},
            backend=self.retriever.name,
            device=_retriever_device(self.retriever),
            status="completed",
        )

    def _generate_trace(
        self,
        ticket: Ticket,
        classification: Classification,
        evidence: list[Evidence],
        draft: Draft,
        dropped: list[str],
    ) -> StageTrace:
        reason = _fallback_reason(self.generator)
        fell_back = bool(reason) or draft.backend != self.generator.name
        output: dict[str, Any] = {
            "reply_draft": draft.text,
            "cited_article_ids": list(draft.cited_article_ids),
            "sufficient_evidence": draft.sufficient_evidence,
        }
        if dropped:
            output["dropped_citations"] = dropped
        if fell_back:
            output["fallback_reason"] = reason or f"{self.generator.name} fell back"
        return StageTrace(
            stage="GENERATE_REPLY",
            summary=f"Drafted a reply citing {len(draft.cited_article_ids)} article(s).",
            input={
                "message": ticket.message,
                "intent": classification.intent.value,
                "goal": classification.goal,
                "constraints": list(classification.constraints),
                "evidence": _evidence_view(evidence, snippet=True),
            },
            output=output,
            backend=draft.backend,
            device=_component_device(draft.backend),
            status="fallback" if fell_back else "completed",
        )


def build_pipeline(
    mode: str = "auto",
    retrieval: str = "tfidf",
    transport: Transport | None = None,
) -> tuple[Pipeline, Backends, list[str]]:
    """Configure a pipeline. Returns (pipeline, configured backends, notes).

    ``transport`` overrides the Together transport (tests inject fakes); the
    key is still required for mode "together" so there is no silent downgrade.
    """
    if mode not in MODES:
        raise ConfigError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    try:
        retriever, note = make_retriever(retrieval)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    notes = [note] if note else []

    classifier: Classifier
    generator: Generator
    if mode == "auto":
        if has_together_key():
            mode = "together"
        else:
            mode = "offline"
            notes.append(f"{ENV_VAR} not found (shell or .env); using the offline fallback")
    if mode == "offline":
        classifier, generator = SimilarityClassifier(), TemplateGenerator()
    else:
        if not has_together_key():
            raise ConfigError(
                f"--mode together needs {ENV_VAR} in the environment; it is not set. "
                "Set it, or run offline with `python main.py`."
            )
        active = transport if transport is not None else TogetherTransport()
        classifier, generator = DeepSeekClassifier(active), KimiGenerator(active)

    backends = Backends(
        retrieval=retriever.name,  # type: ignore[arg-type]
        classifier=classifier.name,  # type: ignore[arg-type]
        generator=generator.name,  # type: ignore[arg-type]
    )
    return Pipeline(retriever, classifier, generator, mode=mode), backends, notes


__all__ = ["ConfigError", "MODES", "Pipeline", "build_pipeline"]
