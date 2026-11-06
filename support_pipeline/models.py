"""Shared typed contract for every pipeline stage.

Other modules import these models; they never redefine them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    deposit_issue = "deposit_issue"
    password_reset = "password_reset"
    withdrawal_issue = "withdrawal_issue"
    account_lock = "account_lock"
    trading_advice_request = "trading_advice_request"
    other = "other"


REQUIRED_RESULT_KEYS: tuple[str, ...] = (
    "ticket_id",
    "intent",
    "retrieved_articles",
    "reply_draft",
    "grounded",
    "confidence",
    "needs_human_escalation",
    "safety_flags",
)

STAGES: tuple[str, ...] = (
    "LOAD_DATA",
    "INDEX_KB",
    "CLASSIFY_INTENT",
    "RETRIEVE_CONTEXT",
    "GENERATE_REPLY",
    "VALIDATE_OUTPUT",
    "WRITE_RESULTS",
)

SAFETY_FLAGS: tuple[str, ...] = (
    "advice_request",
    "insufficient_grounding",
    "policy_sensitive",
    "low_confidence",
    "account_specific_request",
    "conflicting_signals",
    "ungrounded_claim",
)


class Ticket(BaseModel):
    id: str
    message: str
    language: str = "en"


class Article(BaseModel):
    article_id: str
    title: str
    body: str


class LabeledExample(BaseModel):
    message: str
    intent: Intent
    language: str = "en"


class SimilarExample(BaseModel):
    message: str
    intent: Intent
    score: float = Field(ge=0, le=1)


class Classification(BaseModel):
    intent: Intent
    certainty: float = Field(ge=0, le=1)
    goal: str
    constraints: list[str]
    evidence: list[str]
    conflicting_signals: bool = False
    backend: Literal["deepseek", "similarity_fallback"]


class Evidence(BaseModel):
    article_id: str
    title: str
    snippet: str
    lexical_score: float  # tfidf cosine, or dense cosine for the embedding backend
    rerank_score: float | None = None  # raw cross-encoder logit; None for tfidf
    relevance: float = Field(ge=0, le=1)


class Draft(BaseModel):
    text: str
    cited_article_ids: list[str]
    sufficient_evidence: bool
    backend: Literal["kimi", "template_fallback"]


class TicketResult(BaseModel):
    """Final per-ticket output. Field order is the required schema order."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    intent: Intent
    retrieved_articles: list[str]
    reply_draft: str
    grounded: bool
    confidence: float = Field(ge=0, le=1)
    needs_human_escalation: bool
    safety_flags: list[str]


class StageTrace(BaseModel):
    stage: str
    summary: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    backend: str = "pending"
    device: str = "none"
    status: Literal["pending", "completed", "fallback", "skipped", "failed"] = "pending"
    duration_ms: float = Field(default=0, ge=0)  # excluded from serialized artifacts


class Backends(BaseModel):
    retrieval: Literal["tfidf", "embedding"]
    classifier: Literal["similarity_fallback", "deepseek"]
    generator: Literal["template_fallback", "kimi"]


class TicketRun(BaseModel):
    result: TicketResult
    traces: list[StageTrace]
    escalation_reasons: list[str]
    confidence_factors: dict[str, float]
    backends: Backends


__all__ = [
    "Intent",
    "REQUIRED_RESULT_KEYS",
    "STAGES",
    "SAFETY_FLAGS",
    "Ticket",
    "Article",
    "LabeledExample",
    "SimilarExample",
    "Classification",
    "Evidence",
    "Draft",
    "TicketResult",
    "StageTrace",
    "Backends",
    "TicketRun",
]
