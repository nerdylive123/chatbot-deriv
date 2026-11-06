"""Deterministic policy / validation stage (VALIDATE_OUTPUT).

``evaluate`` decides grounding, confidence, safety flags and escalation from the
classification, the selected evidence and the draft. It is pure: no I/O, no
randomness, no dependence on ticket IDs. Model text never sets its own safety
status; this module does.

The regex lists below are **output checks** on the draft text. They are not
intent rules and are never used to choose an intent. They are deliberately
conservative (phrase based), so they may over-escalate.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from .models import SAFETY_FLAGS, Classification, Draft, Evidence, Intent, Ticket

RELEVANCE_THRESHOLD = 0.35
LOW_CONFIDENCE = 0.45
PENALTY_INSUFFICIENT = 0.35
PENALTY_CONFLICT = 0.20
PENALTY_UNGROUNDED = 0.40

POLICY_SENSITIVE_INTENTS = frozenset({Intent.withdrawal_issue, Intent.account_lock})

# Defined locally on purpose (policy must not depend on the fallback generator).
SAFE_REFUSAL = (
    "I'm sorry, but we can't provide market predictions, trading recommendations, "
    "or profit guarantees. Our help center has general educational resources about "
    "how trading works."
)

# One plain-English reason per escalation trigger, keyed by flag.
REASONS: dict[str, str] = {
    "insufficient_grounding": "No help article matched the request closely enough to ground a reply.",
    "policy_sensitive": "Withdrawal and account-lock cases are policy-sensitive and need a human to check account status.",
    "low_confidence": "Overall confidence is below the 0.45 review threshold.",
    "account_specific_request": "The customer asks about details of their own account or transaction, which the help articles cannot answer.",
    "conflicting_signals": "The ticket shows signals for more than one intent.",
    "ungrounded_claim": "The draft contained a claim or citation not supported by the selected help articles.",
}

ESCALATION_FLAGS = frozenset(REASONS)

_I = re.IGNORECASE

# --- Output checks (draft text) -------------------------------------------
# Checked against the draft with refusal sentences removed, so a polite refusal
# such as "we can't provide recommendations on what to buy" is not flagged.
_MARKET_PREDICTION = [
    r"\bwill (go up|rise|fall|drop|increase)\b",
    r"\bis going to (rise|go up|fall)\b",
    r"\bprice target\b",
]
_PROFIT_GUARANTEE = [
    r"\bguarantee(d)? (a )?profits?\b",
    r"\brisk[- ]free\b",
    r"\bsure profit\b",
]
_ASSET_RECOMMENDATION = [
    r"\byou should (buy|sell)\b",
    r"\bbuy\b",
    r"\bsell\b",
]
_OUTCOME_PROMISE = [
    r"\bwill be (approved|processed|credited|released)\b",
    r"\bwithin \d+ (minutes?|hours?|days?)\b",
    r"\bguaranteed\b",
]
_UNSAFE_PATTERNS = [
    re.compile(p, _I)
    for p in _MARKET_PREDICTION + _PROFIT_GUARANTEE + _ASSET_RECOMMENDATION + _OUTCOME_PROMISE
]

# Fabricated account specifics: flagged only when the matched text does not
# appear in any selected evidence snippet.
_ACCOUNT_SPECIFICS = [
    re.compile(p, _I)
    for p in (
        r"\$\s?\d[\d,]*(\.\d+)?",
        r"\b\d+(\.\d+)?\s?(usd|eur)\b",
        r"\byour (balance|account) (is|shows|has)\b",
    )
]

# Refusal language required for trading-advice requests.
_REFUSAL_APOLOGY = re.compile(r"\b(sorry|unable|can(no|')t|cannot|won't|not able)\b", _I)
_REFUSAL_TOPIC = re.compile(r"(advice|recommend|predict|guarantee)", _I)

# Refusal clauses (a refusal marker such as "can't predict") are exempt from the
# unsafe-phrase scan; sentences are split into clauses first so nothing after a
# semicolon, colon, contrast word, or comma splice inherits that exemption.
_REFUSAL_MARKER = re.compile(
    r"\b(can(no|')t|can not|unable to|won't|not able to)\s+(provide|give|offer|share|make|predict|forecast|recommend|suggest|advise|guarantee|tell)\b", _I
)
_CLAUSE_SPLIT = re.compile(
    r"[;:]|\b(?:but|however|though|although|still|yet|instead)\b"
    r"|,\s*(?=(?:you|we|i|just|so|then|go|try)\b)",
    _I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# --- account_specific_request heuristic (ticket + classifier text) ---------
# Fires when the ticket (or the classifier's goal/constraints) asks support to
# look up or report this customer's own account data, which the KB cannot do:
#   * an explicit lookup request ("check my account", "look up my transaction"),
#   * a request to state the account balance ("tell me my balance",
#     "what is my exact balance", "how much money do I have"),
#   * a transaction / reference / account identifier ("transaction id",
#     "reference number", "account 12345", any run of 5+ digits),
#   * "why is my account <status>" questions.
# Merely describing a symptom ("my balance still shows zero") does not fire.
_ACCOUNT_SPECIFIC = [
    re.compile(p, _I)
    for p in (
        r"\b(check|look ?up|look into|verify|review)\s+(my|our)\s+(account|balance|transaction|withdrawal|deposit|payment)",
        r"\b(tell|show|give)\s+me\s+(my|the)\s+(exact\s+|current\s+)?(balance|account\s+status|transaction)",
        r"\bwhat\s+is\s+my\s+(exact\s+|current\s+)?(balance|account\s+status)",
        r"\bhow\s+much\s+(money\s+)?(do\s+i\s+have|is\s+in\s+my\s+account)",
        r"\b(transaction|reference|ref|order)\s*(id|number|no\.?|#)",
        r"\baccount\s*(number|no\.?|#)?\s*\d{3,}",
        r"\b\d{5,}\b",
        r"\bwhy\s+is\s+my\s+account\s+\w+",
        r"\btransaction\s+id\b",
    )
]


class PolicyOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    reply_draft: str
    grounded: bool
    confidence: float
    needs_human_escalation: bool
    safety_flags: list[str]
    escalation_reasons: list[str]
    confidence_factors: dict[str, float]


def _normalize(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def _clauses(sentence: str) -> list[str]:
    """Split a sentence at ; : contrast words and comma splices, so a refusal clause
    can't shield advice that follows it ("I can't give advice; you should buy XYZ")."""
    return [c for c in _CLAUSE_SPLIT.split(sentence) if c and c.strip()]


def _strip_refusal_sentences(text: str) -> str:
    """Drop only the refusal clauses themselves; every other clause is still checked."""
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(_normalize(text)):
        kept.extend(c for c in _clauses(sentence) if not _REFUSAL_MARKER.search(c))
    return " ".join(kept)


def has_unsafe_content(text: str, evidence: list[Evidence]) -> bool:
    """Output check: does the draft contain an unsupported or unsafe claim?"""
    checked = _strip_refusal_sentences(text)
    if any(p.search(checked) for p in _UNSAFE_PATTERNS):
        return True
    snippets = " ".join(_normalize(e.snippet).lower() for e in evidence)
    for pattern in _ACCOUNT_SPECIFICS:
        for match in pattern.finditer(checked):
            if match.group(0).lower() not in snippets:
                return True
    return False


def has_refusal(text: str) -> bool:
    text = _normalize(text)
    return bool(_REFUSAL_APOLOGY.search(text) and _REFUSAL_TOPIC.search(text))


def is_account_specific_request(ticket: Ticket, classification: Classification) -> bool:
    """Documented heuristic; see the ``_ACCOUNT_SPECIFIC`` comment above."""
    texts = [ticket.message, classification.goal, *classification.constraints]
    return any(p.search(_normalize(t)) for t in texts for p in _ACCOUNT_SPECIFIC)


def evaluate(
    ticket: Ticket,
    classification: Classification,
    evidence: list[Evidence],
    draft: Draft,
) -> PolicyOutcome:
    flags: set[str] = set()
    reply = draft.text

    best_relevance = max((e.relevance for e in evidence), default=0.0)
    evidence_ids = {e.article_id for e in evidence}

    insufficient = (
        not evidence
        or best_relevance < RELEVANCE_THRESHOLD
        or not draft.sufficient_evidence
        or not draft.cited_article_ids  # an uncited reply can't be traced back to the KB
    )
    if insufficient:
        flags.add("insufficient_grounding")

    unsafe = has_unsafe_content(reply, evidence)
    ungrounded = unsafe or any(cid not in evidence_ids for cid in draft.cited_article_ids)

    if classification.intent == Intent.trading_advice_request:
        flags.add("advice_request")
        if unsafe or not has_refusal(reply):
            reply = SAFE_REFUSAL
            ungrounded = True

    if ungrounded:
        flags.add("ungrounded_claim")
    if classification.intent in POLICY_SENSITIVE_INTENTS:
        flags.add("policy_sensitive")
    if is_account_specific_request(ticket, classification):
        flags.add("account_specific_request")
    if classification.conflicting_signals:
        flags.add("conflicting_signals")

    p_insufficient = PENALTY_INSUFFICIENT if insufficient else 0.0
    p_conflict = PENALTY_CONFLICT if classification.conflicting_signals else 0.0
    p_ungrounded = PENALTY_UNGROUNDED if ungrounded else 0.0
    raw = 0.5 * classification.certainty + 0.5 * best_relevance
    confidence = round(min(1.0, max(0.0, raw - p_insufficient - p_conflict - p_ungrounded)), 3)
    if confidence < LOW_CONFIDENCE:
        flags.add("low_confidence")

    safety_flags = [f for f in SAFETY_FLAGS if f in flags]
    escalation_reasons = [REASONS[f] for f in safety_flags if f in ESCALATION_FLAGS]

    grounded = (
        bool(evidence)
        and bool(draft.cited_article_ids)
        and all(cid in evidence_ids for cid in draft.cited_article_ids)
        and draft.sufficient_evidence
        and best_relevance >= RELEVANCE_THRESHOLD
        and not ungrounded
    )

    return PolicyOutcome(
        reply_draft=reply,
        grounded=grounded,
        confidence=confidence,
        needs_human_escalation=bool(escalation_reasons),
        safety_flags=safety_flags,
        escalation_reasons=escalation_reasons,
        confidence_factors={
            "classifier_certainty": round(classification.certainty, 3),
            "best_relevance": round(best_relevance, 3),
            "penalty_insufficient_evidence": round(p_insufficient, 3),
            "penalty_conflicting_signals": round(p_conflict, 3),
            "penalty_ungrounded_claim": round(p_ungrounded, 3),
            "confidence": confidence,
        },
    )


__all__ = [
    "RELEVANCE_THRESHOLD",
    "LOW_CONFIDENCE",
    "PENALTY_INSUFFICIENT",
    "PENALTY_CONFLICT",
    "PENALTY_UNGROUNDED",
    "SAFE_REFUSAL",
    "REASONS",
    "PolicyOutcome",
    "evaluate",
    "has_refusal",
    "has_unsafe_content",
    "is_account_specific_request",
]
