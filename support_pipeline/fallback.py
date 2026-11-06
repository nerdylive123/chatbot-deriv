"""Offline classifier and generator: no network, no API key, no keyword rules.

`SimilarityClassifier` votes over the similar labeled examples supplied by the
retriever. `TemplateGenerator` composes a reply from one generic framing per
intent plus the retrieved evidence. Neither reads ticket text for decisions,
and neither has per-ticket logic.
"""

from __future__ import annotations

from .models import Classification, Draft, Evidence, Intent, SimilarExample, Ticket

SUFFICIENT_RELEVANCE = 0.35
CONFLICT_RATIO = 0.6
MAX_CITED = 2
MAX_SIMILAR = 3

REFUSAL_SENTENCE = (
    "I'm sorry, but we can't provide market predictions, trading recommendations, or profit guarantees."
)

GOALS: dict[Intent, str] = {
    Intent.deposit_issue: "Resolve a delayed or missing deposit",
    Intent.password_reset: "Regain access by resetting the account password",
    Intent.withdrawal_issue: "Understand a withdrawal problem or decline",
    Intent.account_lock: "Unlock or recover access to a locked account",
    Intent.trading_advice_request: "Obtain trading advice, which support cannot provide",
    Intent.other: "Understand the request and route it to a support agent",
}

_INTENT_ORDER: dict[Intent, int] = {intent: i for i, intent in enumerate(Intent)}


class SimilarityClassifier:
    """Similarity-weighted vote over the (at most three) nearest labeled examples."""

    name = "similarity_fallback"

    def classify(self, ticket: Ticket, similar: list[SimilarExample]) -> Classification:
        examples = [ex for ex in similar[:MAX_SIMILAR] if ex.score > 0]
        if not ticket.message.strip() or not examples:
            return self._result(Intent.other, 0.0, [], False)

        sums: dict[Intent, float] = {}
        for ex in examples:
            sums[ex.intent] = sums.get(ex.intent, 0.0) + ex.score
        # Round to absorb float noise so exact ties fall back to declaration order.
        ranked = sorted(sums.items(), key=lambda kv: (-round(kv[1], 6), _INTENT_ORDER[kv[0]]))
        winner, winner_sum = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(sums.values())
        conflicting = runner_up > 0 and round(runner_up, 6) >= round(CONFLICT_RATIO * winner_sum, 6)
        certainty = round(winner_sum / total, 3)
        return self._result(winner, certainty, [ex.message for ex in examples], conflicting)

    def _result(self, intent: Intent, certainty: float, evidence: list[str], conflicting: bool) -> Classification:
        return Classification(
            intent=intent,
            certainty=certainty,
            goal=GOALS[intent],
            constraints=[],
            evidence=evidence,
            conflicting_signals=conflicting,
            backend="similarity_fallback",
        )


# One generic framing per intent. Never per ticket; never echoes ticket text.
FRAMINGS: dict[Intent, dict[str, str]] = {
    Intent.deposit_issue: {
        "opening": "Thanks for reaching out about your deposit. Here is some general guidance that may help.",
        "next_step": "If this doesn't resolve it, a support agent can review your case.",
    },
    Intent.password_reset: {
        "opening": "Thanks for contacting us about resetting your password. Here are some general steps to try.",
        "next_step": "If this doesn't resolve it, a support agent can review your case.",
    },
    Intent.withdrawal_issue: {
        "opening": "Thanks for reaching out about your withdrawal. Here is some general information.",
        "next_step": "A support agent can review your account status if you need more help.",
    },
    Intent.account_lock: {
        "opening": "Thanks for letting us know you're having trouble accessing your account. Here is some general guidance.",
        "next_step": "If this doesn't resolve it, a support agent can review your case through a secure channel.",
    },
    Intent.trading_advice_request: {
        "opening": REFUSAL_SENTENCE + " We're happy to help with questions about your account or our services.",
        "next_step": "If you have another question, a support agent can review your case.",
    },
    Intent.other: {
        "opening": "Thanks for contacting us.",
        "next_step": "If this doesn't resolve it, a support agent can review your case.",
    },
}

NO_EVIDENCE_STEP = "A support agent will follow up to help with your request."


def _sentence(text: str) -> str:
    text = " ".join(text.split())
    return text if text.endswith((".", "!", "?")) else text + "."


class TemplateGenerator:
    """Composes a safe reply from a per-intent framing and the selected evidence only."""

    name = "template_fallback"

    def draft(self, ticket: Ticket, classification: Classification, evidence: list[Evidence]) -> Draft:
        framing = FRAMINGS[classification.intent]
        cited: list[Evidence] = []
        if classification.intent is not Intent.other:
            cited = [ev for ev in evidence if ev.relevance > 0][:MAX_CITED]

        parts = [framing["opening"]]
        if cited:
            parts += [f'Based on our help article "{ev.title}": {_sentence(ev.snippet)}' for ev in cited]
            parts.append(framing["next_step"])
        else:
            parts.append(NO_EVIDENCE_STEP)

        return Draft(
            text=" ".join(parts),
            cited_article_ids=[ev.article_id for ev in cited],
            sufficient_evidence=any(ev.relevance >= SUFFICIENT_RELEVANCE for ev in cited),
            backend="template_fallback",
        )


__all__ = [
    "REFUSAL_SENTENCE",
    "SUFFICIENT_RELEVANCE",
    "GOALS",
    "FRAMINGS",
    "SimilarityClassifier",
    "TemplateGenerator",
]
