"""Tests for the deterministic policy / validation stage."""

from __future__ import annotations

import pytest

from support_pipeline.models import (
    SAFETY_FLAGS,
    Classification,
    Draft,
    Evidence,
    Intent,
    Ticket,
)
from support_pipeline import policy
from support_pipeline.policy import (
    has_refusal,
    has_unsafe_content,
    LOW_CONFIDENCE,
    PENALTY_CONFLICT,
    PENALTY_INSUFFICIENT,
    PENALTY_UNGROUNDED,
    RELEVANCE_THRESHOLD,
    SAFE_REFUSAL,
    PolicyOutcome,
    evaluate,
)

FACTOR_KEYS = {
    "classifier_certainty",
    "best_relevance",
    "penalty_insufficient_evidence",
    "penalty_conflicting_signals",
    "penalty_ungrounded_claim",
    "confidence",
}


def ticket(message: str = "I forgot my password and I am not receiving the reset email.") -> Ticket:
    return Ticket(id="X1", message=message)


def clf(
    intent: Intent = Intent.password_reset,
    certainty: float = 0.9,
    conflicting: bool = False,
    goal: str = "Regain access to the account",
    constraints: list[str] | None = None,
) -> Classification:
    return Classification(
        intent=intent,
        certainty=certainty,
        goal=goal,
        constraints=constraints or [],
        evidence=[],
        conflicting_signals=conflicting,
        backend="similarity_fallback",
    )


def ev(article_id: str = "A2", relevance: float = 0.8, snippet: str = "Check spam folders and retry.") -> Evidence:
    return Evidence(
        article_id=article_id,
        title=f"Title {article_id}",
        snippet=snippet,
        lexical_score=relevance,
        relevance=relevance,
    )


def draft(
    text: str = "Please check your spam folder and confirm your registered email address.",
    cited: list[str] | None = None,
    sufficient: bool = True,
) -> Draft:
    return Draft(
        text=text,
        cited_article_ids=["A2"] if cited is None else cited,
        sufficient_evidence=sufficient,
        backend="template_fallback",
    )


def run(**kw) -> PolicyOutcome:
    return evaluate(
        kw.get("ticket", ticket()),
        kw.get("classification", clf()),
        kw.get("evidence", [ev()]),
        kw.get("draft", draft()),
    )


# --- constants -------------------------------------------------------------


def test_constants():
    assert RELEVANCE_THRESHOLD == 0.35
    assert LOW_CONFIDENCE == 0.45
    assert PENALTY_INSUFFICIENT == 0.35
    assert PENALTY_CONFLICT == 0.20
    assert PENALTY_UNGROUNDED == 0.40


# --- clean baseline --------------------------------------------------------


def test_clean_case_has_no_flags_and_is_grounded():
    out = run()
    assert out.safety_flags == []
    assert out.escalation_reasons == []
    assert out.needs_human_escalation is False
    assert out.grounded is True
    assert out.reply_draft == draft().text
    assert out.confidence == round(0.5 * 0.9 + 0.5 * 0.8, 3)


# --- formula, rounding, factors --------------------------------------------


def test_confidence_formula_and_rounding():
    out = run(classification=clf(certainty=0.6667), evidence=[ev(relevance=0.4444), ev("A1", 0.3)])
    assert out.confidence == round(0.5 * 0.6667 + 0.5 * 0.4444, 3) == 0.556
    assert set(out.confidence_factors) == FACTOR_KEYS
    assert out.confidence_factors["classifier_certainty"] == 0.667
    assert out.confidence_factors["best_relevance"] == 0.444
    assert out.confidence_factors["penalty_insufficient_evidence"] == 0.0
    assert out.confidence_factors["penalty_conflicting_signals"] == 0.0
    assert out.confidence_factors["penalty_ungrounded_claim"] == 0.0
    assert out.confidence_factors["confidence"] == out.confidence


def test_best_relevance_is_max_of_evidence():
    out = run(evidence=[ev("A1", 0.4), ev("A2", 0.7), ev("A3", 0.5)], draft=draft(cited=["A2"]))
    assert out.confidence_factors["best_relevance"] == 0.7


def test_confidence_clamped_at_zero():
    out = run(
        classification=clf(certainty=0.1, conflicting=True),
        evidence=[],
        draft=draft(text="Your withdrawal will be approved.", cited=[], sufficient=False),
    )
    assert out.confidence == 0.0
    assert out.confidence_factors["confidence"] == 0.0


def test_confidence_clamped_at_one():
    out = run(classification=clf(certainty=1.0), evidence=[ev(relevance=1.0)])
    assert out.confidence == 1.0


def test_penalty_insufficient():
    out = run(draft=draft(sufficient=False))
    assert out.confidence_factors["penalty_insufficient_evidence"] == 0.35
    assert out.confidence == round(0.5 * 0.9 + 0.5 * 0.8 - 0.35, 3)


def test_penalty_conflict():
    out = run(classification=clf(conflicting=True))
    assert out.confidence_factors["penalty_conflicting_signals"] == 0.2
    assert out.confidence == round(0.5 * 0.9 + 0.5 * 0.8 - 0.2, 3)


def test_penalty_ungrounded():
    out = run(draft=draft(cited=["A9"]))
    assert out.confidence_factors["penalty_ungrounded_claim"] == 0.4
    assert out.confidence == round(0.5 * 0.9 + 0.5 * 0.8 - 0.4, 3)


# --- advice_request --------------------------------------------------------


def advice_ticket() -> Ticket:
    return ticket("Can you tell me which asset will go up today so I can make profit?")


def test_advice_request_flag_fires_with_refusal_kept():
    refusal = "Sorry, we can't provide trading advice or recommendations."
    out = run(
        ticket=advice_ticket(),
        classification=clf(Intent.trading_advice_request),
        evidence=[ev("A5", 0.6)],
        draft=draft(text=refusal, cited=["A5"]),
    )
    assert out.safety_flags == ["advice_request"]
    assert out.reply_draft == refusal
    assert out.needs_human_escalation is False  # advice_request alone does not escalate
    assert out.escalation_reasons == []


def test_advice_request_does_not_fire_for_other_intents():
    assert "advice_request" not in run().safety_flags


def test_missing_refusal_replaced_with_safe_refusal():
    out = run(
        ticket=advice_ticket(),
        classification=clf(Intent.trading_advice_request),
        evidence=[ev("A5", 0.6)],
        draft=draft(text="Here is some general information about markets.", cited=["A5"]),
    )
    assert out.reply_draft == SAFE_REFUSAL
    assert "advice_request" in out.safety_flags
    assert "ungrounded_claim" in out.safety_flags
    assert out.grounded is False
    assert out.needs_human_escalation is True


def test_unsafe_advice_draft_replaced_even_with_refusal():
    out = run(
        ticket=advice_ticket(),
        classification=clf(Intent.trading_advice_request),
        evidence=[ev("A5", 0.6)],
        draft=draft(
            text="Sorry, I can't give advice. But gold will go up today, so you should buy it.",
            cited=["A5"],
        ),
    )
    assert out.reply_draft == SAFE_REFUSAL
    assert "ungrounded_claim" in out.safety_flags


def test_safe_refusal_itself_passes_checks():
    out = run(
        ticket=advice_ticket(),
        classification=clf(Intent.trading_advice_request),
        evidence=[ev("A5", 0.6)],
        draft=draft(text=SAFE_REFUSAL, cited=["A5"]),
    )
    assert out.reply_draft == SAFE_REFUSAL
    assert out.safety_flags == ["advice_request"]


def test_refusal_sentence_with_buy_and_recommendations_not_flagged():
    text = (
        "I'm sorry, but we can't provide recommendations on what to buy or sell. "
        "Our help center explains how trading works."
    )
    out = run(
        ticket=advice_ticket(),
        classification=clf(Intent.trading_advice_request),
        evidence=[ev("A5", 0.6)],
        draft=draft(text=text, cited=["A5"]),
    )
    assert out.reply_draft == text
    assert "ungrounded_claim" not in out.safety_flags


# --- insufficient_grounding ------------------------------------------------


def test_insufficient_grounding_no_evidence():
    out = run(evidence=[], draft=draft(cited=[]))
    assert "insufficient_grounding" in out.safety_flags
    assert out.grounded is False
    assert out.confidence_factors["best_relevance"] == 0.0


def test_insufficient_grounding_draft_says_insufficient():
    out = run(draft=draft(sufficient=False))
    assert "insufficient_grounding" in out.safety_flags
    assert out.grounded is False


def test_relevance_threshold_boundary():
    below = run(evidence=[ev(relevance=0.349)])
    at = run(evidence=[ev(relevance=0.35)])
    assert "insufficient_grounding" in below.safety_flags
    assert below.grounded is False
    assert "insufficient_grounding" not in at.safety_flags
    assert at.grounded is True


# --- policy_sensitive ------------------------------------------------------


@pytest.mark.parametrize("intent", [Intent.withdrawal_issue, Intent.account_lock])
def test_policy_sensitive_fires(intent):
    out = run(classification=clf(intent))
    assert "policy_sensitive" in out.safety_flags
    assert out.needs_human_escalation is True


@pytest.mark.parametrize("intent", [Intent.deposit_issue, Intent.password_reset, Intent.other])
def test_policy_sensitive_does_not_fire(intent):
    assert "policy_sensitive" not in run(classification=clf(intent)).safety_flags


# --- low_confidence --------------------------------------------------------


def test_low_confidence_boundary():
    # 0.5*0.5 + 0.5*0.4 = 0.45 -> not low; 0.5*0.5 + 0.5*0.398 = 0.449 -> low
    at = run(classification=clf(certainty=0.5), evidence=[ev(relevance=0.4)])
    below = run(classification=clf(certainty=0.5), evidence=[ev(relevance=0.398)])
    assert at.confidence == 0.45
    assert "low_confidence" not in at.safety_flags
    assert at.needs_human_escalation is False
    assert below.confidence == 0.449
    assert "low_confidence" in below.safety_flags
    assert below.needs_human_escalation is True


# --- account_specific_request ---------------------------------------------


def test_account_specific_request_fires():
    out = run(ticket=ticket("Can you check my account 12345 and tell me my exact balance?"))
    assert "account_specific_request" in out.safety_flags
    assert out.needs_human_escalation is True


def test_account_specific_request_from_classifier_constraints():
    out = run(classification=clf(constraints=["user wants their transaction id status checked"]))
    assert "account_specific_request" in out.safety_flags


def test_account_specific_request_does_not_fire_on_password_ticket():
    assert "account_specific_request" not in run().safety_flags


@pytest.mark.parametrize(
    "message",
    [
        "I deposited by bank card two hours ago but my balance still shows zero. What should I check?",
        "I forgot my password and I am not receiving the reset email.",
        "Can you tell me which asset will go up today so I can make profit?",
    ],
)
def test_account_specific_request_not_on_sample_tickets(message):
    assert policy.is_account_specific_request(ticket(message), clf()) is False


# --- conflicting_signals ---------------------------------------------------


def test_conflicting_signals_fires():
    out = run(classification=clf(conflicting=True))
    assert "conflicting_signals" in out.safety_flags
    assert out.needs_human_escalation is True


def test_conflicting_signals_does_not_fire():
    assert "conflicting_signals" not in run().safety_flags


# --- ungrounded_claim ------------------------------------------------------


def test_cited_unknown_id_is_ungrounded():
    out = run(draft=draft(cited=["A2", "A9"]))
    assert "ungrounded_claim" in out.safety_flags
    assert out.grounded is False
    assert out.needs_human_escalation is True


@pytest.mark.parametrize(
    "text",
    [
        "Your withdrawal will be approved within 24 hours.",
        "The funds will be credited soon.",
        "Bitcoin is going to rise this week.",
        "This is a risk-free strategy.",
        "We guarantee a profit.",
        "You should buy EUR/USD now.",
        "Your balance is $250.",
        "We refunded 100 USD to you.",
    ],
)
def test_unsafe_phrases_in_non_advice_draft(text):
    out = run(classification=clf(Intent.deposit_issue), draft=draft(text=text))
    assert "ungrounded_claim" in out.safety_flags
    assert out.grounded is False
    assert out.needs_human_escalation is True
    assert out.reply_draft == text  # non-advice drafts are flagged, not rewritten


def test_ungrounded_claim_does_not_fire_on_safe_draft():
    assert "ungrounded_claim" not in run().safety_flags


def test_account_amount_present_in_evidence_is_allowed():
    out = run(
        evidence=[ev(snippet="Minimum deposit is 10 USD.")],
        draft=draft(text="The minimum deposit is 10 USD."),
    )
    assert "ungrounded_claim" not in out.safety_flags


# --- escalation reasons, ordering, determinism ----------------------------


def test_flags_unique_and_in_fixed_order_and_reasons_match_triggers():
    out = run(
        ticket=ticket("Can you check my account 12345 and tell me my exact balance?"),
        classification=clf(Intent.withdrawal_issue, certainty=0.2, conflicting=True),
        evidence=[ev(relevance=0.1)],
        draft=draft(text="Your withdrawal will be approved.", cited=["A9"]),
    )
    expected = [
        "insufficient_grounding",
        "policy_sensitive",
        "low_confidence",
        "account_specific_request",
        "conflicting_signals",
        "ungrounded_claim",
    ]
    assert out.safety_flags == expected
    assert len(set(out.safety_flags)) == len(out.safety_flags)
    assert [f for f in SAFETY_FLAGS if f in out.safety_flags] == out.safety_flags
    assert len(out.escalation_reasons) == 6
    assert len(set(out.escalation_reasons)) == 6
    assert all(isinstance(r, str) and r.endswith(".") for r in out.escalation_reasons)


def test_single_trigger_gives_single_reason():
    out = run(classification=clf(Intent.account_lock))
    assert out.escalation_reasons == [policy.REASONS["policy_sensitive"]]


def test_determinism():
    kwargs = dict(
        ticket=advice_ticket(),
        classification=clf(Intent.trading_advice_request, certainty=0.7),
        evidence=[ev("A5", 0.33)],
        draft=draft(text="Prices will rise.", cited=["A5"]),
    )
    assert run(**kwargs) == run(**kwargs)
    assert run(**kwargs).model_dump() == run(**kwargs).model_dump()


def test_llm_refusal_with_refusal_verbs_is_kept():
    # Real Kimi output on the sample trading-advice ticket: a safe refusal that
    # names what it won't do ("predict which asset will go up").
    refusal = ("I’m sorry, but I can’t predict which asset will go up today, recommend an asset to "
               "trade, or guarantee profits. For guidance tailored to your financial circumstances, "
               "please consider consulting a licensed financial advisor.")
    assert has_refusal(refusal)
    assert not has_unsafe_content(refusal, [])


def test_contrast_after_refusal_verb_is_still_flagged():
    sneaky = "I can't predict prices, but gold will go up today."
    assert has_unsafe_content(sneaky, [])
