"""Offline classifier and template generator tests (no network, no keys)."""

from __future__ import annotations

import re

import pytest

from support_pipeline.data import load_articles, load_examples, load_tickets
from support_pipeline.fallback import REFUSAL_SENTENCE, SimilarityClassifier, TemplateGenerator
from support_pipeline.models import Classification, Evidence, Intent, SimilarExample, Ticket


def _ticket(message: str = "help me please") -> Ticket:
    return Ticket(id="X1", message=message)


def _ex(intent: Intent, score: float, message: str | None = None) -> SimilarExample:
    return SimilarExample(message=message or f"example for {intent.value}", intent=intent, score=score)


def _ev(article_id: str, relevance: float, title: str | None = None, snippet: str | None = None) -> Evidence:
    return Evidence(
        article_id=article_id,
        title=title or f"Title {article_id}",
        snippet=snippet or f"Snippet text for {article_id}.",
        lexical_score=relevance,
        relevance=relevance,
    )


def _cls(intent: Intent) -> Classification:
    return Classification(
        intent=intent, certainty=0.9, goal="g", constraints=[], evidence=[], backend="similarity_fallback"
    )


# ---------------------------------------------------------------- SimilarityClassifier


def test_classifier_name_and_backend():
    clf = SimilarityClassifier()
    assert clf.name == "similarity_fallback"
    result = clf.classify(_ticket(), [_ex(Intent.deposit_issue, 0.5)])
    assert result.backend == "similarity_fallback"


def test_classifier_sums_scores_per_intent():
    similar = [
        _ex(Intent.withdrawal_issue, 0.5),
        _ex(Intent.deposit_issue, 0.3),
        _ex(Intent.deposit_issue, 0.3),
    ]
    result = SimilarityClassifier().classify(_ticket(), similar)
    assert result.intent is Intent.deposit_issue
    assert result.certainty == round(0.6 / 1.1, 3)
    # runner-up 0.5 >= 0.6 * 0.6
    assert result.conflicting_signals is True


def test_classifier_tie_broken_by_intent_declaration_order():
    similar = [_ex(Intent.account_lock, 0.4), _ex(Intent.password_reset, 0.4)]
    result = SimilarityClassifier().classify(_ticket(), similar)
    assert result.intent is Intent.password_reset
    assert result.certainty == 0.5
    assert result.conflicting_signals is True


def test_classifier_tie_with_float_noise_still_uses_declaration_order():
    similar = [_ex(Intent.account_lock, 0.3), _ex(Intent.deposit_issue, 0.1), _ex(Intent.deposit_issue, 0.2)]
    result = SimilarityClassifier().classify(_ticket(), similar)
    assert result.intent is Intent.deposit_issue


def test_classifier_conflict_threshold_boundary():
    below = SimilarityClassifier().classify(_ticket(), [_ex(Intent.deposit_issue, 1.0), _ex(Intent.other, 0.599)])
    at = SimilarityClassifier().classify(_ticket(), [_ex(Intent.deposit_issue, 1.0), _ex(Intent.other, 0.6)])
    assert below.conflicting_signals is False
    assert at.conflicting_signals is True


def test_classifier_single_intent_is_certain_and_not_conflicting():
    similar = [_ex(Intent.account_lock, 0.7), _ex(Intent.account_lock, 0.2)]
    result = SimilarityClassifier().classify(_ticket(), similar)
    assert result.intent is Intent.account_lock
    assert result.certainty == 1.0
    assert result.conflicting_signals is False


def test_classifier_zero_score_runner_up_is_not_conflict():
    similar = [_ex(Intent.account_lock, 0.7), _ex(Intent.deposit_issue, 0.0)]
    result = SimilarityClassifier().classify(_ticket(), similar)
    assert result.conflicting_signals is False


def test_classifier_empty_similar_returns_other():
    result = SimilarityClassifier().classify(_ticket(), [])
    assert result.intent is Intent.other
    assert result.certainty == 0.0
    assert result.evidence == []


def test_classifier_all_zero_scores_returns_other():
    result = SimilarityClassifier().classify(_ticket(), [_ex(Intent.deposit_issue, 0.0)])
    assert result.intent is Intent.other
    assert result.certainty == 0.0


@pytest.mark.parametrize("message", ["", "   \n\t"])
def test_classifier_blank_message_returns_other(message):
    result = SimilarityClassifier().classify(_ticket(message), [_ex(Intent.deposit_issue, 0.9)])
    assert result.intent is Intent.other
    assert result.certainty == 0.0


def test_classifier_evidence_is_matched_example_messages_and_goal_is_generic():
    similar = [
        _ex(Intent.deposit_issue, 0.5, "money not arrived"),
        _ex(Intent.deposit_issue, 0.4, "card top-up pending"),
        _ex(Intent.other, 0.1, "hello"),
    ]
    result = SimilarityClassifier().classify(_ticket("anything"), similar)
    assert result.evidence == ["money not arrived", "card top-up pending", "hello"]
    assert result.goal and "anything" not in result.goal
    assert isinstance(result.constraints, list)


def test_classifier_ignores_ticket_text_keywords():
    """No keyword rules: the label comes only from the similar examples."""
    result = SimilarityClassifier().classify(
        _ticket("withdrawal withdrawal password locked asset"), [_ex(Intent.deposit_issue, 0.2)]
    )
    assert result.intent is Intent.deposit_issue


def test_classifier_every_intent_has_goal():
    for intent in Intent:
        result = SimilarityClassifier().classify(_ticket(), [_ex(intent, 0.5)])
        assert result.intent is intent
        assert result.goal.strip()


EXPECTED_SAMPLE_INTENTS = {
    "T1": Intent.deposit_issue,
    "T2": Intent.password_reset,
    "T3": Intent.withdrawal_issue,
    "T4": Intent.trading_advice_request,
    "T5": Intent.account_lock,
}


def test_classifier_real_tfidf_similarities_on_sample_tickets(repo_root):
    try:
        from support_pipeline.retrieval import TfidfRetriever
    except ImportError:
        pytest.skip("support_pipeline.retrieval.TfidfRetriever not available yet")
    retriever = TfidfRetriever()
    retriever.index(load_articles(repo_root / "kb_articles.json"), load_examples(repo_root / "examples.json"))
    clf = SimilarityClassifier()
    tickets = load_tickets(repo_root / "tickets.json")
    got = {t.id: clf.classify(t, retriever.similar_examples(t)).intent for t in tickets}
    assert got == EXPECTED_SAMPLE_INTENTS


# ---------------------------------------------------------------- TemplateGenerator


def test_generator_name_and_backend():
    gen = TemplateGenerator()
    assert gen.name == "template_fallback"
    draft = gen.draft(_ticket(), _cls(Intent.deposit_issue), [_ev("A1", 0.6)])
    assert draft.backend == "template_fallback"


def test_generator_cites_evidence_with_title_and_snippet():
    ev = _ev("A1", 0.6, title="Card deposit processing times", snippet="Card deposits are usually instant.")
    draft = TemplateGenerator().draft(_ticket(), _cls(Intent.deposit_issue), [ev])
    assert 'Based on our help article "Card deposit processing times": Card deposits are usually instant.' in draft.text
    assert draft.cited_article_ids == ["A1"]
    assert draft.sufficient_evidence is True


def test_generator_cites_at_most_two_positive_relevance_items_in_order():
    evidence = [_ev("A2", 0.7), _ev("A9", 0.0), _ev("A1", 0.4), _ev("A3", 0.36)]
    draft = TemplateGenerator().draft(_ticket(), _cls(Intent.password_reset), evidence)
    assert draft.cited_article_ids == ["A2", "A1"]
    assert "Title A9" not in draft.text
    assert "Title A3" not in draft.text


def test_generator_cites_only_provided_ids():
    evidence = [_ev("K7", 0.5), _ev("K8", 0.4)]
    for intent in Intent:
        draft = TemplateGenerator().draft(_ticket(), _cls(intent), evidence)
        assert set(draft.cited_article_ids) <= {"K7", "K8"}


def test_generator_insufficient_when_best_cited_below_threshold():
    draft = TemplateGenerator().draft(_ticket(), _cls(Intent.deposit_issue), [_ev("A1", 0.34)])
    assert draft.cited_article_ids == ["A1"]
    assert draft.sufficient_evidence is False


def test_generator_threshold_inclusive():
    draft = TemplateGenerator().draft(_ticket(), _cls(Intent.deposit_issue), [_ev("A1", 0.35)])
    assert draft.sufficient_evidence is True


@pytest.mark.parametrize("evidence", [[], [_ev("A1", 0.0), _ev("A2", 0.0)]])
def test_generator_no_usable_evidence_defers_to_agent(evidence):
    draft = TemplateGenerator().draft(_ticket(), _cls(Intent.deposit_issue), evidence)
    assert draft.cited_article_ids == []
    assert draft.sufficient_evidence is False
    assert "support agent will follow up" in draft.text
    assert "Based on our help article" not in draft.text


def test_generator_other_intent_defers_to_agent_even_with_evidence():
    draft = TemplateGenerator().draft(_ticket(), _cls(Intent.other), [_ev("A1", 0.8)])
    assert draft.cited_article_ids == []
    assert draft.sufficient_evidence is False
    assert "support agent will follow up" in draft.text


def test_generator_trading_advice_refuses_and_includes_no_advice_snippet():
    no_advice = _ev(
        "A5",
        0.33,
        title="No investment or trading advice",
        snippet="Support agents must not provide market predictions, profit guarantees, or asset recommendations.",
    )
    ticket = _ticket("Can you tell me which asset will go up today so I can make profit?")
    draft = TemplateGenerator().draft(ticket, _cls(Intent.trading_advice_request), [no_advice])
    assert draft.text.startswith(REFUSAL_SENTENCE)
    assert no_advice.snippet in draft.text
    assert draft.cited_article_ids == ["A5"]
    assert draft.sufficient_evidence is False  # 0.33 < 0.35: single rule, no special case
    assert "will go up" not in draft.text.replace(no_advice.snippet, "")
    assert "asset" not in draft.text.replace(no_advice.snippet, "").lower()


def test_generator_trading_advice_refuses_without_evidence():
    draft = TemplateGenerator().draft(_ticket(), _cls(Intent.trading_advice_request), [])
    assert draft.text.startswith(REFUSAL_SENTENCE)
    assert draft.cited_article_ids == []
    assert draft.sufficient_evidence is False


def test_refusal_sentence_wording():
    assert REFUSAL_SENTENCE == (
        "I'm sorry, but we can't provide market predictions, trading recommendations, or profit guarantees."
    )


FORBIDDEN = [r"guarantee", r"will be approved", r"\bwithin\b", r"\bbuy\b", r"will go up"]


@pytest.mark.parametrize("intent", list(Intent))
@pytest.mark.parametrize("with_evidence", [True, False])
def test_generator_framings_contain_no_forbidden_phrases(intent, with_evidence):
    evidence = [_ev("A1", 0.6, title="Neutral title", snippet="Neutral snippet.")] if with_evidence else []
    ticket = _ticket("Will my withdrawal be approved within an hour? Which stock should I buy, will it go up?")
    text = TemplateGenerator().draft(ticket, _cls(intent), evidence).text
    framing = text.replace(REFUSAL_SENTENCE, "").lower()
    for pattern in FORBIDDEN:
        assert not re.search(pattern, framing), (intent, pattern, text)


def test_generator_does_not_echo_ticket_text():
    ticket = _ticket("ZETA-UNIQUE-TOKEN my account number is 123456")
    for intent in Intent:
        text = TemplateGenerator().draft(ticket, _cls(intent), [_ev("A1", 0.6)]).text
        assert "ZETA-UNIQUE-TOKEN" not in text
        assert "123456" not in text


def test_generator_withdrawal_makes_no_promises():
    text = TemplateGenerator().draft(_ticket(), _cls(Intent.withdrawal_issue), [_ev("A3", 0.6)]).text.lower()
    for phrase in ["approved", "will be processed", "hours", "days", "guarantee"]:
        assert phrase not in text.replace("snippet text for a3.", "")


def test_generator_ends_with_safe_next_step():
    text = TemplateGenerator().draft(_ticket(), _cls(Intent.deposit_issue), [_ev("A1", 0.6)]).text
    assert "support agent" in text.split("Based on our help article")[-1]
