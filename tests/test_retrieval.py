import subprocess
import sys

import pytest

from support_pipeline.data import load_articles, load_examples, load_tickets
from support_pipeline.models import Article, Evidence, Intent, LabeledExample, Ticket
from support_pipeline.retrieval import (
    CROSS_ENCODER_MODEL,
    ENCODER_MODEL,
    EmbeddingRetriever,
    TfidfRetriever,
    make_retriever,
)
from support_pipeline.runtime import LocalModelPool
from support_pipeline.text import split_sentences, tokenize

from .conftest import REPO_ROOT

# Test-only expectations keyed by sample ticket ID (pipeline logic never uses IDs).
EXPECTED_TOP_ARTICLE = {"T1": "A1", "T2": "A2", "T3": "A3", "T4": "A5", "T5": "A4"}
EXPECTED_INTENT = {
    "T1": Intent.deposit_issue,
    "T2": Intent.password_reset,
    "T3": Intent.withdrawal_issue,
    "T4": Intent.trading_advice_request,
    "T5": Intent.account_lock,
}


@pytest.fixture(scope="module")
def sample():
    return load_tickets(), load_articles(), load_examples()


@pytest.fixture(scope="module")
def tfidf(sample):
    _, articles, examples = sample
    retriever = TfidfRetriever()
    retriever.index(articles, examples)
    return retriever


def _ticket(message: str) -> Ticket:
    return Ticket(id="Q", message=message)


# ---------------------------------------------------------------- text.py


def test_tokenize_lowercases_drops_stopwords_and_punctuation():
    assert tokenize("The Card, my BALANCE!") == ["card", "balance"]


def test_tokenize_light_suffix_stripping():
    assert tokenize("deposits pending declined processes") == [
        "deposit",
        "pend",
        "declin",
        "process",
    ]


def test_tokenize_keeps_short_stems():
    # "used" -> "us" would be too short, so only rules leaving >= 3 chars apply.
    assert tokenize("used bed sing") == ["used", "bed", "sing"]


def test_tokenize_empty_and_non_ascii():
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("dépôt") == ["d", "p", "t"]  # documented limit: accents split words


def test_split_sentences():
    assert split_sentences("One. Two? Three! Four") == ["One.", "Two?", "Three!", "Four"]
    assert split_sentences("") == []
    assert split_sentences("  ") == []


# ---------------------------------------------------------------- TF-IDF retrieval


def test_tfidf_name():
    assert TfidfRetriever().name == "tfidf"


@pytest.mark.parametrize("ticket_id", sorted(EXPECTED_TOP_ARTICLE))
def test_sample_ticket_top_article_is_topical(sample, tfidf, ticket_id):
    tickets, _, _ = sample
    ticket = next(t for t in tickets if t.id == ticket_id)
    evidence = tfidf.retrieve(ticket)
    assert evidence[0].article_id == EXPECTED_TOP_ARTICLE[ticket_id]


def test_retrieve_output_invariants(sample, tfidf):
    tickets, articles, _ = sample
    known = {a.article_id for a in articles}
    for ticket in tickets:
        evidence = tfidf.retrieve(ticket)
        assert 1 <= len(evidence) <= 3
        ids = [e.article_id for e in evidence]
        assert len(set(ids)) == len(ids)
        for position, item in enumerate(evidence):
            assert isinstance(item, Evidence)
            assert item.article_id in known
            assert 0.0 <= item.relevance <= 1.0
            assert round(item.relevance, 3) == item.relevance
            assert item.lexical_score == item.relevance
            assert item.rerank_score is None
            assert 0 < len(item.snippet) <= 240
            if position > 0:
                assert item.relevance >= 0.35
        keys = [(-e.relevance, e.article_id) for e in evidence]
        assert keys == sorted(keys)


def test_title_and_snippet_come_from_article(sample, tfidf):
    tickets, articles, _ = sample
    by_id = {a.article_id: a for a in articles}
    for ticket in tickets:
        for item in tfidf.retrieve(ticket):
            article = by_id[item.article_id]
            assert item.title == article.title
            assert item.snippet in article.body


def test_snippet_is_best_matching_sentence():
    article = Article(
        article_id="X",
        title="Help",
        body="Unrelated filler words here. Reset the password from the login page.",
    )
    retriever = TfidfRetriever()
    retriever.index([article], [])
    [evidence] = retriever.retrieve(_ticket("how do I reset my password"))
    assert evidence.snippet == "Reset the password from the login page."


def test_snippet_truncated_on_word_boundary():
    long_sentence = " ".join(["deposit"] * 60) + "."
    retriever = TfidfRetriever()
    retriever.index([Article(article_id="L", title="Deposits", body=long_sentence)], [])
    [evidence] = retriever.retrieve(_ticket("deposit"))
    assert len(evidence.snippet) <= 240
    assert evidence.snippet.endswith("…")
    assert evidence.snippet[:-1].split(" ") == ["deposit"] * len(evidence.snippet[:-1].split(" "))


def test_short_snippet_not_marked_truncated(tfidf, sample):
    tickets, _, _ = sample
    for ticket in tickets:
        for item in tfidf.retrieve(ticket):
            assert not item.snippet.endswith("…")


def test_ties_broken_by_article_id():
    twin = dict(title="Card deposit times", body="Card deposits can be delayed.")
    retriever = TfidfRetriever()
    retriever.index([Article(article_id="B", **twin), Article(article_id="A", **twin)], [])
    evidence = retriever.retrieve(_ticket("card deposit delayed"))
    assert [e.article_id for e in evidence] == ["A", "B"]
    assert evidence[0].relevance == evidence[1].relevance


def test_empty_kb_returns_nothing():
    retriever = TfidfRetriever()
    retriever.index([], [])
    assert retriever.retrieve(_ticket("card deposit")) == []
    assert retriever.similar_examples(_ticket("card deposit")) == []


def test_zero_overlap_query_returns_lowest_id_with_zero_relevance(sample):
    _, articles, examples = sample
    retriever = TfidfRetriever()
    retriever.index(list(reversed(articles)), examples)
    for message in ("zzzz qqqq xylophone", "", "   "):
        evidence = retriever.retrieve(_ticket(message))
        assert len(evidence) == 1
        assert evidence[0].article_id == "A1"
        assert evidence[0].relevance == 0.0
        assert retriever.similar_examples(_ticket(message)) == []


def test_use_before_index_raises():
    retriever = TfidfRetriever()
    with pytest.raises(RuntimeError):
        retriever.retrieve(_ticket("x"))
    with pytest.raises(RuntimeError):
        retriever.similar_examples(_ticket("x"))


def test_no_state_leaks_between_queries(sample, tfidf):
    tickets, _, _ = sample
    first = [tfidf.retrieve(t) for t in tickets]
    tfidf.retrieve(_ticket("withdrawal password deposit lock profit"))
    tfidf.similar_examples(_ticket("withdrawal password deposit lock profit"))
    second = [tfidf.retrieve(t) for t in tickets]
    assert first == second


def test_fresh_instances_identical(sample):
    tickets, articles, examples = sample

    def run():
        retriever = TfidfRetriever()
        retriever.index(articles, examples)
        return [
            (
                [e.model_dump() for e in retriever.retrieve(t)],
                [s.model_dump() for s in retriever.similar_examples(t)],
            )
            for t in tickets
        ]

    assert run() == run()


def test_reindex_replaces_corpus(sample):
    _, articles, examples = sample
    retriever = TfidfRetriever()
    retriever.index(articles, examples)
    retriever.index([Article(article_id="Z", title="Only", body="Only article.")], [])
    assert [e.article_id for e in retriever.retrieve(_ticket("card deposit"))] == ["Z"]
    assert retriever.similar_examples(_ticket("card deposit")) == []


# ---------------------------------------------------------------- similar examples


def test_similar_examples_invariants(sample, tfidf):
    tickets, _, examples = sample
    messages = {e.message for e in examples}
    for ticket in tickets:
        similar = tfidf.similar_examples(ticket)
        assert len(similar) <= 3
        for item in similar:
            assert item.message in messages
            assert 0 < item.score <= 1
            assert round(item.score, 3) == item.score
        scores = [s.score for s in similar]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("ticket_id", sorted(EXPECTED_INTENT))
def test_similar_examples_top1_intent(sample, tfidf, ticket_id):
    tickets, _, _ = sample
    ticket = next(t for t in tickets if t.id == ticket_id)
    similar = tfidf.similar_examples(ticket)
    assert similar, "expected at least one similar example"
    assert similar[0].intent == EXPECTED_INTENT[ticket_id]


def test_similar_examples_ties_keep_original_order():
    examples = [
        LabeledExample(message="card deposit", intent=Intent.deposit_issue),
        LabeledExample(message="card deposit", intent=Intent.other),
    ]
    retriever = TfidfRetriever()
    retriever.index([], examples)
    similar = retriever.similar_examples(_ticket("card deposit"))
    assert [s.intent for s in similar] == [Intent.deposit_issue, Intent.other]


# ---------------------------------------------------------------- embedding backend (fakes)

VOCAB = ("deposit", "password", "withdrawal", "lock", "profit")


class FakeEncoder:
    """Bag-of-keywords encoder over a fixed vocabulary (not normalized on purpose)."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = []
        for text in texts:
            lowered = text.lower()
            out.append([2.0 * lowered.count(word) for word in VOCAB])
        return out


class FakeReranker:
    """Logit = keyword overlap score minus a constant; overridable per article title."""

    def __init__(self, logits_by_title=None):
        self.logits_by_title = logits_by_title or {}
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        scores = []
        for query, doc in pairs:
            title = doc.split("\n", 1)[0]
            if title in self.logits_by_title:
                scores.append(self.logits_by_title[title])
            else:
                overlap = sum(1 for w in VOCAB if w in query.lower() and w in doc.lower())
                scores.append(4.0 * overlap - 2.0)
        return scores


def _fake_embedding(reranker=None):
    encoder = FakeEncoder()
    reranker = reranker or FakeReranker()
    loads = []

    def encoder_factory(device):
        loads.append(("encoder", device))
        return encoder

    def reranker_factory(device):
        loads.append(("reranker", device))
        return reranker

    retriever = EmbeddingRetriever(encoder_factory=encoder_factory, reranker_factory=reranker_factory)
    return retriever, encoder, reranker, loads


KB = [
    Article(article_id="K3", title="Lock help", body="Account lock happens. Wait for cooldown."),
    Article(article_id="K1", title="Deposit help", body="Deposit delays happen. Check history."),
    Article(article_id="K2", title="Password help", body="Password reset steps. Check spam."),
]
EXAMPLES = [
    LabeledExample(message="my deposit is missing", intent=Intent.deposit_issue),
    LabeledExample(message="forgot password", intent=Intent.password_reset),
    LabeledExample(message="account lock", intent=Intent.account_lock),
]


def test_embedding_name_and_lazy_import():
    retriever, _, _, loads = _fake_embedding()
    assert retriever.name == "embedding"
    assert loads == []  # nothing loaded before index


def test_embedding_retrieve_relevance_is_sigmoid_of_logit():
    retriever, _, _, _ = _fake_embedding()
    retriever.index(KB, EXAMPLES)
    evidence = retriever.retrieve(_ticket("deposit missing"))
    top = evidence[0]
    assert top.article_id == "K1"
    assert top.rerank_score == 2.0
    assert top.relevance == round(1 / (1 + 2.718281828459045 ** -2.0), 3) == 0.881
    assert top.lexical_score == 1.0
    assert top.snippet in KB[1].body
    # Other articles: logit -2 -> sigmoid 0.119 < 0.35, so only one kept.
    assert len(evidence) == 1


def test_embedding_selection_keeps_up_to_three_above_threshold():
    reranker = FakeReranker({"Lock help": 0.0, "Deposit help": 0.0, "Password help": 0.0})
    retriever, _, _, _ = _fake_embedding(reranker)
    retriever.index(KB, EXAMPLES)
    evidence = retriever.retrieve(_ticket("deposit"))
    # all logits 0 -> relevance 0.5 for all; tie broken by article ID
    assert [e.article_id for e in evidence] == ["K1", "K2", "K3"]
    assert all(e.relevance == 0.5 for e in evidence)


def test_embedding_ordering_by_rerank_not_dense():
    reranker = FakeReranker({"Lock help": 3.0, "Deposit help": 1.0, "Password help": -5.0})
    retriever, _, _, _ = _fake_embedding(reranker)
    retriever.index(KB, EXAMPLES)
    evidence = retriever.retrieve(_ticket("deposit"))
    assert [e.article_id for e in evidence] == ["K3", "K1"]
    assert evidence[0].lexical_score == 0.0  # dense cosine kept separately


def test_embedding_reranks_at_most_five():
    articles = [Article(article_id=f"M{i}", title=f"T{i}", body="deposit info.") for i in range(8)]
    retriever, _, reranker, _ = _fake_embedding()
    retriever.index(articles, [])
    retriever.retrieve(_ticket("deposit"))
    assert len(reranker.pairs) == 5


def test_embedding_empty_kb_and_before_index():
    retriever, _, _, _ = _fake_embedding()
    with pytest.raises(RuntimeError):
        retriever.retrieve(_ticket("deposit"))
    with pytest.raises(RuntimeError):
        retriever.similar_examples(_ticket("deposit"))
    retriever.index([], [])
    assert retriever.retrieve(_ticket("deposit")) == []
    assert retriever.similar_examples(_ticket("deposit")) == []


def test_embedding_similar_examples_clipped_and_positive():
    retriever, _, _, _ = _fake_embedding()
    retriever.index(KB, EXAMPLES)
    similar = retriever.similar_examples(_ticket("password"))
    assert [s.intent for s in similar] == [Intent.password_reset]
    assert similar[0].score == 1.0
    assert retriever.similar_examples(_ticket("nothing relevant")) == []


def test_embedding_pool_keeps_one_model_resident():
    retriever, _, _, loads = _fake_embedding()
    retriever.index(KB, EXAMPLES)
    retriever.retrieve(_ticket("deposit"))
    pool = retriever.pool
    assert pool.active == CROSS_ENCODER_MODEL
    retriever.similar_examples(_ticket("deposit"))
    assert pool.active == ENCODER_MODEL
    assert [kind for kind, _ in loads] == ["encoder", "reranker", "encoder"]
    assert all(device == pool.device for _, device in loads)


def test_embedding_accepts_explicit_pool():
    encoder, reranker = FakeEncoder(), FakeReranker()
    pool = LocalModelPool(
        {ENCODER_MODEL: lambda device: encoder, CROSS_ENCODER_MODEL: lambda device: reranker}
    )
    retriever = EmbeddingRetriever(pool=pool)
    retriever.index(KB, EXAMPLES)
    assert retriever.retrieve(_ticket("lock"))[0].article_id == "K3"


# ---------------------------------------------------------------- factory and import hygiene


def test_make_retriever_default_is_tfidf():
    retriever, reason = make_retriever()
    assert isinstance(retriever, TfidfRetriever)
    assert reason == ""
    retriever, reason = make_retriever("tfidf")
    assert isinstance(retriever, TfidfRetriever) and reason == ""


def test_make_retriever_embedding_falls_back_here():
    # torch / sentence-transformers are not installed in the test environment.
    retriever, reason = make_retriever("embedding")
    assert isinstance(retriever, TfidfRetriever)
    assert reason.strip()


def test_make_retriever_unknown_preference():
    with pytest.raises(ValueError):
        make_retriever("bm25")


def test_import_does_not_load_heavy_libraries():
    code = (
        "import support_pipeline.retrieval, sys; "
        "assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
