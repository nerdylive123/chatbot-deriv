"""Retrieval backends: pure-Python TF-IDF (default) and optional local embeddings.

The query is always the ticket message. Both backends share one selection rule:
sort by (-relevance, article_id), keep the top article, then up to two more with
relevance >= RELEVANCE_THRESHOLD. This module never imports torch or
sentence_transformers at import time; the embedding backend loads them lazily.
"""

from __future__ import annotations

import importlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .models import Article, Evidence, LabeledExample, SimilarExample, Ticket
from .runtime import LocalModelPool, ModelFactory
from .text import split_sentences, tokenize

ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

MAX_RESULTS = 3
MAX_EXAMPLES = 3
RELEVANCE_THRESHOLD = 0.35
SNIPPET_LIMIT = 240
RERANK_DEPTH = 5


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self, articles: Sequence[Article], examples: Sequence[LabeledExample]) -> None: ...

    def similar_examples(self, ticket: Ticket) -> list[SimilarExample]: ...

    def retrieve(self, ticket: Ticket) -> list[Evidence]: ...


# ---------------------------------------------------------------- shared helpers


def truncate_snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    """Trim to <= limit chars on a word boundary, appending "…" only if trimmed."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "…"


def select_evidence(candidates: list[Evidence]) -> list[Evidence]:
    """Deterministic order; top article always kept, others need the threshold."""
    ranked = sorted(candidates, key=lambda e: (-e.relevance, e.article_id))
    if not ranked:
        return []
    kept = [ranked[0]]
    for item in ranked[1:]:
        if len(kept) >= MAX_RESULTS:
            break
        if item.relevance >= RELEVANCE_THRESHOLD:
            kept.append(item)
    return kept


def _passages(article: Article) -> tuple[list[str], list[str]]:
    """Return (passage texts, body sentences); passage 0 is the title alone."""
    sentences = split_sentences(article.body)
    return [article.title] + [f"{article.title} {s}" for s in sentences], sentences


def _best_sentence(sentences: list[str], sentence_scores: list[float], title: str) -> str:
    if not sentences:
        return title
    best = max(range(len(sentences)), key=lambda i: (sentence_scores[i], -i))
    return sentences[best]


def _rank_examples(
    examples: list[LabeledExample], scores: list[float]
) -> list[SimilarExample]:
    scored = []
    for position, (example, score) in enumerate(zip(examples, scores)):
        value = round(min(max(score, 0.0), 1.0), 3)
        if value > 0:
            scored.append((-value, position, example))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [
        SimilarExample(message=ex.message, intent=ex.intent, score=-neg)
        for neg, _, ex in scored[:MAX_EXAMPLES]
    ]


# ---------------------------------------------------------------- TF-IDF


class _TfidfIndex:
    """Smoothed IDF, sublinear TF, L2-normalized sparse vectors, cosine scoring."""

    def __init__(self, documents: Sequence[str]) -> None:
        tokenized = [tokenize(doc) for doc in documents]
        n_docs = len(tokenized)
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        self.idf: dict[str, float] = {
            term: math.log((1 + n_docs) / (1 + df[term])) + 1.0 for term in sorted(df)
        }
        self.vectors: list[dict[str, float]] = [self._vectorize(tokens) for tokens in tokenized]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(t for t in tokens if t in self.idf)
        weights = {
            term: (1.0 + math.log(counts[term])) * self.idf[term] for term in sorted(counts)
        }
        norm = math.sqrt(math.fsum(w * w for w in weights.values()))
        if norm == 0:
            return {}
        return {term: w / norm for term, w in weights.items()}

    def scores(self, query: str) -> list[float]:
        q = self._vectorize(tokenize(query))
        if not q:
            return [0.0] * len(self.vectors)
        return [
            math.fsum(weight * vec[term] for term, weight in q.items() if term in vec)
            for vec in self.vectors
        ]


class TfidfRetriever:
    """Default backend: passage-level TF-IDF over articles, plus an example index."""

    name = "tfidf"

    def __init__(self) -> None:
        self._articles: list[Article] | None = None
        self._examples: list[LabeledExample] = []
        self._passage_index: _TfidfIndex | None = None
        self._passage_owner: list[int] = []  # passage -> article position
        self._passage_sentence: list[int] = []  # passage -> sentence position (-1 = title)
        self._sentences: list[list[str]] = []
        self._example_index: _TfidfIndex | None = None

    def index(self, articles: Sequence[Article], examples: Sequence[LabeledExample]) -> None:
        texts: list[str] = []
        owners: list[int] = []
        sentence_ids: list[int] = []
        sentences_per_article: list[list[str]] = []
        for position, article in enumerate(articles):
            passages, sentences = _passages(article)
            sentences_per_article.append(sentences)
            texts.extend(passages)
            owners.extend([position] * len(passages))
            sentence_ids.extend(range(-1, len(sentences)))
        self._articles = list(articles)
        self._examples = list(examples)
        self._passage_index = _TfidfIndex(texts)
        self._passage_owner = owners
        self._passage_sentence = sentence_ids
        self._sentences = sentences_per_article
        self._example_index = _TfidfIndex([e.message for e in self._examples])

    def _require_index(self) -> None:
        if self._articles is None:
            raise RuntimeError("TfidfRetriever.index() must be called before querying")

    def retrieve(self, ticket: Ticket) -> list[Evidence]:
        self._require_index()
        assert self._passage_index is not None and self._articles is not None
        passage_scores = self._passage_index.scores(ticket.message)
        best = [0.0] * len(self._articles)
        sentence_scores = [[0.0] * len(s) for s in self._sentences]
        for score, owner, sentence in zip(
            passage_scores, self._passage_owner, self._passage_sentence
        ):
            best[owner] = max(best[owner], score)
            if sentence >= 0:
                sentence_scores[owner][sentence] = score
        candidates = []
        for position, article in enumerate(self._articles):
            relevance = round(min(max(best[position], 0.0), 1.0), 3)
            snippet = _best_sentence(
                self._sentences[position], sentence_scores[position], article.title
            )
            candidates.append(
                Evidence(
                    article_id=article.article_id,
                    title=article.title,
                    snippet=truncate_snippet(snippet),
                    lexical_score=relevance,
                    rerank_score=None,
                    relevance=relevance,
                )
            )
        return select_evidence(candidates)

    def similar_examples(self, ticket: Ticket) -> list[SimilarExample]:
        self._require_index()
        assert self._example_index is not None
        return _rank_examples(self._examples, self._example_index.scores(ticket.message))


# ---------------------------------------------------------------- optional embeddings


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(v) for v in vector]
    norm = math.sqrt(math.fsum(v * v for v in values))
    return [v / norm for v in values] if norm else [0.0] * len(values)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def _default_encoder_factory(device: str) -> Any:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return SentenceTransformer(ENCODER_MODEL, device=device, local_files_only=True)


def _default_reranker_factory(device: str) -> Any:
    import torch  # noqa: PLC0415
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    # Ask for raw logits; relevance = sigmoid(logit) is applied here, not by the library.
    try:
        return CrossEncoder(
            CROSS_ENCODER_MODEL,
            device=device,
            local_files_only=True,
            activation_fn=torch.nn.Identity(),
        )
    except TypeError:  # older sentence-transformers naming
        return CrossEncoder(
            CROSS_ENCODER_MODEL,
            device=device,
            local_files_only=True,
            default_activation_function=torch.nn.Identity(),
        )


class EmbeddingRetriever:
    """Optional backend: dense MiniLM search, cross-encoder rerank of the top 5.

    Encoders expose ``encode(list[str]) -> list[vector]``; rerankers expose
    ``predict(list[(query, doc)]) -> list[logit]``. Tests inject fakes.
    """

    name = "embedding"

    def __init__(
        self,
        encoder_factory: ModelFactory | None = None,
        reranker_factory: ModelFactory | None = None,
        pool: LocalModelPool | None = None,
    ) -> None:
        if pool is None:
            pool = LocalModelPool(
                {
                    ENCODER_MODEL: encoder_factory or _default_encoder_factory,
                    CROSS_ENCODER_MODEL: reranker_factory or _default_reranker_factory,
                }
            )
        self.pool = pool
        self._articles: list[Article] | None = None
        self._examples: list[LabeledExample] = []
        self._passage_vectors: list[list[float]] = []
        self._passage_owner: list[int] = []
        self._passage_sentence: list[int] = []
        self._sentences: list[list[str]] = []
        self._example_vectors: list[list[float]] = []

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoder = self.pool.acquire(ENCODER_MODEL)
        return [_normalize(v) for v in encoder.encode(texts)]

    def index(self, articles: Sequence[Article], examples: Sequence[LabeledExample]) -> None:
        texts: list[str] = []
        owners: list[int] = []
        sentence_ids: list[int] = []
        sentences_per_article: list[list[str]] = []
        for position, article in enumerate(articles):
            passages, sentences = _passages(article)
            sentences_per_article.append(sentences)
            texts.extend(passages)
            owners.extend([position] * len(passages))
            sentence_ids.extend(range(-1, len(sentences)))
        self._examples = list(examples)
        self._passage_vectors = self._encode(texts)
        self._example_vectors = self._encode([e.message for e in self._examples])
        self._passage_owner = owners
        self._passage_sentence = sentence_ids
        self._sentences = sentences_per_article
        self._articles = list(articles)

    def _require_index(self) -> None:
        if self._articles is None:
            raise RuntimeError("EmbeddingRetriever.index() must be called before querying")

    def retrieve(self, ticket: Ticket) -> list[Evidence]:
        self._require_index()
        assert self._articles is not None
        if not self._articles:
            return []
        [query] = self._encode([ticket.message])
        dense = [-1.0] * len(self._articles)
        sentence_scores = [[-1.0] * len(s) for s in self._sentences]
        for vector, owner, sentence in zip(
            self._passage_vectors, self._passage_owner, self._passage_sentence
        ):
            score = _dot(query, vector)
            dense[owner] = max(dense[owner], score)
            if sentence >= 0:
                sentence_scores[owner][sentence] = score
        order = sorted(
            range(len(self._articles)),
            key=lambda i: (-dense[i], self._articles[i].article_id),
        )[: min(RERANK_DEPTH, len(self._articles))]
        reranker = self.pool.acquire(CROSS_ENCODER_MODEL)
        pairs = [
            (ticket.message, f"{self._articles[i].title}\n{self._articles[i].body}") for i in order
        ]
        logits = [float(x) for x in reranker.predict(pairs)]
        candidates = []
        for i, logit in zip(order, logits):
            article = self._articles[i]
            snippet = _best_sentence(self._sentences[i], sentence_scores[i], article.title)
            candidates.append(
                Evidence(
                    article_id=article.article_id,
                    title=article.title,
                    snippet=truncate_snippet(snippet),
                    lexical_score=round(dense[i], 3),
                    rerank_score=round(logit, 3),
                    relevance=round(_sigmoid(logit), 3),
                )
            )
        return select_evidence(candidates)

    def similar_examples(self, ticket: Ticket) -> list[SimilarExample]:
        self._require_index()
        if not self._examples:
            return []
        [query] = self._encode([ticket.message])
        scores = [_dot(query, vector) for vector in self._example_vectors]
        return _rank_examples(self._examples, scores)


# ---------------------------------------------------------------- factory


def _embedding_unavailable_reason() -> str:
    """Return "" if the embedding backend can run offline, else a plain reason."""
    try:
        importlib.import_module("sentence_transformers")
    except Exception as exc:  # ImportError, or torch failing to load
        return f"sentence-transformers is not usable ({type(exc).__name__}); using TF-IDF retrieval"
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except Exception:
        return "huggingface_hub is unavailable to check the model cache; using TF-IDF retrieval"
    for model in (ENCODER_MODEL, CROSS_ENCODER_MODEL):
        try:
            cached = try_to_load_from_cache(model, "config.json")
        except Exception:
            cached = None
        if not isinstance(cached, str):
            return f"model {model} is not in the local cache; using TF-IDF retrieval"
    return ""


def make_retriever(preference: str = "tfidf") -> tuple[Retriever, str]:
    """Build the preferred retriever, falling back to TF-IDF with a reason."""
    if preference == "tfidf":
        return TfidfRetriever(), ""
    if preference == "embedding":
        reason = _embedding_unavailable_reason()
        if reason:
            return TfidfRetriever(), reason
        return EmbeddingRetriever(), ""
    raise ValueError(f"unknown retrieval backend {preference!r}; expected 'tfidf' or 'embedding'")


__all__ = [
    "CROSS_ENCODER_MODEL",
    "ENCODER_MODEL",
    "RELEVANCE_THRESHOLD",
    "EmbeddingRetriever",
    "Retriever",
    "TfidfRetriever",
    "make_retriever",
    "select_evidence",
    "truncate_snippet",
]
