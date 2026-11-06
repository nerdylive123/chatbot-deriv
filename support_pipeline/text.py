"""Deterministic text helpers. Standard library only.

Known limit: the token pattern is ASCII-only, so accented characters split words
("dépôt" -> "d", "p", "t"). Non-English tickets therefore score low and escalate.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

STOPWORDS: frozenset[str] = frozenset(
    (
        "a an the i my me you your is are was were be to of and or but in on at for so "
        "can do what why how which will it this that after by not still am have has"
    ).split()
)

_SUFFIXES: tuple[str, ...] = ("ing", "ed", "es", "s")
_MIN_STEM = 3


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, split on [a-z0-9]+, drop stopwords, strip one light suffix."""
    return [_stem(tok) for tok in _TOKEN.findall(text.lower()) if tok not in STOPWORDS]


def split_sentences(body: str) -> list[str]:
    """Split on whitespace that follows ., ! or ?; blank pieces are dropped."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(body.strip()) if part.strip()]


__all__ = ["STOPWORDS", "tokenize", "split_sentences"]
