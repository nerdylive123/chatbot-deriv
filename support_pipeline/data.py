"""Disk loaders for tickets, KB articles, and labeled examples.

Each loader preserves input order and raises DataError with a clear message on
any problem. Ticket messages may be blank; IDs may not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import Article, LabeledExample, Ticket

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PATHS: dict[str, Path] = {
    "tickets": REPO_ROOT / "tickets.json",
    "articles": REPO_ROOT / "kb_articles.json",
    "examples": REPO_ROOT / "examples.json",
}

M = TypeVar("M", bound=BaseModel)


class DataError(ValueError):
    """Raised when an input data file is missing or invalid."""


def _read_array(path: Path) -> list[Any]:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{path}: file not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})") from None
    except UnicodeDecodeError:
        raise DataError(f"{path}: invalid JSON (file is not UTF-8)") from None
    if not isinstance(data, list):
        raise DataError(f"{path}: expected a JSON array at top level, got {type(data).__name__}")
    return data


def _parse(path: Path, items: list[Any], model: type[M]) -> list[M]:
    parsed: list[M] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise DataError(f"{path}: entry {index} must be a JSON object")
        try:
            parsed.append(model.model_validate(item))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<entry>'}: {err['msg']}"
                for err in exc.errors()
            )
            raise DataError(f"{path}: entry {index} is invalid: {problems}") from None
    return parsed


def _check_ids(path: Path, ids: list[str], label: str) -> None:
    seen: set[str] = set()
    for index, value in enumerate(ids):
        if not value.strip():
            raise DataError(f"{path}: entry {index} has a blank {label}")
        if value in seen:
            raise DataError(f"{path}: duplicate {label} {value!r}")
        seen.add(value)


def load_tickets(path: Path = DEFAULT_PATHS["tickets"]) -> list[Ticket]:
    path = Path(path)
    tickets = _parse(path, _read_array(path), Ticket)
    _check_ids(path, [t.id for t in tickets], "ticket id")
    return tickets


def load_articles(path: Path = DEFAULT_PATHS["articles"]) -> list[Article]:
    path = Path(path)
    articles = _parse(path, _read_array(path), Article)
    _check_ids(path, [a.article_id for a in articles], "article_id")
    return articles


def load_examples(path: Path = DEFAULT_PATHS["examples"]) -> list[LabeledExample]:
    """Load labeled examples. An optional "id" field is documentation only and is dropped."""
    path = Path(path)
    return _parse(path, _read_array(path), LabeledExample)


__all__ = [
    "DEFAULT_PATHS",
    "REPO_ROOT",
    "DataError",
    "load_tickets",
    "load_articles",
    "load_examples",
]
