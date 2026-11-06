"""LLM-by-default behaviour: .env loading, auto mode, retries, full-article context, validation."""

import json
import os
import shutil

import pytest

import main as cli
import validate
from support_pipeline.agents import DeepSeekClassifier, KimiGenerator
from support_pipeline.data import load_articles, load_examples, load_tickets
from support_pipeline.env import load_dotenv
from support_pipeline.models import Article, Ticket
from support_pipeline.pipeline import MODES, build_pipeline
from support_pipeline.retrieval import SNIPPET_LIMIT, TfidfRetriever


# ------------------------------------------------------------------ .env loading

def test_load_dotenv_sets_missing_values_without_overriding(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPPORT_PIPELINE_NO_DOTENV")
    monkeypatch.setenv("ALREADY_SET", "keep")
    env = tmp_path / ".env"
    env.write_text('# comment\nexport NEW_ONE="quoted value"\nALREADY_SET=override\nnot a pair\n', encoding="utf-8")
    loaded = load_dotenv(env)
    assert loaded == ["NEW_ONE"]
    assert os.environ["NEW_ONE"] == "quoted value"
    assert os.environ["ALREADY_SET"] == "keep"
    monkeypatch.delenv("NEW_ONE")


def test_load_dotenv_can_be_disabled(tmp_path):
    env = tmp_path / ".env"
    env.write_text("SHOULD_NOT_LOAD=1\n", encoding="utf-8")
    assert load_dotenv(env) == []
    assert "SHOULD_NOT_LOAD" not in os.environ


def test_load_dotenv_missing_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPPORT_PIPELINE_NO_DOTENV")
    assert load_dotenv(tmp_path / "absent.env") == []


# ------------------------------------------------------------------ auto mode

def test_auto_is_a_mode_and_the_cli_default():
    assert "auto" in MODES
    assert cli._parser().parse_args([]).mode == "auto"


def test_auto_without_key_runs_offline_with_a_note():
    pipeline, backends, notes = build_pipeline("auto")
    assert pipeline.mode == "offline"
    assert (backends.classifier, backends.generator) == ("similarity_fallback", "template_fallback")
    assert any("TOGETHER_API_KEY" in note for note in notes)


def test_auto_with_key_uses_deepseek_and_kimi(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-real")
    pipeline, backends, _ = build_pipeline("auto", transport=object())
    assert pipeline.mode == "together"
    assert (backends.classifier, backends.generator) == ("deepseek", "kimi")


def test_main_auto_without_key_records_offline_mode(tmp_path):
    assert cli.main(["--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "debug_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "offline"


# ------------------------------------------------------------------ retries and budgets

class FlakyTransport:
    def __init__(self, reply: str, failures: int = 1):
        self.reply, self.failures, self.calls = reply, failures, []

    def chat_json(self, model, messages, schema, max_tokens, temperature):
        self.calls.append(max_tokens)
        if len(self.calls) <= self.failures:
            raise RuntimeError("transient")
        return self.reply


def _sample():
    tickets = load_tickets()
    retriever = TfidfRetriever()
    retriever.index(load_articles(), load_examples())
    return tickets[0], retriever


def test_kimi_retries_once_then_succeeds():
    ticket, retriever = _sample()
    evidence = retriever.retrieve(ticket)
    transport = FlakyTransport(json.dumps({"text": "Please check your transaction history.",
                                           "cited_article_ids": ["A1"], "sufficient_evidence": True}))
    classification = DeepSeekClassifier(FlakyTransport("{}", failures=99)).classify(ticket, [])
    draft = KimiGenerator(transport).draft(ticket, classification, evidence)
    assert draft.backend == "kimi"
    assert len(transport.calls) == 2


def test_deepseek_retries_once_then_falls_back():
    ticket, retriever = _sample()
    transport = FlakyTransport("{}", failures=99)
    classifier = DeepSeekClassifier(transport)
    result = classifier.classify(ticket, retriever.similar_examples(ticket))
    assert result.backend == "similarity_fallback"
    assert len(transport.calls) == 2


def test_reasoning_models_get_room_to_answer():
    assert KimiGenerator.MAX_TOKENS >= 2000
    assert DeepSeekClassifier.MAX_TOKENS >= 1000


# ------------------------------------------------------------------ full-article context

def test_short_article_body_is_passed_whole():
    tickets, retriever = load_tickets(), TfidfRetriever()
    articles = load_articles()
    retriever.index(articles, load_examples())
    [evidence] = retriever.retrieve(tickets[0])
    a1 = next(a for a in articles if a.article_id == "A1")
    assert evidence.snippet == a1.body
    assert "check transaction history" in evidence.snippet


def test_long_article_falls_back_to_best_sentence():
    body = ("Filler words " * 50) + ". Reset the password from the login page."
    retriever = TfidfRetriever()
    retriever.index([Article(article_id="X", title="Help", body=body)], [])
    [evidence] = retriever.retrieve(Ticket(id="Q", message="how do I reset my password"))
    assert evidence.snippet == "Reset the password from the login page."
    assert len(evidence.snippet) <= SNIPPET_LIMIT


# ------------------------------------------------------------------ validation of LLM artifacts

def _llm_like_root(tmp_path, repo_root):
    for name in ("tickets.json", "kb_articles.json", "examples.json", "README.md"):
        shutil.copy(repo_root / name, tmp_path / name)
    assert cli.main(["--mode", "offline", "--data-dir", str(tmp_path), "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "debug_report.json").read_text(encoding="utf-8"))
    report["mode"] = "together"
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    for item in results:  # LLM wording differs from the template; that must be allowed
        if item["intent"] != "trading_advice_request":
            text = "A differently worded, LLM-written reply."
            item["reply_draft"] = text
            for trace in report["traces"][item["ticket_id"]]:  # a real run saves the draft in its trace
                if trace["stage"] == "GENERATE_REPLY":
                    trace["output"]["reply_draft"] = text
    (tmp_path / "debug_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (tmp_path / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return tmp_path


def test_validate_accepts_llm_artifacts_without_a_key(tmp_path, repo_root):
    root = _llm_like_root(tmp_path, repo_root)
    assert validate.validate(root) == []


def test_validate_rejects_llm_artifacts_with_changed_retrieval(tmp_path, repo_root):
    root = _llm_like_root(tmp_path, repo_root)
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    results[1]["retrieved_articles"] = ["A5"]
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    errors = validate.validate(root)
    assert any("retriev" in e for e in errors), errors
