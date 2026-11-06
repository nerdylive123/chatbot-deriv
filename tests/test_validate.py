"""validate.py: every check fails loudly on a broken artifact and passes on a good one.

Tests may reference sample IDs (T1, T4); pipeline logic must not.
"""

import json
import shutil
from pathlib import Path

import pytest

import main as cli
import validate
from support_pipeline.together_client import ENV_VAR

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILES = ("tickets.json", "kb_articles.json", "examples.json", "README.md")


@pytest.fixture(scope="module")
def pristine(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pristine")
    for name in INPUT_FILES:
        shutil.copyfile(REPO_ROOT / name, root / name)
    assert cli.main(["--data-dir", str(root), "--out", str(root)]) == 0
    return root


@pytest.fixture
def root(pristine, tmp_path) -> Path:
    target = tmp_path / "root"
    shutil.copytree(pristine, target)
    return target


def read_results(root: Path) -> list[dict]:
    return json.loads((root / "results.json").read_text(encoding="utf-8"))


def write_results(root: Path, results: list[dict]) -> None:
    (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def joined(errors: list[str]) -> str:
    return "\n".join(errors)


def test_real_repo_validates():
    assert validate.validate(REPO_ROOT) == []


def test_fresh_tmp_root_validates(root):
    assert validate.validate(root) == []


def test_main_returns_zero_and_prints_ok(root, capsys):
    assert validate.main(["--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "OK:" in out


def test_main_returns_one_and_prints_errors(root, capsys):
    (root / "README.md").unlink()
    assert validate.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "README.md" in out


@pytest.mark.parametrize("name", ["tickets.json", "kb_articles.json", "results.json", "debug_report.json", "README.md"])
def test_missing_file(root, name):
    (root / name).unlink()
    errors = validate.validate(root)
    assert errors
    assert any(name in e and "missing" in e for e in errors)


@pytest.mark.parametrize("name", ["tickets.json", "kb_articles.json", "results.json", "debug_report.json"])
def test_invalid_json(root, name):
    (root / name).write_text("{not json", encoding="utf-8")
    errors = validate.validate(root)
    assert any(name in e and "invalid JSON" in e for e in errors)


def test_results_not_a_list(root):
    (root / "results.json").write_text('{"ticket_id": "T1"}', encoding="utf-8")
    assert any("results.json" in e and "array" in e for e in validate.validate(root))


def test_extra_key(root):
    results = read_results(root)
    results[0]["internal_notes"] = "x"
    write_results(root, results)
    errors = validate.validate(root)
    assert any("extra" in e and "internal_notes" in e for e in errors)


def test_missing_key(root):
    results = read_results(root)
    del results[1]["grounded"]
    write_results(root, results)
    errors = validate.validate(root)
    assert any("missing" in e and "grounded" in e for e in errors)


def test_missing_ticket_result(root):
    results = [r for r in read_results(root) if r["ticket_id"] != "T3"]
    write_results(root, results)
    errors = validate.validate(root)
    assert any("T3" in e and "no result" in e for e in errors)


def test_duplicate_result(root):
    results = read_results(root)
    results.append(dict(results[1]))
    write_results(root, results)
    errors = validate.validate(root)
    assert any("duplicate" in e and "T2" in e for e in errors)


def test_unknown_ticket_id(root):
    results = read_results(root)
    extra = dict(results[1])
    extra["ticket_id"] = "Z9"
    results.append(extra)
    write_results(root, results)
    assert any("unknown ticket" in e and "Z9" in e for e in validate.validate(root))


def test_unknown_article_id(root):
    results = read_results(root)
    results[1]["retrieved_articles"] = ["A99"]
    write_results(root, results)
    errors = validate.validate(root)
    assert any("A99" in e and "unknown article" in e for e in errors)


@pytest.mark.parametrize("articles", [[], ["A1", "A2", "A3", "A4"]])
def test_article_count_out_of_range(root, articles):
    results = read_results(root)
    results[1]["retrieved_articles"] = articles
    write_results(root, results)
    assert any("1-3" in e for e in validate.validate(root))


def test_confidence_out_of_range(root):
    results = read_results(root)
    results[0]["confidence"] = 1.5
    write_results(root, results)
    errors = validate.validate(root)
    assert any("confidence" in e and "1.5" in e for e in errors)


def test_invalid_intent(root):
    results = read_results(root)
    results[0]["intent"] = "refund_request"
    write_results(root, results)
    assert any("intent" in e and "refund_request" in e for e in validate.validate(root))


def test_non_bool_flag(root):
    results = read_results(root)
    results[0]["grounded"] = "yes"
    write_results(root, results)
    assert any("grounded" in e and "bool" in e for e in validate.validate(root))


def test_unknown_safety_flag(root):
    results = read_results(root)
    results[0]["safety_flags"] = ["made_up_flag"]
    write_results(root, results)
    assert any("made_up_flag" in e for e in validate.validate(root))


def test_t4_not_refused(root):
    results = read_results(root)
    t4 = next(r for r in results if r["ticket_id"] == "T4")
    t4["reply_draft"] = "Gold will go up today, buy now."
    write_results(root, results)
    errors = validate.validate(root)
    assert any("T4" in e and "refusal" in e for e in errors)
    assert any("T4" in e and "prediction" in e for e in errors)


def test_t4_wrong_intent_and_flag(root):
    results = read_results(root)
    t4 = next(r for r in results if r["ticket_id"] == "T4")
    t4["intent"] = "other"
    t4["safety_flags"] = []
    write_results(root, results)
    errors = validate.validate(root)
    assert any("T4" in e and "trading_advice_request" in e for e in errors)
    assert any("T4" in e and "advice_request" in e for e in errors)


def test_t1_missing_a1(root):
    results = read_results(root)
    t1 = next(r for r in results if r["ticket_id"] == "T1")
    t1["retrieved_articles"] = ["A2"]
    write_results(root, results)
    assert any("T1" in e and "A1" in e for e in validate.validate(root))


def test_changed_bytes_fail_reproducibility(root):
    path = root / "results.json"
    text = path.read_text(encoding="utf-8")
    marker = '"reply_draft": "Thanks'
    assert marker in text
    path.write_bytes(text.replace(marker, '"reply_draft": "Thanks ', 1).encode("utf-8"))
    errors = validate.validate(root)
    assert len(errors) == 1, errors
    assert "results.json" in errors[0]
    assert "reproduc" in errors[0]
    assert "line" in errors[0]


def test_changed_debug_report_fails_reproducibility(root):
    path = root / "debug_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["tickets"][0]["escalation_reasons"] = ["edited by hand"]
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    errors = validate.validate(root)
    assert any("debug_report.json" in e and "reproduc" in e for e in errors)


def test_together_mode_validates_without_key(root, monkeypatch):
    # LLM-generated artifacts must stay checkable by an evaluator who has no key.
    monkeypatch.delenv(ENV_VAR, raising=False)
    path = root / "debug_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["mode"] = "together"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert validate.validate(root) == []


def test_embedding_backend_unavailable(root, monkeypatch):
    monkeypatch.setattr(validate, "embedding_unavailable_reason", lambda: "sentence-transformers is not usable")
    path = root / "debug_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["backends"]["retrieval"] = "embedding"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    errors = validate.validate(root)
    assert any("embedding" in e and "python main.py" in e for e in errors)


def test_validate_does_not_modify_root(root):
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    validate.validate(root)
    after = {p.name: p.read_bytes() for p in root.iterdir()}
    assert before == after
