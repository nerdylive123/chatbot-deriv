# Support Ticket Pipeline

Classifies customer support tickets, retrieves the matching help articles, drafts a grounded reply, and decides with deterministic rules whether a human must review it. Output: `results.json` (one result per ticket) and `debug_report.json` (why each decision was made).

The default path is fully offline. It needs no API key, no model download and no torch: just Python and `pip install -r requirements.txt` (Pydantic and pytest).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py                  # writes results.json and debug_report.json
python validate.py              # checks the artifacts and byte-compares a fresh rerun
python -m pytest -q             # full test suite (no network, key, model or GPU)
```

`main.py` options: `--mode offline|together`, `--retrieval tfidf|embedding`, `--data-dir DIR` (inputs), `--out DIR` (artifacts). `validate.py` takes `--root DIR`.

## Pipeline

```
LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS
  tickets,     TF-IDF over   <=3 similar       ticket message      ticket + compact   deterministic    results.json
  KB,          KB passages   labeled examples  is the query;       classification +   policy: grounded, debug_report.json
  examples     and examples  -> intent         1-3 articles        selected evidence  confidence, flags,
                                                                   only               escalation
```

| Module | Role |
|---|---|
| `main.py` | CLI: build the pipeline, load inputs, run, write artifacts |
| `validate.py` | Artifact checks and reproducibility rerun |
| `support_pipeline/models.py` | Shared Pydantic contract (`Ticket`, `Evidence`, `TicketResult`, ...), intents, stages, flags |
| `support_pipeline/data.py` | Strict loaders for `tickets.json`, `kb_articles.json`, `examples.json` |
| `support_pipeline/text.py` | Deterministic tokenizer and sentence splitter |
| `support_pipeline/retrieval.py` | `TfidfRetriever` (default) and optional `EmbeddingRetriever` |
| `support_pipeline/runtime.py` | Device detection and one-model-at-a-time pool for local models |
| `support_pipeline/fallback.py` | Offline `SimilarityClassifier` and `TemplateGenerator` |
| `support_pipeline/agents.py` | Together-backed `DeepSeekClassifier` and `KimiGenerator`, each wrapping the offline fallback |
| `support_pipeline/together_client.py` | Lazy Together SDK transport; key never printed or logged |
| `support_pipeline/policy.py` | Grounding, confidence, safety flags, escalation, output safety checks |
| `support_pipeline/pipeline.py` | Stage orchestration and traces |
| `support_pipeline/serialize.py` | Deterministic JSON writing (3-decimal floats, LF, fixed key order) |

## Backends and modes

| Stage | Default | Optional |
|---|---|---|
| Retrieval | `tfidf`: pure-Python passage TF-IDF, standard library only | `embedding` (`--retrieval embedding`): `sentence-transformers/all-MiniLM-L6-v2` + `cross-encoder/ms-marco-MiniLM-L6-v2` rerank. Needs `pip install -r requirements-local-models.txt` and the models already cached (`local_files_only=True`); otherwise falls back to TF-IDF and prints why |
| Classification | `similarity_fallback`: similarity-weighted vote over the top-3 nearest labeled examples. No keyword rules | `deepseek` (`--mode together`): `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Drafting | `template_fallback`: one generic framing per intent plus the retrieved snippets | `kimi` (`--mode together`): `moonshotai/Kimi-K3.1-Pro` |

Together mode needs `pip install -r requirements-together.txt` and `TOGETHER_API_KEY` in the environment. Without the key, `python main.py --mode together` exits with code 2 and a clear message (no silent downgrade). Any provider error or malformed/invalid JSON falls back to the offline component for that ticket, and the fallback is recorded in the per-ticket `backends` and traces. The two model IDs and their structured-output support come from Together's documentation and are **unverified** until `scripts/smoke_together.py` is run with a key. The local-model backend is likewise unverified on Python 3.14; check it with `scripts/smoke_models.py`. Both smoke scripts are opt-in, print `SKIPPED: ...` and exit 0 when their prerequisites are missing, and never run under pytest.

## Context boundary

- **Classifier (DeepSeek)** receives only the ticket (id, message, language) and at most 3 similar labeled examples (message, intent, score). Its system prompt treats ticket and example text as untrusted data.
- **Generator (Kimi or template)** receives only the ticket, the compact classification (intent, goal, constraints) and the selected evidence (article ID, title, snippet, relevance). It never sees candidate rankings, similar examples, other tickets, prompts from earlier stages, or credentials. Citations to articles outside the selected set are dropped and counted as an ungrounded claim.
- **Policy** decides `grounded`, confidence, flags and escalation. Model text never sets its own safety status.

## Confidence, escalation and safety flags

```
confidence = clamp(0.5 * classifier_certainty + 0.5 * best_relevance - penalties, 0, 1)   # rounded to 3 decimals
penalties:  0.35 insufficient evidence (no evidence, best relevance < 0.35, or the draft says evidence is insufficient)
            0.20 conflicting signals (runner-up intent has >= 60% of the winner's similarity mass)
            0.40 ungrounded claim
```

Retrieval keeps the top article plus up to two more with relevance >= 0.35. Confidence below 0.45 is `low_confidence`.

Flags, in fixed order: `advice_request`, `insufficient_grounding`, `policy_sensitive` (withdrawal or account-lock intent), `low_confidence`, `account_specific_request`, `conflicting_signals`, `ungrounded_claim`. Every flag except `advice_request` triggers escalation and adds one plain-English reason to `debug_report.json`.

The draft is scanned for market predictions, profit guarantees, asset recommendations, payment/withdrawal outcome promises, and account figures not present in the evidence; any hit sets `ungrounded_claim`. Trading-advice requests always get a polite refusal; if a draft lacks one, the safe template refusal replaces it.

## Artifacts

- `results.json`: array of results in input order, each with exactly `ticket_id`, `intent`, `retrieved_articles`, `reply_draft`, `grounded`, `confidence`, `needs_human_escalation`, `safety_flags`.
- `debug_report.json`: `{"mode", "backends", "tickets": [...], "traces": {...}}`. Each `tickets` entry has `ticket_id`, `predicted_intent`, `retrieved_article_titles`, `escalated`, `escalation_reasons`, `confidence_factors` and the `backends` that actually ran. `traces` maps each ticket ID to its seven stage traces (compact input/output, backend, device, status; no timings).

The committed artifacts were generated by `python main.py` in offline TF-IDF mode. Output is deterministic (sorted ties, 3-decimal floats, fixed key order, LF line endings), so `validate.py` reruns the pipeline in the recorded mode into a temp directory and byte-compares both files. If the artifacts were produced in together mode and no key is present, or with an embedding backend that isn't available, it fails with a message telling you to rerun `python main.py`.

`validate.py` also checks: required files exist and parse; tickets and KB load; exactly one result per ticket; exact keys and types; allowed intents; 1-3 known article IDs; confidence in [0, 1]; known flags; T4 is `trading_advice_request`, flagged `advice_request` and politely refused with no prediction text; T1 retrieves `A1`.

## Current sample results (offline)

| Ticket | Intent | Articles | Grounded | Confidence | Escalate | Flags |
|---|---|---|---|---|---|---|
| T1 | deposit_issue | A1 | true | 0.544 | false | |
| T2 | password_reset | A2 | true | 0.802 | false | |
| T3 | withdrawal_issue | A3 | true | 0.801 | true | policy_sensitive |
| T4 | trading_advice_request | A5 | false | 0.316 | true | advice_request, insufficient_grounding, low_confidence |
| T5 | account_lock | A4 | true | 0.881 | true | policy_sensitive |

T4 is refused, and also escalated because its best lexical relevance (0.33) sits just under the 0.35 grounding threshold. The thresholds were not tuned to the sample tickets.

## Tests

`python -m pytest -q` runs unit tests per module plus:

- `tests/test_acceptance.py`: the real offline pipeline on the supplied files (T4 refusal, T1 retrieves A1, exact schema, one result per ticket, committed artifacts equal a fresh run byte-for-byte).
- `tests/test_edge_cases.py` with `tests/fixtures/edge_tickets.json`: empty, whitespace, non-English, unknown-intent, ambiguous, trading manipulation, prompt injection, and renamed copies of the sample tickets (identical results except `ticket_id`).
- `tests/test_validate.py`: each `validate.py` check fails on a broken copy of the artifacts.

Tests may reference sample IDs; pipeline logic never does.

## Known limits

- Retrieval and examples are English-oriented. Non-English tickets usually score near zero and escalate rather than get a confident answer, and their predicted intent can be wrong (in the edge fixtures, a Spanish withdrawal question is labeled `trading_advice_request` and an office-hours question `deposit_issue`; both are escalated with low confidence).
- Scores are not calibrated probabilities. The 0.35 and 0.45 thresholds are conservative and hand-set.
- The labeled example set is small (18+ examples), so paraphrases far from the examples classify poorly.
- Output safety checks are phrase-based and conservative, so they may over-escalate. For example, any run of 5+ digits in a ticket counts as an account-specific request.
- Offline drafts are a per-intent framing plus retrieved snippets, so they read as templated. The KB text was written for support agents, so quoted snippets can sound agent-facing ("Ask the client to confirm..."; T4 quotes "Support agents must not provide...").
- The snippet is the best-matching sentence, not necessarily the most actionable one. T1's reply quotes why card deposits are delayed, not the "confirm payment, check transaction history" steps.
- Together mode is covered only by fake-transport tests until the smoke script runs with a key, and it is not guaranteed reproducible. The embedding backend is reproducible only on the same machine and library versions.

Design notes and rationale: [`docs/DESIGN.md`](docs/DESIGN.md).
