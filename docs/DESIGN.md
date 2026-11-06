# Support Ticket Pipeline: Design Notes

## Decided architecture

The pipeline runs seven explicit stages: `LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS`. Each stage sits behind a small interface (`Retriever`, `Classifier`, `Generator`), so retrieval, classification, and generation can be swapped independently.

**Retrieval is always local:** pure-Python, passage-level TF-IDF (smoothed IDF, sublinear TF, cosine). Each KB article is split into title and title+sentence passages, and the article score is its best passage. The evidence passed to the drafter is the whole article when it fits in 480 characters, otherwise the best passage.

**Default mode `auto`: the LLMs do the language work** whenever `TOGETHER_API_KEY` is available, from the shell or a git-ignored `.env`.
- `deepseek-ai/DeepSeek-V4.1-Flash-1017` classifies.
- `moonshotai/Kimi-K3.1-Pro` writes every reply from the selected articles.
- Calls go to Together's REST API over the standard library, so no SDK is needed.
- Every response is validated with Pydantic, and each call gets one retry.
- Both are reasoning models, so they get token budgets for reasoning before the JSON answer: 1000 for DeepSeek, 2000 for Kimi.

**Offline fallback**, used without a key, on a failed call, or with `--mode offline`:
- Classification is a similarity-weighted vote over the 3 nearest labeled examples in `examples.json`. No keyword rules.
- Drafting is a per-intent safe framing composed with the retrieved article text.
- It needs no API key, model download, or torch.

**Optional retrieval backend** (`--retrieval embedding`): `sentence-transformers/all-MiniLM-L6-v2` dense search, then `cross-encoder/ms-marco-MiniLM-L6-v2` reranks the top `min(5, corpus size)` candidates. It's used only when installed (`requirements-local-models.txt`) and already cached; otherwise the pipeline falls back to TF-IDF and records why.

## Responsibility split

1. Load and validate tickets, KB, and labeled examples.
2. Index KB passages and examples (TF-IDF by default).
3. Classify: find ≤3 similar examples, then the similarity classifier (or DeepSeek, which receives only the ticket and those examples) returns a validated `Classification`.
4. Retrieve: the ticket message is the query. Keep the top article, plus up to two more with relevance ≥ 0.35. Order is deterministic, with article ID as the tie-breaker.
5. Generate: the generator (template or Kimi) receives only the ticket, the compact classification, and the selected evidence (ID, title, snippet, relevance). It never receives candidate lists, example-search results, prompts, tool history, credentials, or other tickets. Kimi's prompt includes the snippets and forbids answering beyond them.
6. Validate: deterministic policy computes `grounded`, confidence, flags, and escalation. Model text never sets its own safety status.
7. Write `results.json` and `debug_report.json`.

## Confidence, escalation, and safety

`confidence = clamp(0.5 × classifier certainty + 0.5 × best relevance − penalties, 0, 1)`, with penalties of 0.35 for insufficient evidence, 0.20 for conflicting signals, and 0.40 for an ungrounded claim. Confidence below **0.45** sets `low_confidence`.

Certainty is the winning label's share of the summed example similarity. Signals conflict when the runner-up has ≥ 60% of the winner's similarity mass.

Escalation is a separate rule decision. It triggers on:
- low confidence
- best relevance < 0.35, or no evidence
- a policy-sensitive intent (`withdrawal_issue`, `account_lock`)
- conflicting signals
- an account-specific request
- an ungrounded claim

Flags, in fixed order: `advice_request`, `insufficient_grounding`, `policy_sensitive`, `low_confidence`, `account_specific_request`, `conflicting_signals`, `ungrounded_claim`.

Output safety checks scan the draft for market predictions, profit guarantees, asset recommendations, payment/withdrawal outcome promises, and account specifics absent from evidence. Any hit sets `ungrounded_claim`, marks the result not grounded, and escalates. Trading-advice requests always get a polite refusal. If the refusal is missing, the safe template refusal replaces the draft.

`debug_report.json` gives each ticket a flat entry: `ticket_id`, `predicted_intent`, `retrieved_article_titles`, `escalated`, plain-English `escalation_reasons`, `confidence_factors`, `backends`. Full stage traces sit under a separate `traces` key.

## Reproducibility

The committed `results.json` and `debug_report.json` are generated with DeepSeek and Kimi (`"mode": "together"`). LLM wording isn't byte-reproducible, even at temperature 0, so reproducibility is checked on the deterministic parts. `validate.py` never calls an API and needs no key:
- Retrieval is local and identical in every mode, so each ticket's `retrieved_articles` must equal a fresh offline rerun.
- The offline path is run twice and must be byte-identical.
- LLM text is checked by schema and the deterministic safety policy.

Artifacts generated offline are byte-compared against a rerun in full. All floats are rounded to 3 decimals before serialization, JSON uses LF line endings, and traces exclude timings. The optional embedding backend is reproducible only on the same machine and library versions.

## Delivery and testing

Every behavior follows RED → GREEN → REFACTOR. Acceptance tests run the real offline pipeline on the supplied files:
- T4 is classified `trading_advice_request`, refused politely, and flagged `advice_request`, with no prediction text.
- T1 retrieves `A1`.
- Every result has exactly the 8 required keys.
- Every ticket gets one result.

Tests may name sample IDs; pipeline logic never does. A renamed-ID test enforces this. Provider and local-model calls are faked in unit tests. Real calls are opt-in smoke scripts.

Run: `python main.py`. Validate: `python validate.py`. Test: `python -m pytest`.

## Requirement coverage

| Brief item | Design element | Proof |
|---|---|---|
| MUST 1 pipeline, clean checkout | 7 named stages; `auto` default runs with or without a key | `test_pipeline` stage order; clean-env run |
| MUST 2 classification | 6-label enum; example-vote classifier / schema-bound DeepSeek | `test_fallback` (5/5 samples), `test_agents` |
| MUST 3 retrieval | Passage TF-IDF, top 1–3, ID tie-break, separate stage | `test_retrieval`; `validate.py` ID check |
| MUST 4 grounded reply | Evidence-only generator input, cited IDs, refusal framing | `test_fallback`, `test_agents`, `test_policy` |
| MUST 5 validation | `TicketResult` (extra keys forbidden) + policy checks | `test_policy`; `validate.py` schema checks |
| MUST 6 basic tests | Real-pipeline acceptance tests | `test_acceptance.py`; `validate.py` T1/T4 |
| SHOULD 7 confidence/escalation | Formula, penalties, 0.45 cutoff, trigger rules | `test_policy` |
| SHOULD 8 safety flags | Seven fixed-order flags | `test_policy` (one test per flag) |
| SHOULD 9 explainability | Flat `debug_report.json` entries | `test_pipeline` debug-shape test |
| STRETCH 10 swappable | `Retriever` / `Classifier` / `Generator` protocols | Fake-injection tests |
| STRETCH 11 offline fallback | No key or failed call → similarity classifier + template drafts | Acceptance tests run without a key |
| Required artifacts | Committed inputs, outputs, README, `validate.py`, tests | `validate.py` file-existence check |
| Validation checks | Files, JSON, one result each, schema, T4, IDs, reproducibility | `validate.py`, `test_validate.py` |
| Technical constraints | Local files only; no ID logic; no hardcoded answers; no trading recs | Renamed-ID, lazy-import, and policy tests |

## Known limits

- Retrieval and examples are English-oriented. Non-English tickets usually score near zero and escalate rather than get a confident answer.
- Scores are not calibrated probabilities. The 0.35 and 0.45 thresholds are conservative and hand-set. T4's best lexical score (~0.33 in a prototype) falls just under 0.35, so it is refused and also escalated for insufficient grounding.
- The labeled example set is small (22), so classification of paraphrases far from the examples is weak. Edge tests show wrong labels on out-of-domain, non-English, and injected text (for example "What are your office hours?" → `deposit_issue`). Every such case gets low confidence or a policy flag and escalates, so the safety outcome holds even when the label doesn't.
- Output safety checks are phrase-based and conservative, so they may over-escalate. For example, any run of 5+ digits in a ticket (a phone number, say) counts as an account-specific request.
- Offline fallback drafts are composed from per-intent framings plus retrieved article text, not hardcoded per ticket. They read as templated; with a key, Kimi writes the reply instead.
- LLM replies vary between runs, so only retrieval and the offline path are byte-reproducible.

## Sources

Model IDs and structured-output support come from Together's documentation. The opt-in smoke test (`scripts/smoke_together.py`) confirmed both IDs live, with validated JSON from each, on 2026-11-06.

- https://docs.together.ai/docs/serverless/models
- https://docs.together.ai/docs/changelog
- https://docs.together.ai/docs/json-mode
- https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
- https://www.sbert.net/docs/pretrained-models/ce-msmarco.html

## Optional: execution inspector (not built)

Not required by the challenge, and not part of this submission. If it's added later, a small page would show a ticket selector, the ordered stage list, and a stage inspector with each stage's compact input/output. It would label the boundary where generation receives only the selected evidence, never show credentials or provider logs, and use native controls and text for every state. For now, `scripts/try_ticket.py` prints the same stage trace in the terminal.
