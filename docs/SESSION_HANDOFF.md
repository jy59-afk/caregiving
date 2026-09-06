# Session handoff — paste this into a new session

> Purpose: a compact state snapshot so a new session doesn't re-read prior
> chat history. Read this + [CLAUDE.md](../CLAUDE.md) and you have full context.
> Update this file at the end of each working session.

**Last updated:** 2026-09-06 · after build slice 8 (evaluation harness)

---

## What this project is

RespiteSG / Respite Navigator — an agentic chatbot matching Singapore family
caregivers to respite-care services. Matching + activation layer, not a
directory. Full brief: [CLAUDE.md](../CLAUDE.md),
[docs/PROJECT_BRIEF.md](PROJECT_BRIEF.md). Build order + architecture table
are in CLAUDE.md.

## Environment (already set up — do not redo)

- Repo: `C:\Users\tanyi\Desktop\hackathon`, git branch `main`. This is a
  **standalone repo** (the earlier one was mis-rooted at the Windows home dir).
- **Python 3.12** venv at `.venv/` (torch/faiss have no 3.13+ wheels; system
  also has a 3.14 that must NOT be used here). Activate:
  `.venv\Scripts\activate` — or call `.venv/Scripts/python.exe` directly.
- All deps installed and working. Ranges in `requirements.txt`, exact freeze
  in `requirements-lock.txt` (langchain 1.4, langgraph 1.2, pydantic 2.13,
  faiss-cpu 1.15, sentence-transformers 6.0, torch 2.14, groq 1.7, boto3, streamlit).
- `.env` exists (gitignored) and now holds a **real working `GROQ_API_KEY`**
  plus AWS creds. `.env.example` is back to blank placeholders.
- **Security note (2026-09-06):** those creds were briefly committed to the
  tracked `.env.example` in commit `31ded02`. That commit was HEAD and never
  pushed; it was amended to `5e75d2c` with the secret removed, and the secret
  is gone from all local history (`git log -S` is clean). Nothing leaked to
  `origin`. The user chose **not** to rotate the keys (judged low-risk).

## Commits so far

```
(uncommitted) Add the evaluation harness                          [build slice 8]
HEAD     Add the Streamlit chat UI                                [build slice 7]
         Wire the LangGraph graph behind hard guardrails          [build slice 6]
         Add the three orchestration nodes, tested in isolation   [build slice 5]
         Add provider-agnostic model access wrapper (src/llm.py)  [build slice 4]
         Add docs/SESSION_HANDOFF.md for cross-session context    (amended — see security note above)
019f1a0  Build vector index + validate hybrid retrieval (recall@3 = 0.80)
06320ec  Add curated respite-service dataset (13 SG services)
fd78e05  Scaffold RespiteSG V1 project
```
Run `git log --oneline -6` for exact hashes. A commit `2a38680` is in the OLD
home-dir repo — ignore it. Slices 1–7 are committed **and pushed** to
`origin/main` (`github.com/jy59-afk/caregiving`). Slice 8 is committed locally;
push is blocked from the agent environment — run `git push origin main` yourself.

## Completed build slices

### 1. Scaffold ✅
`src/ data/ docs/ tests/`, `.gitignore` (note: gitignore inline `#` comments
don't work — patterns are on their own lines), `.gitattributes`, `.env(.example)`,
`requirements*.txt`, `README.md`, `CLAUDE.md`, `docs/PROJECT_BRIEF.md`,
`conftest.py` (puts `src/` on sys.path), `pytest.ini`.

### 2. Data layer ✅
`data/services.json` — 13 real SG respite services (NTUC Health, Dementia
Singapore / New Horizon, St Luke's ElderCare, Homage, Apex Harmony Lodge,
All Saints Home, AWWA). Schema + provenance + honesty note in
[data/README.md](../data/README.md). Fees are **pre-subsidy approximations
from public sources**, flagged for re-verification. Coverage is deliberately
varied: 6 care types, 12 areas, S$19–120 budget span, intentional same-area
near-duplicates for disambiguation tests. `data/services.sample.json` stays
as the 3-record schema example.

### 3. Vector index + hybrid retrieval ✅
- `src/ingest.py` — flattens each record to one passage, embeds with local
  MiniLM (`EMBEDDING_BACKEND=local`) or Bedrock Titan (`=bedrock`), persists
  FAISS to `data/vector_index/` (gitignored build artifact).
- `src/retrieval_tool.py` — `search_services(profile, k=8)`: semantic search
  first, then hard-filter on `budget` / `area`, return top 3. FAISS index is
  **lazy-loaded** (import never requires a prebuilt index). Signature is the
  drop-in the Match & Rank node will call.
- `src/config.py` — `from config import settings`: typed pydantic view of `.env`.
- `tests/test_retrieval.py` — hard-filter unit tests + `recall@3` (10 labeled
  queries, skips until index built). **Current: recall@3 = 0.80 (8/10).**
- Rebuild + test: `python src/ingest.py && pytest -q`

### 4. Model access wrapper ✅
- `src/llm.py` — the single seam every node uses to call an LLM:
  - `generate(messages, *, model_role, max_tokens, temperature, json_mode)
    -> LLMResult` (raw text + token usage + which model/backend served it).
  - `complete(messages, *, model_role, response_schema=None, ...)
    -> BaseModel | str` — the node-facing call. With a `response_schema` it
    forces JSON, parses (tolerating ```json fences / stray prose), and
    validates against the pydantic schema; **any failure raises
    `LLMSchemaError`**. This is where the "every model output validated
    before use" guardrail lives, so every node inherits it.
  - `model_role` is `"extract"` (cheap/fast) or `"reasoning"` (stronger);
    `_model_id_for()` maps `(backend, role)` to the id in `settings`.
  - Backends: `groq` (dev) via the `groq` SDK, `bedrock` (demo) via `boto3`
    `bedrock-runtime` **Converse** API. Both SDKs are lazy-imported inside
    their dispatch branch — importing `llm` needs neither installed/configured.
    `_split_for_converse()` translates chat-completions messages into
    Converse's separate `system=[...]` + `messages=[{role, content:[{text}]}]`.
  - Exception hierarchy: `LLMError` -> `LLMBackendError` (transport) /
    `LLMSchemaError` (guardrail) / bare `LLMError` (bad `LLM_BACKEND`).
- `src/check_env.py` — smoke test: prints active backend/models, sends one
  `model_role="extract"` completion, prints reply + token usage, exits
  non-zero on any failure. **Passes against Groq now.**
- `tests/test_llm.py` — 16 tests, no network. Monkeypatches `llm.generate`
  to a canned `LLMResult` and checks: free-text passthrough, JSON parsing
  (bare / fenced / prose-wrapped), json_mode + schema-instruction injection,
  and that non-JSON / schema-violating / wrong-type output **raises**. Plus
  `_model_id_for` mapping, unknown-backend guard, missing-key guard.
- **Groq model ids changed.** The account behind the current `GROQ_API_KEY`
  only exposes `openai/gpt-oss-{20b,120b}`, `qwen/qwen3.*`, `groq/compound*`
  (no Llama 3.1/3.3). Defaults are now `GROQ_EXTRACT_MODEL=openai/gpt-oss-20b`
  and `GROQ_REASONING_MODEL=openai/gpt-oss-120b` in `config.py` / `.env` /
  `.env.example`. These are reasoning-style models — they spend hidden tokens
  before the visible answer, so keep `max_tokens` generous (>=256 even for
  one-word replies). The `groq` SDK surfaces only the final answer in
  `choice.message.content` (reasoning goes in a separate `.reasoning` field
  we ignore), so the wrapper needs no special handling.
- Bedrock path is written but **unexercised** — no AWS access yet. Verify
  `converse()` response shape (`output.message.content[].text`, `usage`)
  against a real call when Bedrock creds land.
- Run: `python src/check_env.py` · `pytest -q` (19 tests, all green)

### 5. Orchestration nodes ✅
The three worker nodes, **each written and unit-tested in isolation — no
LangGraph graph yet** (that is slice 6).
- `src/state.py` — the one place the state contract + every output schema
  lives:
  - `RespiteState` (`TypedDict, total=False`) — CLAUDE.md's keys verbatim:
    `messages`, `caregiver_profile`, `candidate_matches`, `drafted_message`,
    `consent_given`, `iteration_count`. `new_state(messages)` builds a fresh
    one with every key present-and-empty; `render_transcript()` flattens
    messages to a `Caregiver:/Assistant:` transcript (drops system + blank turns).
  - Pydantic schemas, all with `Field(description=...)` written *for the model*
    (llm.complete injects them as the JSON-schema contract): `CaregiverProfile`
    (needs_description required; budget/area/schedule/relationship/
    clarifying_question optional), `RankedMatch` + `RankedMatches`,
    `DraftedMessage` (service_name/subject/body).
- `src/intake.py` — `run_intake(state) -> {"caregiver_profile", "iteration_count"}`.
  `llm.complete(model_role="extract", response_schema=CaregiverProfile)`.
  Empty/blank transcript → returns an empty profile carrying a
  `clarifying_question`, **without** calling the model. Schema failure
  propagates (not swallowed).
- `src/match_rank.py` — `run_match_and_rank(state) -> {"candidate_matches",
  "iteration_count"}`. Stage 1: `retrieval_tool.search_services(profile.model_dump())`
  (budget/area hard-filtered there, not by the model). Stage 2: LLM rerank
  (`model_role="reasoning"`, `response_schema=RankedMatches`) reorders / drops
  / writes the one-line "why". **`_reconcile()` guardrail:** model owns order +
  `why` only; `name` must match a retrieved candidate (invented/dup names
  dropped); `grounding_passage` + `similarity_score` are always copied from
  retrieval. Empty retrieval → `[]`, model not called. Missing profile → `ValueError`.
- `src/explain_draft.py` — `run_explain_and_draft(state) -> {"drafted_message",
  "iteration_count"}`. Drafts an enquiry message for `candidate_matches[0]` only,
  grounded in that match's passage (`model_role="reasoning"`,
  `response_schema=DraftedMessage`). No matches → `drafted_message=None`, model
  not called. **No send/book — returns draft text only.**
- Every node bumps `iteration_count` even on its early-return path.
- Tests: `tests/test_intake.py` (6), `tests/test_match_rank.py` (8),
  `tests/test_explain_draft.py` (5) — all mock `llm.complete` (and, for match,
  `match_rank.search_services`); no network. **`pytest -q` = 35 green.**
- Verified end-to-end once against **real Groq + real FAISS index** (scratch
  script, not committed): intake→match→draft on the Belinda-style prompt
  produced a valid profile, a grounded match, and a sensible enquiry draft.
  Note: the rerank legitimately returns <3 matches when only 1–2 candidates
  genuinely fit.

### 6. Guardrails + wired graph ✅
- `src/guardrails.py` — the three hard guardrails in one auditable module:
  - `iteration_cap_reached(state)` — `state["iteration_count"] >=
    settings.max_iterations` (live read, nothing baked in at import). This is
    the routing check the graph consults after every worker node.
  - `ALLOWED_TOOLS = frozenset({"search_services", "draft_message"})` — the
    closed tool set. `assert_tool_allowed(name)` raises `GuardrailViolation`
    on anything else.
  - `scan_source_for_action_tools()` — static regex scan of `src/*.py` for a
    top-level `def send_*/book_*/submit_*/call_*/...`. Returns `[]` today; the
    test fails loudly if a real-world-action tool is ever added.
  - Re-exports `LLMSchemaError` so the graph layer names the schema guardrail
    by one symbol.
- `src/graph.py` — `build_graph()` compiles the LangGraph state machine;
  `run(messages, *, consent_given=False)` builds fresh state and invokes it.
  Topology: `Intake → Match & Rank → Explain & Draft → Consent gate → Output
  → END`, with conditional edges:
  - after Intake: cap tripped **or** `profile.clarifying_question` set → Output.
  - after Match & Rank: cap tripped **or** empty `candidate_matches` → Output.
  - `explain_draft → consent_gate → output` is unconditional.
  - `run_consent_gate` is the HITL seam — **no sender exists**, so it only
    coerces `consent_given` to bool and performs no side effect.
  - `run_output` picks one of four `outcome`s and writes `final_response`
    (the caregiver-facing text): `matches_ready` / `no_matches` /
    `needs_clarification` / `stopped_iteration_cap`.
- `src/state.py` gained two **Output-only** keys: `outcome` (Literal) and
  `final_response` (str), plus the `Outcome` type alias. Worker nodes never
  touch them.
- Tests: `tests/test_guardrails.py` (9) + `tests/test_graph.py` (7) — model
  and `search_services` mocked, no network / no FAISS index. Cover: cap
  boundary, allow-list, source scan (+ a planted-sender sanity check), the
  happy path through all nodes, both short-circuit branches, the cap forcing
  early Output, and `LLMSchemaError` aborting the whole run. **`pytest -q` =
  55 green.**
- Not yet exercised end-to-end against real Groq + real index (slice 5's
  scratch script predates the graph). Worth one manual `python src/graph.py`
  run once the index is built.

## Known issues / decisions

- **2 recall@3 misses** are queries framed around the caregiver's situation
  ("I have to fly overseas…") not the care need. Expected to be fixed by the
  Match & Rank node's LLM reranking + Titan embeddings, not by hacking the
  dataset. Don't chase 1.0 with MiniLM alone.
- **langchain-community is being sunset.** `src/ingest.py` still imports
  `langchain_community.vectorstores.FAISS` and the deprecated
  `HuggingFaceEmbeddings`/`BedrockEmbeddings` path. Migrate to
  `langchain-huggingface` + `langchain-aws` eventually (FAISS wrapper has no
  standalone package yet). Warnings are filtered in `pytest.ini`. Not urgent.
- Model choice: Groq for dev, Bedrock Converse for demo — switch via
  `LLM_BACKEND`. Dev models are now `openai/gpt-oss-{20b,120b}` (see slice 4
  note). Bedrock stays Haiku 4.5 (extract) / Sonnet 4.5 (reasoning); ids in
  `.env.example` / `config.py`.

### 7. Thin Streamlit chat UI ✅
- `src/app.py` — one chat page over `graph`. Not the graded component; owns no
  matching logic, prompts, or schema.
  - `run_pipeline(messages)` mirrors `graph.run()` but uses
    `compiled.stream(..., stream_mode="updates")` so every node logs one line
    (`node intake -> ['caregiver_profile', 'iteration_count']`) — cheap
    node-transition logging for the demo video, per slice 9's ask. Merges the
    per-node deltas back with plain `dict.update` (RespiteState has no
    reducers, so that matches LangGraph's own merge).
  - Pre-flight: `index_is_built()` checks `settings.index_path/index.faiss`;
    missing → friendly "run `python src/ingest.py`" `st.error` + `st.stop()`,
    no traceback.
  - Every user turn calls `run_pipeline(full transcript)` — the graph is
    stateless between calls, so a follow-up answer to a clarifying question is
    just re-extracted with more context.
  - `outcome == "matches_ready"` → assistant bubble (the Output node's
    `final_response`, which already includes the draft inline) **plus**
    structured blocks: "Options that fit" (name + grounded `why` + an expander
    with the grounding passage) and "Draft enquiry to …" (Subject + Message in
    editable `st.text_area`s — the copy surface; **no send button, by design**).
  - `outcome == "needs_clarification"` → the `final_response` *is* the question;
    just shown in the bubble, answered in the same input.
  - `LLMError` (backend down, or a schema guardrail firing) is caught: the user
    turn is popped so history stays clean and an `st.error` invites a retry.
  - `md_safe()` escapes `$` before every `st.markdown` — a drafted "S$85 per
    session" was rendering as garbled LaTeX otherwise.
  - Sidebar: backend name, `max_iterations`, last `outcome`, "Start over".
- `.claude/launch.json` — `respite-ui` config so the preview tooling / `/run`
  can start `streamlit run src/app.py` on port 8501.
- **Slice-7 fix (committed with this slice):** `src/intake.py` `max_tokens`
  512 → 1536. gpt-oss-20b spends hidden reasoning tokens before the JSON;
  terse input ("I need a break") ironically needs *more* headroom and was
  hitting a Groq 400 `json_validate_failed`. Verified: 512 fails, 1024+ passes.
  All 55 tests still green (no test asserted on the old value).
- **Not committed:** run `git add -A && git commit` yourself (push is blocked
  from the agent env). Files: `src/app.py` (new), `.claude/launch.json` (new),
  `src/intake.py` (edit), `src/graph.py` (demo-prompt edit — see below),
  `CLAUDE.md`, this file.
- Verified end-to-end against **real Groq + real FAISS index**: `matches_ready`
  (Toa Payoh dementia prompt → New Horizon Centre + a grounded draft, `$`
  rendering clean), `needs_clarification` (vague prompt → "Could you tell me
  who you are caring for?"), and node logging all confirmed in the browser.

## Known issues / decisions  (slice 7 additions)

- **The old graph.py demo prompt returned `no_matches`.** It asked for weekday
  daytime cover *in Bishan* under $40 — but Bishan has exactly one record
  (St Luke's **weekend** respite, $19), so retrieval → rerank correctly emptied
  the shortlist. That is right behaviour, wrong demo. `src/graph.py.__main__`
  now uses "near Toa Payoh … above $85", which lands on the dementia day-care
  records. **If the pitch persona (Belinda) is tied to a specific town, either
  add a weekday day-care record for it to `data/services.json` or set the demo
  script in that town's actual coverage.** Do not widen the hard filters to
  paper over this — budget/area being real filters is the point.
- The `matches_ready` view shows the draft twice (inline in the assistant
  bubble via `final_response`, then again in the editable Subject/Message
  boxes). Left as-is: the bubble is the narrative, the boxes are the tool. If
  it grates in the demo, trim the draft out of `run_output`'s `final_response`
  in `src/graph.py` (slice 6 code) rather than string-munging in the UI.

### 8. Evaluation harness ✅
- `src/evaluate.py` — runs 8 scripted caregiver scenarios (`SCENARIOS`) through
  `graph.run` and reports the evaluation-slide numbers:
  - `_install_instrumentation()` monkeypatches `llm.complete` and
    `match_rank.search_services` with counting shims; `_COUNTER` is zeroed per
    scenario (in `run_scenario`) and summed for the headline rates.
  - **schema-validation pass rate** = `(llm_calls - schema_failures) / llm_calls`
    (a `LLMSchemaError` is tallied *and* re-raised — the run aborts, as designed).
  - **tool-call success rate** = `(tool_calls - tool_failures) / tool_calls`.
  - **task-completion rate** = scenarios where `outcome == expected_outcome`.
  - **answer fidelity (proxy)** = `check_fidelity_proxy()`: every shown match has
    a non-empty `why` + grounding passage, no draft body matches
    `_BOOKING_LANGUAGE`. `--judge` adds `judge_fidelity()` — an LLM pass rating
    each `why` SUPPORTED/UNSUPPORTED against its passage (uses `_REAL_GENERATE`
    so it doesn't pollute `_COUNTER`). Full why/passage/draft text is dumped to
    the report appendix for manual review.
  - **recall@3** = `compute_recall_at_3()`, which imports `LABELED_QUERIES` from
    `tests/test_retrieval.py` by file path (single source of truth) and calls
    the *unwrapped* `retrieval_tool.search_services`.
  - Writes a slide-ready markdown table to `docs/EVALUATION.md`; also prints it.
  - The iteration-cap scenario carries `max_iterations=1`; `run_scenario`
    swaps `settings.max_iterations` for that run and restores it in `finally`.
  - Exit code: non-zero unless schema + tool rates are both 1.0 (task-completion
    is expected to wobble with the model, so it doesn't fail the process).
- `tests/test_eval.py` — calls `evaluate.run_suite()` and asserts: schema rate
  == 1.0, tool rate == 1.0, task-completion >= 0.75, 0 fidelity issues,
  recall@3 >= 0.70. **Double-gated** behind `RUN_EVAL=1` *and* a built index, so
  `pytest -q` stays offline (55 passed, 1 skipped).
- **Last real run (Groq, `openai/gpt-oss-{20b,120b}`):** schema 100% (16/16),
  tool 100% (6/6), task-completion 8/8, 0 fidelity issues, recall@3 0.80. Run:
  `python src/evaluate.py` · `RUN_EVAL=1 pytest tests/test_eval.py -v -s`.
- **Not committed yet:** `src/evaluate.py`, `tests/test_eval.py`,
  `docs/EVALUATION.md` (all new), `CLAUDE.md`, this file. `git push origin main`
  yourself (blocked from the agent env).

---

## NEXT STEP — Build slice 9: polish

1. README pass (quickstart, the `python src/evaluate.py` + `streamlit run`
   commands, the guardrail story).
2. Comment-density pass over any thin spots.
3. Richer node-transition logging + a short demo-video script (the eval table
   from `docs/EVALUATION.md` is the evaluation slide).
4. Optional: `--judge` fidelity number for the slide; the `langchain-community`
   migration (still not urgent).
