# Session handoff — paste this into a new session

> Purpose: a compact state snapshot so a new session doesn't re-read prior
> chat history. Read this + [CLAUDE.md](../CLAUDE.md) and you have full context.
> Update this file at the end of each working session.

**Last updated:** 2026-09-06 · after build slice 6 (guardrails + wired graph)

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
HEAD     Wire the LangGraph graph behind hard guardrails          [build slice 6]
         Add the three orchestration nodes, tested in isolation   [build slice 5]
         Add provider-agnostic model access wrapper (src/llm.py)  [build slice 4]
         Add docs/SESSION_HANDOFF.md for cross-session context    (amended — see security note above)
019f1a0  Build vector index + validate hybrid retrieval (recall@3 = 0.80)
06320ec  Add curated respite-service dataset (13 SG services)
fd78e05  Scaffold RespiteSG V1 project
```
Run `git log --oneline -6` for exact hashes. A commit `2a38680` is in the OLD
home-dir repo — ignore it. `origin/main` is still at `fd78e05`; everything
above it is local and **unpushed** (push blocked from the agent environment —
run `git push origin main` yourself).

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

---

## NEXT STEP — Build slice 7: thin Streamlit chat UI

The graph (`src/graph.py`) is done and green. Slice 7 is the thinnest possible
chat front-end over it — explicitly **not** the graded component, so keep it
minimal.

**Deliverables:**
1. `src/app.py` (or `streamlit_app.py`) — a single chat page: text input,
   message history, and on each user turn call `graph.run(messages)` and render
   `final_response`. Show `outcome` somewhere unobtrusive (caption / sidebar).
2. When `outcome == "matches_ready"`, also surface the `candidate_matches`
   (name + why + a peek at the grounding passage) and the `drafted_message`
   (subject + body) in a copy-friendly block — the caregiver sends it, the app
   never does.
3. When `outcome == "needs_clarification"`, the `final_response` *is* the
   question — just show it and let them answer in the same input.
4. Backend is picked by `.env` (`LLM_BACKEND=groq` for dev). The index must be
   built (`python src/ingest.py`) or Match & Rank raises on load — catch that
   and show a friendly "run ingest first" message rather than a stack trace.
5. Node-transition logging can wait for slice 9, but if it's cheap, log each
   node entry now so the demo video has something to show.

**Acceptance:** `streamlit run src/app.py`, type the Belinda prompt, get a
ranked list + a draft; type a vague prompt, get the clarifying question back.

## Build slices still after that

8. Evaluation harness — 5–8 scripted scenarios + the metrics slide
   (schema-validation pass rate, tool-call success rate, task-completion
   rate, answer fidelity, recall@3).
9. Polish — README, comment pass, node-transition logging for the demo video.
