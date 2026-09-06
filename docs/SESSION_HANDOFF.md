# Session handoff — paste this into a new session

> Purpose: a compact state snapshot so a new session doesn't re-read prior
> chat history. Read this + [CLAUDE.md](../CLAUDE.md) and you have full context.
> Update this file at the end of each working session.

**Last updated:** 2026-09-06 · after build slice 5 (orchestration nodes)

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
HEAD     Add the three orchestration nodes, tested in isolation   [build slice 5]
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

## NEXT STEP — Build slice 6: guardrails + wire the graph

Slice 5 built the three nodes (`src/intake.py`, `src/match_rank.py`,
`src/explain_draft.py`) and the state contract (`src/state.py`). Nothing wires
them together yet — do that here, behind explicit guardrails.

**Deliverables:**
1. **Iteration cap** — a routing check that reads `state["iteration_count"]`
   against `settings.max_iterations` (already in `config.py`, default 6) and
   forces the graph to a terminal "Output" node when hit. Every node already
   bumps `iteration_count`, including on early returns.
2. **`allowed_tools` allow-list** — the agent may only ever reach
   `search_services` + `draft_message`-equivalent (the Explain & Draft node).
   Add a test that asserts no send/book/call tool is importable or registered.
   (There is no such tool in the repo — the test guards against one being
   added later.)
3. **Schema validation** — already enforced inside `llm.complete`; slice 6 just
   needs a test at the graph level that a node raising `LLMSchemaError` aborts
   the run rather than emitting unvalidated text.
4. **Wire the LangGraph graph** — `src/graph.py`: `Intake -> Match & Rank ->
   Explain & Draft -> (HITL consent gate) -> Output`. The consent gate reads
   `state["consent_given"]`; it does not auto-send anything (there is no sender).
   Conditional edges: empty `candidate_matches` routes to a "loosen constraints"
   Output branch; `clarifying_question` set on the profile routes back to ask
   the caregiver.

**Acceptance:** `pytest -q` green; graph runs end-to-end on a scripted
transcript with `llm` mocked; iteration cap and allow-list each have a test.

## Build slices still after that

7. Streamlit chat UI (thin).
8. Evaluation harness — 5–8 scripted scenarios + the metrics slide
   (schema-validation pass rate, tool-call success rate, task-completion
   rate, answer fidelity, recall@3).
9. Polish — README, comment pass, node-transition logging for the demo video.
