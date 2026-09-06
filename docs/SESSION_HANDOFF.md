# Session handoff — paste this into a new session

> Purpose: a compact state snapshot so a new session doesn't re-read prior
> chat history. Read this + [CLAUDE.md](../CLAUDE.md) and you have full context.
> Update this file at the end of each working session.

**Last updated:** 2026-09-06 · after build slice 4 (model access wrapper)

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
HEAD     Add provider-agnostic model access wrapper (src/llm.py)  [build slice 4]
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

## NEXT STEP — Build slice 5: orchestration nodes

**Goal:** the three worker nodes of the LangGraph flow (Intake → Match & Rank
→ Explain & Draft), **each written and tested in isolation** before any graph
wiring. Resist building the graph in this slice.

**State object** (define once, probably `src/state.py`): typed — `messages`,
`caregiver_profile`, `candidate_matches`, `drafted_message`, `consent_given`,
`iteration_count`. Pydantic model or `TypedDict`; keep heavy objects here, not
in prompts.

**Deliverables (one node at a time):**
1. **Intake node** — takes the conversation so far, calls
   `llm.complete(..., model_role="extract", response_schema=CaregiverProfile)`
   to pull `needs_description`, `budget`, `area` (+ whatever else Match needs).
   `CaregiverProfile` is the Pydantic schema — this is the first real user of
   the `llm.py` guardrail. Test with 3–4 scripted transcripts.
2. **Match & Rank node** — calls `retrieval_tool.search_services(profile)`
   (already built), then optionally an LLM rerank pass
   (`model_role="reasoning"`) to reorder / drop the ≤3 candidates and attach a
   one-line "why". Output validated against a `RankedMatch` schema. Test that
   budget/area hard filters are respected end-to-end.
3. **Explain & Draft node** — for the top match, drafts the message the
   caregiver would send to the service, grounded in the retrieved
   `grounding_passage` (no invented facts). `model_role="reasoning"`, output
   validated. **No send/book tool** — it only returns draft text.

**Acceptance:** each node importable and unit-tested in isolation with a
mocked or real `llm` call; `pytest -q` green; no LangGraph graph yet.

**Then commit** in the established style and update this file.

## Build slices still after that

6. Guardrails — iteration cap from `MAX_ITERATIONS`, `allowed_tools`
   allow-list, test that no send/book tool is reachable. Then wire the graph.
7. Streamlit chat UI (thin).
8. Evaluation harness — 5–8 scripted scenarios + the metrics slide
   (schema-validation pass rate, tool-call success rate, task-completion
   rate, answer fidelity, recall@3).
9. Polish — README, comment pass, node-transition logging for the demo video.
