# Session handoff — paste this into a new session

> Purpose: a compact state snapshot so a new session doesn't re-read prior
> chat history. Read this + [CLAUDE.md](../CLAUDE.md) and you have full context.
> Update this file at the end of each working session.

**Last updated:** 2026-09-06 · after build slice 3 (vector index + retrieval)

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
- `.env` exists (gitignored), seeded from `.env.example`, **keys are blank** —
  `GROQ_API_KEY` and AWS creds still need filling in by the user.

## Commits so far

```
019f1a0  Build vector index + validate hybrid retrieval (recall@3 = 0.80)
06320ec  Add curated respite-service dataset (13 SG services)
fd78e05  Scaffold RespiteSG V1 project
```
(A 4th commit `2a38680` is in the OLD home-dir repo — ignore it.)

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
  `LLM_BACKEND`. Haiku 4.5 for extraction/classification, Sonnet 4.5 for
  ranking rationale + drafting. Model ids are in `.env.example` / `config.py`.

---

## NEXT STEP — Build slice 4: model access wrapper

**Goal:** one thin module that the orchestration nodes call for LLM
completions, provider-agnostic, so switching Groq → Bedrock is a config change.

**Deliverables:**
1. `src/llm.py` — e.g. `complete(messages, *, model_role: Literal["extract","reasoning"], response_schema: type[BaseModel] | None = None) -> BaseModel | str`
   - Reads `settings.llm_backend`; dispatches to Groq (`groq` SDK) or Bedrock
     Converse (`boto3` `bedrock-runtime`).
   - `model_role` picks the cheap vs strong model id from `settings`.
   - If `response_schema` given: instruct JSON output, parse, validate against
     the pydantic schema, raise on failure (this is the "every model output
     validated" guardrail — keep it here so every node inherits it).
   - Comment every function and non-trivial line (project standard).
2. `src/check_env.py` — smoke test in the training-lab style: prints which
   backend is active, sends one trivial completion, prints the reply + token
   usage, exits non-zero on failure. Run: `python src/check_env.py`
3. `tests/test_llm.py` — test schema-validation logic with a **mocked**
   provider response (no network in unit tests). Test that a malformed /
   non-conforming response raises.

**Acceptance:** `python src/check_env.py` succeeds against Groq once the user
adds `GROQ_API_KEY`; `pytest -q` still green; no provider SDK imported at
module top level unless it's always installed (lazy-import boto3/groq inside
the dispatch branch, mirroring `ingest.get_embeddings`).

**Then commit** with a message in the established style, and update this file.

## Build slices still after that

5. Orchestration nodes (Intake → Match & Rank → Explain & Draft), each tested
   in isolation before wiring the LangGraph graph.
6. Guardrails — iteration cap from `MAX_ITERATIONS`, `allowed_tools`
   allow-list, test that no send/book tool is reachable.
7. Streamlit chat UI (thin).
8. Evaluation harness — 5–8 scripted scenarios + the metrics slide.
9. Polish — README, comment pass, node-transition logging for the demo video.
