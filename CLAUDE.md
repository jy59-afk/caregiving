# RespiteSG / Respite Navigator — project context for Claude Code

Re-read this at the start of every session, then
[docs/SESSION_HANDOFF.md](docs/SESSION_HANDOFF.md) for current state + the next
step. Fuller detail lives in [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) and
[docs/Respite_Navigator_Hackathon_Strategy.md](docs/Respite_Navigator_Hackathon_Strategy.md).

## What this is

An agentic chatbot that matches Singapore family caregivers to respite-care
services. It is a **matching and activation layer**, not a bigger directory —
the gap is discovery and activation energy, not supply.

## Problem statement (POV)

> A family caregiver in Singapore who has already heard that respite care
> exists needs a fast, judgment-free way to find and act on the one option
> that actually fits their week and their wallet — because 82.83% of
> caregivers who are aware respite services exist have still never used one
> (SMU ROSA Research Brief, Mar 2025), not for lack of options, but because
> matching the right service to their specific hours, budget and care needs
> takes more time and certainty than a caregiver already stretched thin has
> to spare.

Persona: Belinda Seet, 63, part-time lecturer, primary caregiver for her
89-year-old mother with Alzheimer's; S$20–40/hour rules respite out before
she can assess fit. Budget is a **first-class filter**, never an afterthought.

## Architecture (V1 — the hackathon build)

Flow: **Intake → Match & Rank → Explain & Draft → (HITL consent) → Output**

| Layer | Choice | Notes |
|---|---|---|
| Interface | Minimal chat UI (Streamlit) | Thin; not the graded component |
| Orchestrate | LangGraph state graph, short single-purpose nodes | Not one giant loop |
| Tools | `search_services(profile) -> list[Match]`, `draft_message(service, profile) -> str` | Docstrings are the prompt — write them like UI copy |
| Guardrails | Hard iteration cap in state; `allowed_tools` allow-list = search + draft only (**no send/book tool exists in V1**); every model output validated against a Pydantic schema before use | This is the "responsible AI" talking point |
| Model access | Groq for dev iteration; Bedrock Converse API for demo/deploy (one-line switch via `LLM_BACKEND`) | |
| Model | Claude Haiku 4.5 for extraction/classification; Claude Sonnet 4.5 where matching nuance / drafting quality matters | Upgrade only where Haiku is measurably wrong |
| Data / KB | ~15 hand-checked service records (name, care_type, days, hours, cost_per_session, area, distance_km, **description**). Structured fields = hard filters; `description` is embedded into a FAISS index and semantically matched | Hybrid retrieval — see below |
| Deploy | Local FastAPI / Streamlit for the demo; Bedrock AgentCore Runtime as a stretch goal | |
| State | Typed object: `messages`, `caregiver_profile`, `candidate_matches`, `drafted_message`, `consent_given`, `iteration_count` | Keep heavy objects in state, not in the prompt |
| Observability | Log node transitions + tool-call outcomes | Feeds the demo video and evaluation slide |

**Hybrid retrieval:** embed the caregiver's free-text need → similarity search
returns a wider candidate set (k=8) → hard-filter out over-budget / wrong-area
→ keep top 3 by similarity. The retrieved passage is passed into Explain &
Draft as grounding so "why this fits" cites real service content.

## Repo layout

```
src/      ingest.py, retrieval_tool.py, config.py — and the LangGraph nodes as they land
data/     services.sample.json (committed), services.json (working dataset, committed),
          vector_index/ (build artifact, gitignored — rebuild with `python src/ingest.py`)
docs/     PROJECT_BRIEF.md, strategy doc, reference/ (source PDFs)
tests/    pytest — unit tests + recall@3 retrieval eval
```

- Secrets in `.env` (gitignored); template in `.env.example`.
- Dependencies in `requirements.txt`.
- Run scripts from the repo root: `python src/ingest.py`, `pytest`.

## Coding standard (non-negotiable for this project)

**Comment every function and every non-trivial line** so the logic stays
auditable for hackathon judges and for our own debugging. Match the comment
density already in `src/ingest.py` / `src/retrieval_tool.py`.

## Build order (one slice per session — resist doing the graph in one prompt)

1. ✅ Scaffold repo
2. ✅ Data layer — 13 curated SG respite services in `data/services.json` (see `data/README.md` for schema + provenance; figures are pre-subsidy approximations from public sources, re-verify before real use)
3. ✅ Vector index & retrieval tool — `python src/ingest.py` builds `data/vector_index/`; `recall@3 = 0.80` on 10 labeled queries in `tests/test_retrieval.py` (raw MiniLM, no reranking)
4. ✅ Model access wrapper — `src/llm.py` (`complete()` / `generate()`, provider-agnostic, schema-validation guardrail baked in); `python src/check_env.py` smoke test passes against Groq; `tests/test_llm.py` covers the validation logic with a mocked provider

## Environment

- **Python 3.12** venv at `.venv/` (torch/faiss have no 3.13+ wheels). Deps installed and working.
- Tested versions frozen in `requirements-lock.txt`; ranges in `requirements.txt`.

## langchain-community migration (cleanup, not urgent)

`langchain-community` is being sunset. Two imports still use it:
`langchain_community.vectorstores.FAISS` and (via the deprecated path)
`HuggingFaceEmbeddings` / `BedrockEmbeddings` in `src/ingest.py`. Migrate to
`langchain-huggingface` and `langchain-aws` (FAISS wrapper has no standalone
package yet). The deprecation warnings are filtered in `pytest.ini` meanwhile.
5. ✅ Orchestration nodes — `src/state.py` (state contract + output schemas), `src/intake.py`, `src/match_rank.py`, `src/explain_draft.py`; each unit-tested in isolation with `llm` mocked (`pytest -q` = 35 green). No graph yet.
6. Guardrails — iteration cap, allow-list, schema validation; test that no send/book tool is reachable
7. Interface — thinnest possible Streamlit chat
8. Evaluation harness — 5–8 scripted scenarios; report schema-validation pass rate, tool-call success rate, task-completion rate, answer fidelity, and recall@3
9. Polish — README, comments pass, node-transition logging for the demo video

## Guardrails that must stay true

- No tool that sends, books, or calls anyone exists in V1. The agent drafts;
  the caregiver sends.
- Every model output is schema-validated before it is used or shown.
- The orchestration loop has a hard iteration cap read from `MAX_ITERATIONS`.
