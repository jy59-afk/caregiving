# RespiteSG / Respite Navigator

An agentic chatbot that matches Singapore family caregivers to respite-care
services — a **matching and activation layer**, not a bigger directory. Built
for the IGNITE Agentic AI Hackathon 2026.

> A caregiver who has already heard that respite care exists needs a fast,
> judgment-free way to find and act on the one option that fits their week and
> their wallet — because **82.83% of caregivers aware that respite services
> exist have still never used one** (SMU ROSA Research Brief, Mar 2025), not
> for lack of options, but because matching one to their specific hours,
> budget and care needs takes more time and certainty than a caregiver
> already stretched thin has to spare.

See [CLAUDE.md](CLAUDE.md) for the working context,
[docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) for the full problem statement
and evidence, and [docs/EVALUATION.md](docs/EVALUATION.md) for the latest
evaluation numbers.

## What it does (V1)

```
 Intake ─▶ Match & Rank ─▶ Explain & Draft ─▶ Consent gate ─▶ Output
   │            │                                (HITL)
   │            └─(nothing fits)──────────────────────────────▶ Output
   └─(needs a clarifying answer, or hit the step cap)─────────▶ Output
```

1. **Intake** — the caregiver describes their situation in plain language; a
   cheap model extracts a typed `CaregiverProfile` (need, budget, area,
   schedule). Missing a must-have fact → it asks one clarifying question.
2. **Match & Rank** — hybrid retrieval over the curated dataset: semantic
   similarity on the free-text need widens the candidate set, then **hard
   filters on budget and area** cut it, then a stronger model reranks the
   survivors and writes a one-line, passage-grounded "why this fits".
3. **Explain & Draft** — drafts a first-enquiry message for the top match,
   grounded in that service's real description.
4. **Consent gate → Output** — the human-in-the-loop seam. **Nothing is sent
   or booked.** The caregiver reviews the draft and sends it themselves.

## Project layout

| Path | What's there |
|---|---|
| `src/ingest.py` | Builds the FAISS vector index from `data/services.json` |
| `src/retrieval_tool.py` | `search_services(profile)` — the hybrid-retrieval tool Match & Rank calls |
| `src/llm.py` | Provider-agnostic model access; the schema-validation guardrail lives here |
| `src/state.py` | The state contract (`RespiteState`) + every Pydantic output schema |
| `src/intake.py`, `src/match_rank.py`, `src/explain_draft.py` | The three worker nodes |
| `src/guardrails.py` | Iteration cap, tool allow-list, source scan for action tools |
| `src/graph.py` | LangGraph wiring + node-transition tracing (`_traced`) |
| `src/app.py` | The thin Streamlit chat UI (not the graded component) |
| `src/evaluate.py` | 8-scenario evaluation harness → `docs/EVALUATION.md` |
| `data/services.json` | 13 curated SG respite services ([data/README.md](data/README.md)) |
| `data/vector_index/` | Built index — **not committed**, rebuild with `python src/ingest.py` |
| `tests/` | `pytest` — unit tests + the `recall@3` retrieval eval |

## Setup

Requires **Python 3.12** (`torch` / `faiss-cpu` have no 3.13+ wheels yet).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell/CMD)
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt   # or: pip install -r requirements-lock.txt (exact tested freeze)
copy .env.example .env            # Windows  (cp on macOS/Linux)
```

Then fill in `.env`:

- `GROQ_API_KEY` — from <https://console.groq.com/keys> (free; used for dev).
  Switch to Bedrock for the demo build with `LLM_BACKEND=bedrock` + AWS creds.
- Leave `EMBEDDING_BACKEND=local` to use the on-device embedding model (no
  keys needed). Switch to `bedrock` for Titan embeddings, then rebuild the index.

## Run

```bash
python src/ingest.py             # build data/vector_index/ (downloads the MiniLM model on first run)
python src/check_env.py          # smoke-test the model backend + print the active models
python src/graph.py              # run one scripted caregiver turn end-to-end, with the node trace
streamlit run src/app.py         # the chat UI on http://localhost:8501
python src/evaluate.py           # 8 scenarios end-to-end → refreshes docs/EVALUATION.md
pytest -q                        # unit tests + recall@3 (offline; the eval suite is gated behind RUN_EVAL=1)
```

`python src/graph.py` and `streamlit run src/app.py` both print a node-by-node
trace to the terminal:

```
respite.graph  -> intake         (step 0)
respite.graph  <- intake         1334 ms  wrote ['caregiver_profile', 'iteration_count']
respite.graph     route: intake -> match_rank
respite.graph  -> match_rank     (step 1)
...
respite.graph     outcome: matches_ready
```

## Guardrails (responsible-AI design)

The agent is fenced in by **mechanism, not prompt wording** — all three checks
live in [`src/guardrails.py`](src/guardrails.py):

- **No send/book tool exists in V1.** `ALLOWED_TOOLS` is a closed set —
  `search_services` + `draft_message` only — and `scan_source_for_action_tools()`
  greps `src/` for any `def send_* / book_* / submit_* / call_* ...`; the test
  suite fails loudly if one is ever added. The agent drafts; the caregiver sends.
- **Every model output is schema-validated before use.** `llm.complete(...,
  response_schema=…)` forces JSON, parses it, validates against a Pydantic
  model, and raises `LLMSchemaError` on any failure — so every node inherits
  the guarantee and a bad output aborts the run instead of being shown.
- **The orchestration loop has a hard iteration cap** (`MAX_ITERATIONS` in
  `.env`). `iteration_cap_reached()` is checked in the conditional edge after
  every worker node; tripping it routes straight to Output.

Latest evaluation run: schema-validation **100%**, tool-call success **100%**,
task-completion **8/8**, 0 answer-fidelity issues, retrieval **recall@3 = 0.80**.
Full table in [docs/EVALUATION.md](docs/EVALUATION.md).

## Status — V1 build complete

- [x] Repo scaffold, config, retrieval starter
- [x] Curated dataset — 13 SG respite services ([data/README.md](data/README.md))
- [x] Vector index + hybrid retrieval — `recall@3 = 0.80` on 10 labeled queries
- [x] Model access wrapper (Groq ↔ Bedrock, one-line switch) + smoke test
- [x] LangGraph nodes: Intake, Match & Rank, Explain & Draft
- [x] Guardrails (iteration cap, tool allow-list, source scan) + wired graph
- [x] Streamlit chat UI
- [x] Evaluation harness → [docs/EVALUATION.md](docs/EVALUATION.md)
- [x] Polish — README, node-transition trace, [demo script](docs/DEMO_SCRIPT.md)
