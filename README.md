# RespiteSG / Respite Navigator

An agentic chatbot that matches Singapore family caregivers to respite-care
services — a **matching and activation layer**, not a bigger directory. Built
for the IGNITE Agentic AI Hackathon 2026.

See [CLAUDE.md](CLAUDE.md) for the working context and
[docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) for the full problem statement
and evidence.

## What it does (V1)

1. **Intake** — the caregiver describes their situation in plain language.
2. **Match & Rank** — hybrid retrieval over ~15 curated services: semantic
   similarity on the free-text need, then hard filters on budget and area.
3. **Explain & Draft** — 2–3 options each with a grounded "why this fits",
   plus a drafted first-enquiry message.
4. **Human-in-the-loop** — nothing is sent or booked; the caregiver reviews
   and sends the draft themselves.

## Project layout

| Path | What's there |
|---|---|
| `src/ingest.py` | Builds the FAISS vector index from `data/services.json` |
| `src/retrieval_tool.py` | `search_services(profile)` — hybrid retrieval tool the Match & Rank node calls |
| `src/config.py` | Typed, validated view of `.env` (`from config import settings`) |
| `data/services.sample.json` | Schema example (committed) |
| `data/services.json` | The curated working dataset (committed) |
| `data/vector_index/` | Built index — **not committed**, rebuild with `python src/ingest.py` |
| `tests/` | Unit tests + the `recall@3` retrieval eval |
| `docs/` | Project brief, strategy guide, source PDFs in `reference/` |

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
- Leave `EMBEDDING_BACKEND=local` to use the on-device embedding model (no
  keys needed). Switch to `bedrock` + AWS creds for the demo build, then
  rebuild the index.

## Run

```bash
python src/ingest.py             # build data/vector_index/ from data/services.json (downloads the MiniLM model on first run)
python src/retrieval_tool.py     # smoke test: prints top matches for an example profile
pytest -q                        # unit tests + recall@3 (skips until the index exists)
```

Current retrieval baseline: **recall@3 = 0.80** (8/10 labeled queries) with raw
MiniLM cosine similarity and no reranking. The two misses are queries phrased
around the *caregiver's* situation rather than the care need ("I have to fly
overseas…"); the Match & Rank node's LLM reranking and Bedrock Titan
embeddings are expected to close that gap.

## Guardrails (responsible-AI design)

- **No send/book tool exists in V1.** The agent drafts; the caregiver sends.
- Every model output is validated against a Pydantic schema before use.
- The orchestration loop has a hard iteration cap (`MAX_ITERATIONS` in `.env`).
- `allowed_tools` is an explicit allow-list: `search_services` + `draft_message` only.

## Status

- [x] Repo scaffold, config, retrieval starter
- [x] Curated dataset — 13 SG respite services ([data/README.md](data/README.md))
- [x] Vector index + hybrid retrieval — `recall@3 = 0.80` on 10 labeled queries
- [ ] Model access wrapper (Groq → Bedrock) + smoke test
- [ ] LangGraph nodes: Intake, Match & Rank, Explain & Draft
- [ ] Guardrails + tests
- [ ] Streamlit chat UI
- [ ] Evaluation harness
