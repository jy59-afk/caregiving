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

## What it does (V1)

```
 Intake ──▶ Match & Rank ──▶ Explain & Draft ──▶ Consent gate ──▶ Output
   │             │                                                   ▲
   │             ├─(nothing fits, caregiver asks to act anyway)──────┤
   │             └─(nothing fits at all)───────────────────────────▶ Output
   └─(needs a clarifying answer, or hit the step cap)───────────────▶ Output
```

One LangGraph pipeline (`src/graph.py`), re-run on the full conversation
transcript every turn, does four things:

1. **Intake** — extracts a typed `CaregiverProfile` from the conversation so
   far: the care need in plain language, budget, area, schedule, and who the
   caregiver is caring for. It also picks up on two follow-up signals — that
   the caregiver explicitly asked to **draft/send an enquiry**, or that they
   **picked a specific option** from a list just shown (by name or by
   position, e.g. "the second one"). If a critical fact is still missing, it
   returns one clarifying question instead of guessing.
2. **Match & Rank** — hybrid retrieval: the free-text need is embedded and
   matched against the service dataset, the candidate set is cut down by
   **hard filters on budget and area**, and a model reranks what's left into
   an ordered shortlist (up to 4) with a one-line, passage-grounded "why this
   fits" for each. **If nothing clears every constraint**, it doesn't dead-end:
   it deterministically ranks the next-best services by location → budget →
   schedule (no model call) as a clearly-labelled fallback list.
3. **Explain & Draft** — drafts a ready-to-send enquiry (subject + body) for
   the top match. On the fallback path, it only drafts if the caregiver asked
   for it or named an option — and words the message as a first question,
   since a fallback option doesn't fully fit.
4. **Consent gate → Output** — the human-in-the-loop seam. **No tool that
   sends, books, or contacts anyone exists anywhere in this codebase.** The
   Output node assembles the caregiver-facing reply (the shortlist, or the
   fallback list, or the clarifying question) and, when there's a draft,
   presents it for the caregiver to review, edit, and send themselves.

**Product front-end:** the pipeline is served over HTTP by `src/api.py`
(FastAPI), which powers the Airbnb-style web app in `web/` — ranked photo
cards synced to a live Leaflet map, with a chat panel to describe the care
situation. When a draft is ready, the page offers an **"open in your email
app"** (`mailto:`) button, or a link to the centre's own web contact form —
so sending is always a step the caregiver takes in their own mail client,
never something the app does for them.

The graded proof-of-concept is the **Agentic AI pipeline** in `src/`
(`graph.py` and everything it calls); `web/` + `src/api.py` is the one
interface that ships with it.

---

## 1. Environment setup

### Requirements

- **Python 3.12** — `torch` and `faiss-cpu` don't ship 3.13+ wheels yet, so a
  newer interpreter will fail at `pip install`.
- **AWS Bedrock access** in `us-east-1` with model access granted for Claude
  Sonnet 4.5, Claude Haiku 4.5, and Cohere Embed English v3 — this is the
  stack the project was built and demoed on, and the only one documented
  here.
- **Run every command from the repo root** (the folder this README is in) —
  `conftest.py`, the `.env` loader in `src/config.py`, and the relative paths
  in `src/api.py`/`src/config.py` (`data/`, `web/`) are all anchored there.
  There are no extra `PYTHONPATH`/path variables to set by hand.

> The codebase also supports a free `LLM_BACKEND=groq` + on-device
> `EMBEDDING_BACKEND=local` mode for local dev iteration without an AWS
> account (see `.env.example`). It isn't documented as a setup path here
> because the graded/demo build — and this README — targets Bedrock only.

### 1.1 Create and activate a virtual environment

```powershell
# Windows (PowerShell or CMD), from the repo root
py -3.12 -m venv .venv
.venv\Scripts\activate
```

```bash
# macOS / Linux, from the repo root
python3.12 -m venv .venv
source .venv/bin/activate
```

Your shell prompt should now show `(.venv)`. Every command below assumes the
venv is active.

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

- `requirements.txt` uses version **ranges** (a fresh resolve, kept current).
- `requirements-lock.txt` is the exact, tested freeze from a working venv —
  use it instead (`pip install -r requirements-lock.txt`) if a fresh resolve
  of the ranges misbehaves on your machine.
- No Docker file is provided — the venv + `requirements.txt` above is the
  supported setup path.
- **`faiss-cpu` is required no matter what** — it's the vector-index engine
  retrieval is built on, unrelated to which LLM answers. `groq` and
  `sentence-transformers` are only imported if you switch `LLM_BACKEND`/
  `EMBEDDING_BACKEND` to the local dev mode above; they're listed in
  `requirements.txt` for that path but pip installing them costs nothing if
  you never use it.

### 1.3 Secrets and keys — `.env`

Real secrets are **never** committed; `.env` is git-ignored
(see [`.gitignore`](.gitignore)) and only the template, `.env.example`, is
tracked.

```powershell
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

Then open `.env` and set:

| Variable | Notes |
|---|---|
| `LLM_BACKEND` | `bedrock` |
| `AWS_REGION` | `us-east-1` |
| `AWS_PROFILE` (or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN`) | prefer an SSO profile (`aws sso login`) over pasted temporary keys, which expire mid-demo |
| `BEDROCK_EXTRACT_MODEL` | `us.anthropic.claude-haiku-4-5-...` — must use the `us.` cross-region inference-profile prefix; bare `anthropic.` ids aren't invocable |
| `BEDROCK_REASONING_MODEL` | `us.anthropic.claude-sonnet-4-5-...` |
| `EMBEDDING_BACKEND` | `bedrock` |
| `BEDROCK_EMBED_MODEL` | `cohere.embed-english-v3` |
| `MAX_ITERATIONS` | hard cap on the agent's orchestration loop (guardrail, see below) — `6` is a sensible default |

> **Never** paste real key values into a commit, a chat, or a shared file.
> If a `.env` with live keys is ever accidentally committed, treat those
> keys as compromised: rotate/regenerate them (AWS IAM) immediately, even
> after removing the file from git.

### 1.4 Verify the setup

```bash
python src/check_env.py
```

Prints the active backend/models, sends one test completion, and embeds a
test string via Bedrock — exits non-zero with a readable reason on any
failure (bad credentials, no model access granted, wrong region, etc.), so
run this before chasing a bug in the app itself.

---

## 2. Run it — see the product webpage

```bash
python src/ingest.py                                       # 1. build the FAISS index from data/services.json
python -m uvicorn api:app --app-dir src --port 8000         # 2. start the web app
```

Then open **http://localhost:8000** — that's the product webpage (ranked
photo cards + live map). Step 1 only needs to be re-run when
`data/services.json` changes; it calls Bedrock to embed each record.

`uvicorn ... --reload` (see `.claude/launch.json`) auto-restarts on code
changes if you're iterating on `src/` or `web/`.

### What you'll see in the terminal

Both `uvicorn` and `python src/graph.py` (below) print a node-by-node trace
of the agent's run:

```
respite.graph  -> intake         (step 0)
respite.graph  <- intake         1334 ms  wrote ['caregiver_profile', 'iteration_count']
respite.graph     route: intake -> match_rank
respite.graph  -> match_rank     (step 1)
...
respite.graph     outcome: matches_ready
```

### One scripted run, no server

```bash
python src/graph.py
```

Runs one hard-coded caregiver turn straight through the graph and prints the
trace + result — the fastest way to confirm the pipeline itself works,
independent of the web server.

### Tests and evaluation

```bash
pytest -q                 # unit tests + recall@3 retrieval check (offline, no API calls)
python src/evaluate.py    # 8 scripted end-to-end scenarios, calls the live model
```

`pytest` is safe to run with no API usage; `src/evaluate.py` makes real
Bedrock calls and prints a results table to the terminal.

---

## 3. Project structure — what's in every file

```
hackathon/
├── README.md                     this file
├── requirements.txt              dependency ranges — `pip install -r requirements.txt`
├── requirements-lock.txt         exact tested freeze — use if requirements.txt misbehaves
├── pytest.ini                    pytest config: test path, warning filters for known third-party deprecations
├── conftest.py                   puts src/ on sys.path so tests can `import config`, `import graph`, etc.
├── .env.example                  template for local secrets — copy to .env and fill in (see §1.3)
├── .env                          your real secrets — gitignored, never committed
├── .gitignore                    what git excludes (venv, caches, .env, build artifacts — see below)
├── .gitattributes                line-ending + binary-diff rules (LF in repo, binary for pdf/png/faiss/pkl)
├── .claude/launch.json           VS Code debug config for running the web app
│
├── src/                           the graded Agentic AI component
│   ├── config.py                  reads .env into a typed, validated Settings object (single source of config)
│   ├── state.py                   the LangGraph state contract (RespiteState) + every Pydantic I/O schema
│   ├── ingest.py                  builds data/vector_index/ (FAISS) from data/services.json
│   ├── retrieval_tool.py          search_services(profile) — the hybrid-retrieval tool Match & Rank calls
│   ├── llm.py                     model access (Bedrock Converse) + the schema-validation guardrail
│   ├── intake.py                  node: extracts CaregiverProfile from the conversation, or asks a clarifying question
│   ├── match_rank.py              node: retrieval + hard filters (budget/area) + LLM rerank + fallback ranking
│   ├── explain_draft.py           node: drafts the first-enquiry message for the top (or requested) match
│   ├── guardrails.py              iteration cap, tool allow-list, static scan for any send/book tool
│   ├── graph.py                   wires the nodes into the LangGraph state machine + node-transition trace
│   ├── geo.py                     locate(service) — Singapore town-centroid coordinates for the map
│   ├── api.py                     FastAPI app: serves web/, /images, and POST /api/chat + GET /api/featured
│   └── check_env.py               smoke test: confirms Bedrock model + embedding access actually work
│
├── web/                            the product front-end (served by src/api.py)
│   ├── index.html                 page shell: header, chat panel, results (cards + map)
│   ├── app.js                     fetches /api/featured + /api/chat, renders cards, drives the Leaflet map
│   └── styles.css                 all styling
│
├── data/                           knowledge base the agent matches against
│   ├── README.md                  record schema, provenance, dataset-coverage notes — read this before editing services.json
│   ├── services.json              76 curated SG respite services — the working dataset (committed; graded deliverable)
│   ├── services.sample.json       3-record schema example only
│   ├── contacts.md                researched enquiry email/web-form/phone per centre
│   ├── vector_index/               FAISS build artifact — gitignored, rebuild with `python src/ingest.py`
│   └── images/                     card photos (Pexels licence, no attribution needed) — see data/images/README.md
│
└── tests/                          pytest suite — one file per src/ module, plus test_api.py for the FastAPI layer
```

## Guardrails (responsible-AI design)

The agent is fenced in by **mechanism, not prompt wording** — all three checks
live in [`src/guardrails.py`](src/guardrails.py):

- **No send/book tool exists anywhere in this codebase.** `ALLOWED_TOOLS` is
  a closed set — `search_services` + `draft_message` only — and
  `scan_source_for_action_tools()` greps `src/` for any
  `def send_* / book_* / submit_* / call_* ...`; the test suite fails loudly
  if one is ever added. The agent drafts; the caregiver sends.
- **Every model output is schema-validated before use.** `llm.complete(...,
  response_schema=…)` forces JSON, parses it, validates against a Pydantic
  model, and raises `LLMSchemaError` on any failure — so every node inherits
  the guarantee and a bad output aborts the run instead of being shown.
- **The orchestration loop has a hard iteration cap** (`MAX_ITERATIONS` in
  `.env`). `iteration_cap_reached()` is checked in the conditional edge after
  every worker node; tripping it routes straight to Output.

When nothing clears every constraint, Match & Rank does not dead-end: it
returns the next-best services ranked location > budget > schedule (hard
filters ignored, deterministic, no draft) as `fallback_matches`, and Output
shows them with a "check what they can take on" caveat — drafting an enquiry
from that list only if the caregiver explicitly asks or names an option.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install` fails on `torch`/`faiss-cpu` | You're not on Python 3.12 — recreate the venv with `py -3.12` / `python3.12` |
| `/api/chat` returns 503 "service index isn't built yet" | Run `python src/ingest.py` from the repo root |
| `python src/check_env.py` fails | Check `.env`: `LLM_BACKEND=bedrock`, correct `AWS_REGION`, and that your AWS credentials/profile actually have model access granted for the configured Bedrock models |
| Bedrock `AccessDeniedException` | Request model access for Claude Sonnet 4.5 / Haiku 4.5 / Cohere Embed English v3 in the Bedrock console for the region in `.env` |
| Blank/broken page at `localhost:8000` | Confirm uvicorn's `--app-dir src` flag is present and you're running it from the repo root |
| `ModuleNotFoundError` running `pytest` | Run `pytest` from the repo root (not `tests/`) — `conftest.py` puts `src/` on the path only from there |