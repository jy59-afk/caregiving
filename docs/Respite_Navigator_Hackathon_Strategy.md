# Respite Navigator — Hackathon Strategy & Build Guide
*IGNITE Agentic AI Hackathon 2026 · Compiled from `Respite Navigator.pdf`, `The Caregiving Gap.pdf`, and the three IGNITE training decks (Sessions 1–3)*

---

## 1. What Respite Navigator Is — Product Summary & Key Features

**One-line pitch:** A friendly chat that helps a caregiver in Singapore find a few trustworthy hours of respite care — matched to their situation, budget and schedule — without filling in another form.

**The problem it targets isn't supply, it's discovery.** Singapore already has respite care services (day-care centres, home-help, short-stay options). The evidence brief is explicit that this is *not* a directory problem: only 50.09% of informal caregivers even know respite care exists for them, and of those who do, 82.83% have never used it. Even Australia's fully-funded, nationally advertised Carer Gateway only reached 6% uptake. So the product is built as a **matching and activation layer**, not a bigger listing site.

### How it works (V1 — the hackathon build)
1. **Natural-language intake** — the caregiver describes their situation in their own words (who they care for, what would help, when they're free, what they can spend). No form, no clinical jargon, no login wall.
2. **Matching against a curated knowledge base** — roughly 15 hand-checked local respite services, filtered/ranked by budget, schedule fit, proximity, and care type.
3. **Plain-language recommendations** — 2–3 options, each with a short "why this fits" reason tied back to what the caregiver said (not just a similarity score).
4. **Human-in-the-loop consent** — the agent drafts the first enquiry message; nothing is booked, called, or sent without the caregiver explicitly approving it.

### Key features to demonstrate for judges
| Feature | Why it matters for scoring |
|---|---|
| Conversational, form-free intake | Directly answers the "63% income loss" / time-poverty evidence — the product must work in the five minutes a caregiver actually has |
| Budget-first filtering | Belinda Seet's story (see §2) shows cost alone ends a search before it starts — budget has to be a first-class filter, not an afterthought |
| Explainable matches ("why this fits") | Distinguishes the agent from a search engine; maps to the "Effectiveness of the Solution" and "Technical Quality" judging criteria |
| Draft-then-consent workflow | Shows deliberate, responsible agent design (a guardrail, not a missing feature) — good talking point for "Technical Quality and Superiority" |
| Small, curated dataset (~15 services) | Keeps V1 scope honest and demoable in the hackathon window — avoids "The Boiling Ocean" failure mode |

### Roadmap context (mention, don't build)
- **V2 — Living Directory:** partnered data feeds so the service list stays current; moves from "draft a message" to "book the slot" with confirmation.
- **V3 — Companion:** proactive check-ins ("it's been six weeks, want me to look again?") and hand-off to a human support line when the issue is beyond logistics.

Keep V2/V3 on the "Roadmap & Future Potential" slide only — building toward them in the architecture (§3) is good for scalability points, but the hackathon deliverable is V1.

---

## 2. Problem Statement Deliverable

The Session 3 training is explicit about how problem statements are graded, and it gives a POV template, six failure modes, and a five-question pressure test. Below is a ready-to-use statement built to survive all of them, plus the supporting evidence trail so you can cite sources on the slide.

### The POV statement (for the slide)

> **A family caregiver in Singapore who has already heard that respite care exists needs a fast, judgment-free way to find and act on the one option that actually fits their week and their wallet — because 82.83% of caregivers who are aware respite services exist have still never used one (SMU ROSA Research Brief, Mar 2025), not for lack of options, but because matching the right service to their specific hours, budget and care needs takes more time and certainty than a caregiver already stretched thin has to spare.**

### The persona, to make it concrete
> Belinda Seet, 63, a part-time lecturer and primary caregiver for her 89-year-old mother who has Alzheimer's, says she cannot afford respite care priced at S$20–S$40 an hour — the cost alone rules it out before she can even assess whether a service would fit her mother's needs. *(Reported in Malay Mail/CNA, "The Big Read," 2021.)*

### Supporting evidence to cite on the slide
- **210,000+** caregivers in Singapore today — roughly the population of Bishan and Toa Payoh combined *(SMU 2020 survey, cited by MOH)*
- **50.09%** of informal caregivers are even aware respite care services exist *(SMU ROSA Research Brief, Mar 2025)*
- **82.83%** of those who *are* aware have still never used it *(SMU ROSA Research Brief, Mar 2025)*
- **S$56,877/yr** average income loss for caregivers who changed work arrangements because of caregiving *(SMU ROSA Research Brief, Mar 2025)*
- **6%** uptake of Australia's fully-funded national Carer Gateway by June 2023 — the international proof that this is an awareness/matching gap, not a supply gap *(Australian Government Dept. of Health, Disability and Ageing, 2025)*

### Self-check against the six failure modes
| Failure mode | Does this statement avoid it? |
|---|---|
| The Solution in Disguise | Yes — no mention of chat, AI, or matching *technology* in the statement itself |
| The Everyone Problem | Yes — named persona (Belinda Seet), specific moment (cost ruling out the search before it starts) |
| The Missing Because | Yes — cited, dated, sourced statistic |
| The Boiling Ocean | Yes — scoped to *one caregiver finding one service*, not "fixing eldercare in Singapore" |
| The Solved Problem | Yes — Carer Gateway evidence shows even a mature, funded solution hasn't closed this gap |
| The Comfortable Guess | Yes — grounded in SMU ROSA research and five caregivers' publicly reported experiences, not team assumption |

### Pressure test
1. **Can we name one person?** Yes — Belinda Seet, 63, part-time lecturer, caring for her mother.
2. **Can we cite the evidence?** Yes — SMU ROSA Research Brief, March 2025 (82.83%, 50.09%), plus the CNA/TODAY "Big Read" (2021).
3. **Would that person recognise themselves?** Yes — her own quoted words describe exactly this: cost ruling out the search.
4. **Does it survive a different solution?** Yes — a hotline redesign, a subsidy program, or a printed directory could all *also* be answers to this same statement; it doesn't presuppose an agent.
5. **Read aloud test:** "A caregiver needs a fast, judgment-free way to find and act on respite care that fits their week and wallet, because most who know it exists have never used it." — a listener would ask *how*, not *what are you building* — it passes.

### Solution justification (keep on a separate slide — "why agentic AI, and not just a directory")
Use this only after the problem statement slide, since the training explicitly grades these as two separate answers:

> A static directory already exists and hasn't closed the gap — the bottleneck isn't information, it's translating one caregiver's specific, messy situation (unpaid time available, budget, a parent's needs) into a short, ranked, *explained* shortlist, and then lowering the activation energy to actually reach out. That's a planning-and-matching task with fuzzy natural-language input, which is exactly where an agent (versus a fixed-form search filter) earns its place: it extracts structured constraints from an unstructured conversation, reasons about fit rather than keyword overlap, and adapts its next question to what the caregiver has already said.
>
> **Test:** would this problem still exist if agentic AI had never been invented? *Yes* — that confirms the problem statement above is clean. The agentic approach is justified separately, in the solution.

---

## 3. High-Level Architecture & Wireframe Workflow

This maps Respite Navigator's V1 build onto the exact "Agentic AI Stack" taught in the IGNITE sessions (Interface → Orchestrate → Access → Model, with Guardrails and State as cross-cutting concerns), so the technical architecture slide can lift vocabulary the judges already recognise from the same training.

### End-to-end workflow (left to right)

```
┌──────────────┐      ┌───────────────────────────────────────────────────────────────┐      ┌──────────────┐
│   CAREGIVER  │      │                     ORCHESTRATOR (LangGraph                    │      │   CAREGIVER  │
│  (chat: text │─────▶│               or Claude Agent SDK state graph)                 │─────▶│  (reviews &  │
│  or voice)   │      │                                                                 │      │   sends)     │
└──────────────┘      │  ┌───────────────┐   ┌────────────────┐   ┌──────────────────┐ │      └──────────────┘
                       │  │  1. INTAKE &  │──▶│ 2. MATCH & RANK│──▶│ 3. EXPLAIN + DRAFT│ │
                       │  │  EXTRACTION   │   │  (tool call)   │   │   (tool call)     │ │
                       │  │  node         │   │                │   │                   │ │
                       │  └───────────────┘   └────────────────┘   └──────────────────┘ │
                       │        │                     │                      │           │
                       │        ▼                     ▼                      ▼           │
                       │   updates STATE:        search_services()     draft_message()   │
                       │   {care_recipient,       hybrid: vector        no send/book     │
                       │    needs, budget,        similarity + hard     tool exists in   │
                       │    schedule, area}       filter (FAISS/Chroma) V1 — HITL gate   │
                       └───────────────────────────────────────────────────────────────────┘
                                          │                                    │
                                          ▼                                    ▼
                                  MODEL ACCESS LAYER                   GUARDRAILS
                             Bedrock Converse API /                — iteration cap in state
                             ChatBedrockConverse                   — allow-listed tools only
                             • Haiku 4.5 → intake extraction,      — no auto-send/book tool exists
                               classification (cheap, fast)          in V1 at all
                             • Sonnet 4.5 → ranking rationale,     — every tool result validated
                               message drafting (needs nuance)       before being shown to user
```

### Layer-by-layer, mapped to the taught stack

| Stack layer | What Respite Navigator uses (V1) | Notes |
|---|---|---|
| **Interface** | Minimal chat UI (web page or Streamlit) — text input now, voice can be a stretch goal | No framework required; keep it thin, it's not the graded component |
| **Orchestrate** | LangGraph state graph *or* Claude Agent SDK (pick one — see decision note below) with nodes: Intake → Match & Rank → Explain & Draft → (HITL) → Output | Short, single-purpose nodes per the "context rot is real" guidance — don't build one giant loop |
| **Tools** | `search_services(profile) -> list[Match]` (filters/scores the curated dataset), `draft_message(service, profile) -> str` | Tool docstrings are the prompt — write them as carefully as any UI copy |
| **Guardrails & limits** | Hard iteration cap held in state; `allowed_tools` is an explicit allow-list (search + draft only, no send/book); every model output validated against a Pydantic schema before use | This *is* your "responsible AI" talking point for the deck |
| **Model access** | Bedrock Converse API (or `ChatBedrockConverse` if using LangChain/LangGraph) | Use Groq for cheap/free local dev iteration, switch the model id to Bedrock for the demo/deploy build |
| **Model** | Claude Haiku 4.5 for extraction/classification; Claude Sonnet 4.5 where matching nuance or drafting quality matters | Matches the "$1/$5 default, upgrade only where Haiku is measurably wrong" guidance from the training |
| **Data / knowledge base** | Hand-checked service records (name, hours, cost, care type, area, contact, **plus a free-text description**), embedded into a vector index (FAISS/Chroma) with structured fields kept as metadata | Upgraded to a **hybrid** design — see "Adding RAG" below. Structured fields stay as hard filters; the free-text description is what gets embedded and semantically matched |
| **Deploy & serve** | Local FastAPI/`@app.entrypoint` handler for the demo; Bedrock AgentCore Runtime deployment as a stretch goal | AgentCore deployment is optional but scores well on "Technical Quality — minimal work needed for production" |
| **Cross-cutting: State** | Typed state object: `messages`, `caregiver_profile`, `candidate_matches`, `drafted_message`, `consent_given`, `iteration_count` | Keep heavy objects (full service list, full match scores) in state, not in the prompt |
| **Cross-cutting: Observability** | Log node transitions and tool-call outcomes for the demo video and for the evaluation slide (see below) | Lets you visibly narrate "plan → act → adapt" in the video, which the judging rubric explicitly rewards |

### Decision note: LangGraph vs. Claude Agent SDK
- **LangGraph** if your team wants explicit, inspectable control over the flowchart (intake → match → draft is a known, fixed sequence — LangGraph's "strict step-by-step" mental model fits well and is easy to diagram for the deck).
- **Claude Agent SDK** if you're developing primarily *inside* Claude Code already (see §4) and want the agent's own file/bash tool access for building and testing itself — better for teams who want the orchestration and the dev tooling to be the same technology story.
- Either is defensible; pick the one your team is faster in, and say so plainly in the "Technical Architecture" slide rather than hedging.

### Adding RAG / Vector Retrieval (upgrade from flat JSON filtering)

Once service descriptions carry real content worth matching semantically — programme style, atmosphere, how a centre handles confusion or mobility limits — a flat JSON filter can't capture that, and vector retrieval earns its place. **Keep it hybrid, not pure-vector:** budget and schedule are hard, checkable constraints that embeddings judge poorly ("is $45 ≤ $50" is not a semantic question), so those stay as metadata filters; the free-text fit — the part a caregiver actually agonises over — is what gets embedded and searched.

| Piece | Choice | Why |
|---|---|---|
| Embedding model | Amazon Titan Text Embeddings V2 via Bedrock | Same AWS account/keys as the LLM calls — no new vendor to configure |
| Local/free dev fallback | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs on-device) | Iterate before Bedrock access is granted, same pattern as using Groq for dev and Bedrock for the demo build |
| Vector store | FAISS or Chroma | Zero infrastructure, in-memory or on-disk — right-sized for tens to low hundreds of records; skip Pinecone/OpenSearch Serverless, that's infra you don't need |
| Chunking | One passage per service (structured fields + free-text description flattened into one paragraph) | Only split into multiple chunks per service if a write-up genuinely runs long |

**Retrieval flow:** embed the caregiver's free-text need → similarity search returns a wider candidate set (e.g. top 8) → hard-filter out anything over budget or outside the area → keep the top 3 remaining by similarity score. The retrieved passage itself gets passed into the Explain & Draft node as grounding context, so the "why this fits" reasoning cites real service content instead of inventing a justification — this also gives you a concrete, defensible RAG story for the technical architecture slide rather than an unmotivated buzzword.

Starter code (`ingest.py` to build the index, `retrieval_tool.py` as the drop-in `search_services()` replacement, sample data schema, and a README) is provided alongside this document. The function signature stays the same as the JSON-filter version, so the LangGraph/Claude Agent SDK graph itself doesn't need to change — only the tool's internals.

**Evaluate retrieval, don't just claim it.** Build a small labeled set (8–10 caregiver-style queries with a known best-fit service) and report **recall@3** — the share of queries where the correct service appears in the top 3 retrieved — alongside the schema validation pass rate and tool-call success rate already planned for the evaluation slide.

### Where V2/V3 attach (for the "Roadmap" slide's diagram, not for the build)
- V2 adds a `book_service()` tool behind the same consent gate, plus a live data-sync job replacing the static JSON.
- V3 adds a scheduled "check-in" trigger and an escalation branch that routes to a human-support tool when the conversation signals something beyond logistics (e.g., safety, mental health).

---

## 4. Piloting Claude Code to Build This, Feature by Feature

The training material stresses two things that translate directly into how to drive Claude Code on this project: **build short, single-purpose slices** (context rot is real — don't try to get the whole agent working in one prompt), and **keep documentation/tests light but present** (a good README plus in-line comments is what judges will actually read). Here's a concrete sequence.

### 0. Set up a persistent context file first
Before writing any feature, give Claude Code a short project brief it can re-read every session — this project's custom instruction already asks for a commented line after each function/line of code, so bake that in explicitly:

> "This is Respite Navigator, an agentic chatbot that matches Singapore caregivers to respite care services. [paste the POV problem statement from §2]. Architecture: [paste the layer table from §3]. Coding standard: comment every function and non-trivial line so the logic stays auditable for hackathon judges and for our own debugging. Folder structure: `src/`, `data/`, `docs/`, `tests/`. Use `.env` for secrets, `requirements.txt` (or `uv`) for dependencies."

Save this as `docs/PROJECT_BRIEF.md` or a `CLAUDE.md` at the repo root — Claude Code (and any teammate) reads it at the start of a session instead of you re-explaining context every time.

### 1. Scaffold the repo (one session)
Ask Claude Code to create the folder structure (`src/`, `data/`, `docs/`, `tests/`), a `requirements.txt` or `uv` project, a `.env.example`, and a README skeleton with the sections judges are told to look for (instructions to run, overview of each file, environment setup). Verify it runs (`uv sync` or `pip install -r requirements.txt`) before moving on.

### 2. Data layer (one session)
Give Claude Code the curated services (or ask it to help you structure them from your own research) as a schema: `{name, care_type, days, hours, cost_per_session, area, distance_km, contact, description}` — the `description` field is what makes retrieval worthwhile, so write it the way a caregiver would read it, not as a form field.

### 2b. Vector index & retrieval tool (one session)
Ask Claude Code to write an ingestion script that flattens each record into one embeddable passage and builds a FAISS/Chroma index (start with a free local embedding model for dev, Bedrock Titan for the demo build), plus a `search_services(profile) -> list[Match]` tool that does similarity search first and hard-filters on budget/area after. Write a handful of labeled query→expected-service pairs and a `recall@3` test — don't move to the next slice until retrieval actually finds the right service on your labeled examples, not just until the code runs without errors.

### 3. Model access layer (one session)
Ask Claude Code to write a thin wrapper around the model call — start with Groq (free, fast to iterate) using the same interface you'll later point at Bedrock's Converse API, so switching later is a one-line change. Have it write a smoke-test script (`00_check_env.py`-style, matching the pattern from the training labs) that confirms the key works before building anything on top of it.

### 4. Orchestration — one node at a time (multiple short sessions)
This is the step most worth resisting the urge to do in one giant prompt. Build and test each node in isolation before wiring the graph:
- **Intake & Extraction node** — turns free text into the structured `caregiver_profile`. Prompt Claude Code with a handful of example caregiver messages (including messy ones) and ask for a Pydantic schema plus extraction logic and tests.
- **Match & Rank node** — calls `search_services()`, returns top 2–3.
- **Explain & Draft node** — generates the "why this fits" reasoning and the draft outreach message, as two distinct, separately testable functions (this also gives you a clean "answer fidelity" metric to report on the evaluation slide).
- Only after each node passes its own tests, ask Claude Code to wire them into the LangGraph graph (or Claude Agent SDK sub-agents) and add the conditional edges.

### 5. Guardrails (one session)
Ask explicitly for: a hard iteration cap stored in state, an `allowed_tools` allow-list that does *not* include any send/book action in V1, and schema validation on every model output before it's shown to the user. Ask Claude Code to write a test that asserts no send/book tool is reachable from the graph at all — this is a strong, checkable claim for the "Technical Quality" criterion.

### 6. Interface (one session)
Build the thinnest possible chat UI that calls the orchestrator (Streamlit is usually fastest for a hackathon). Don't let this slice grow — it's not what's graded.

### 7. Evaluation harness (one session — don't skip this, it's explicitly graded)
Write a small script that runs 5–8 scripted caregiver scenarios (including edge cases like "no budget stated," "two care recipients," "no matching service exists") and reports the metrics the training suggests: schema validation pass rate, tool-call success rate, task completion rate, and a manual answer-fidelity score. Add **recall@3** for the retrieval layer specifically (does the correct service appear in the top 3 retrieved, against your labeled query set from step 2b). This becomes the "Testing/Evaluation" content the slide deck needs, and it's evidence, not a claim.

### 8. Polish for submission (one session)
README finalization (setup, run instructions, file-by-file overview), a final read-through for in-line comments, and light instrumentation (print/log statements at each node transition) so the recorded demo visibly shows the agent planning, acting, and adapting — which is literally what the presentation criterion asks you to demonstrate.

### Prompting habits that keep this on track
- One feature = one focused Claude Code prompt with explicit acceptance criteria ("`search_services` returns top 3 by fit score, each with a one-line reason; add pytest coverage for budget-only and schedule-only filters").
- Ask for a short plan before code on anything touching the orchestration graph, and review it before letting Claude Code implement — cheaper to redirect a plan than a finished node.
- Commit after each slice passes its tests, so a bad iteration is easy to roll back.
- When something breaks, paste the actual failing test output rather than re-describing the bug from memory — Claude Code debugs from evidence faster than from a paraphrase.

---
*Sources: `Respite Navigator.pdf`, `The Caregiving Gap.pdf` (SMU ROSA Research Brief Mar 2025; MOH; SMU 2020 survey; Malay Mail/CNA "The Big Read" 2021; Australian Dept. of Health, Disability and Ageing 2025), and the IGNITE Agentic AI Hackathon Session 1–3 training decks (17–24 Aug 2026).*
