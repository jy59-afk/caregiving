# Demo video script — Respite Navigator (V1)

Target length **~4:30** (hard floor 4:00, ceiling 5:00 — check the official
submission rules for a cap before recording). Two windows on screen: the
**Streamlit chat** (left) and the **terminal running it** (right, showing the
node trace). Keep the terminal visible the whole time — the rubric explicitly
rewards *visibly* showing the agent plan, act and adapt.

Pre-roll checklist:

```bash
python src/ingest.py          # once, if data/vector_index/ is missing
python src/check_env.py        # confirm the model backend answers
streamlit run src/app.py      # leave the launch terminal visible
```

Have a second terminal ready for the `pytest` and `python src/evaluate.py`
segments so you're not waiting on Streamlit.

Timings are cumulative. If a live model call runs long, talk over it — the
trace on the right gives you something to narrate.

---

## 0:00 – 0:30 · The problem

Talking head or title card.

> "In Singapore, **82.83% of family caregivers who already know respite care
> exists have never used it** — that's the SMU ROSA brief from March 2025. Not
> because there aren't enough services. Because matching one to your exact
> hours, your budget and your relative's care needs takes more time and more
> certainty than a caregiver stretched this thin has to spare. 63% of them
> have taken an income hit to keep caregiving.
>
> Respite Navigator is the **matching and activation layer** that closes that
> gap. It finds the one option that fits, explains why, and drafts the first
> message. It never books anything — that stays with the caregiver."

## 0:30 – 1:00 · The architecture, in one breath

Show the flow diagram (from the README) or `src/graph.py`'s docstring.

> "It's a LangGraph state machine, not one big prompt loop. Four stages:
> **Intake** turns free text into a typed profile. **Match & Rank** does
> hybrid retrieval — semantic search to cast a wide net, then hard filters,
> then an LLM rerank. **Explain & Draft** writes the enquiry. Then a
> **human-in-the-loop consent gate**, and Output. Each node is short and
> single-purpose, and every transition is logged — you'll see that on the
> right as we go."

## 1:00 – 2:00 · The happy path

Type as Belinda in the chat:

> "My mother has Alzheimer's and gets agitated in the afternoons. I lecture
> part-time and need weekday daytime cover near Toa Payoh. I can't really go
> above $85 a session."

While it runs, **read the trace out loud** as it appears:

> "`-> intake` — it's pulling a structured profile: need is dementia day care,
> budget 85, area Toa Payoh. `route: intake -> match_rank` — it had enough to
> proceed, so it didn't stop to ask anything.
>
> `-> match_rank` — semantic retrieval returns a wide candidate set of eight,
> then it **hard-filters out anything over $85 or outside the area** — that's
> code, not the model — then the model reranks the survivors and writes the
> one-line 'why'. `-> explain_draft` — drafting the enquiry for the top match
> only. `-> consent_gate`, `-> output`, `outcome: matches_ready`."

Back to the chat, walk the result slowly:

> "Ranked options. Each 'why this fits' is **grounded** — open 'Where this
> comes from' and you see the actual service description the reason was drawn
> from. Nothing invented.
>
> And the drafted enquiry — subject and body — in an editable box. It asks
> about availability, fees after subsidy, and transport. **There's no send
> button.** Belinda edits this and sends it from her own email. That's the
> deliberate line: the agent drafts, the human sends."

## 2:00 – 2:35 · It adapts — the clarifying-question branch

"Start over". Type something deliberately thin (this is the eval's
`vague_needs_clarification` prompt):

> "I'm exhausted and I don't know how much longer I can keep doing this. I
> need a break."

> "`-> intake`, then `route: intake -> output  (clarifying question)`. It
> didn't guess. It recognised there's no care need, no budget, no area to work
> from, and asked one warm, targeted question —" (read the bubble) "— about
> who she's caring for and what support they need. Answer it in the same box
> and the next turn re-runs with the fuller picture. That's the
> plan-act-**adapt** loop the rubric asks for."

## 2:35 – 3:05 · Budget is a first-class filter

"Start over". Type:

> "Same situation as before — mother with Alzheimer's, weekday daytime near
> Toa Payoh — but my absolute ceiling is $30 a session."

> "`outcome: no_matches`. It returns **nothing**, and says so honestly —
> suggesting she widen the budget or the area, or consider a different care
> type. It does **not** quietly surface a $60 option and hope she doesn't
> notice the price. For a caregiver who's already lost income, budget is never
> an afterthought."

## 3:05 – 3:55 · Guardrails — enforced by mechanism

Screen: `src/guardrails.py`, then a terminal.

> "The 'responsible AI' story here isn't 'we told the model to behave'. It's
> four mechanisms."

Point at each:

- **`ALLOWED_TOOLS`** — a frozen set of exactly two: `search_services` and
  `draft_message`. There is no send, book, or call tool anywhere in the
  codebase.
- **`scan_source_for_action_tools()`** — a static scan that greps `src/` for
  any `def send_* / book_* / submit_* / call_*`. Run it:

  ```bash
  pytest -q tests/test_guardrails.py
  ```

  > "Green. If anyone ever adds a tool that acts on the caregiver's behalf,
  > this test goes red."

- **Schema validation** — `llm.complete(..., response_schema=…)` forces JSON,
  parses it, validates against a Pydantic model, and **raises** on any
  mismatch. Every node inherits it; a bad model output aborts the run instead
  of reaching the caregiver.
- **Iteration cap** — `MAX_ITERATIONS` in `.env`, checked in the conditional
  edge after *every* node. Show the eval's `iteration_cap_stops_run` scenario:
  it ends on `stopped_iteration_cap` with a "nothing was sent" message.

## 3:55 – 4:30 · Evaluation

Second terminal:

```bash
python src/evaluate.py
```

Then open `docs/EVALUATION.md`.

> "Eight scripted caregiver conversations — clean matches, budget-forces-no-
> match, area-forces-no-match, vague-needs-clarification, the follow-up that
> resolves it, an in-home request, night respite, and the iteration cap. All
> run end-to-end against the real model and the real index.
>
> **Schema-validation pass rate: 100%.** Tool-call success: 100%.
> **Task completion: 8 out of 8** — including the scenarios that are supposed
> to refuse or ask back. Zero answer-fidelity issues: every displayed reason
> traces to a real passage, no draft implies a booking. Retrieval recall@3 is
> 0.80.
>
> The harness writes this table straight into the repo. It *is* our
> evaluation slide — regenerated every run, not hand-typed."

Close:

> "Respite Navigator — it finds the fit, explains it in the service's own
> words, drafts the ask, and stops there. The caregiver stays in control of
> the one action that matters."

---

## Optional add-ons if you want to reach 5:00

- **Follow-up turn after the clarifying question** (adds ~20s): answer *"my
  husband had a stroke last year, mostly mobile now but needs physiotherapy
  and supervision during the day; we're in Choa Chu Kang and can manage up to
  about $70 a session"* and show the run proceed to `matches_ready` on the
  second turn (St Luke's ElderCare Teck Whye — day care + rehab). Visible
  proof the graph is stateless between turns and just re-extracts with more
  context.
- **Backend switch** (adds ~15s): show `LLM_BACKEND=groq` vs `bedrock` in
  `.env` and the one-line switch in `src/llm.py` — "Groq for dev iteration,
  Bedrock Converse for the demo, same code path".
- **The state object** (adds ~20s): show `RespiteState` in `src/state.py` —
  "heavy objects live in state, not in the prompt: messages, profile,
  candidate matches, draft, consent flag, iteration count".

## If a live run misbehaves

The model is non-deterministic. If Match & Rank returns fewer than 3 options,
that's *correct* when only 1–2 services genuinely fit — say so rather than
re-rolling on camera. If a run errors, the UI shows a retry message, not a
stack trace; just resubmit. Pre-record a clean happy-path take as a fallback.
