# Demo video script — Respite Navigator (V1)

Target length **≤ 3 minutes**. Two windows on screen: the **Streamlit chat**
(left) and the **terminal running it** (right, showing the node trace). Have
the FAISS index already built and `LLM_BACKEND` configured.

Pre-roll checklist:

```bash
python src/ingest.py          # once, if data/vector_index/ is missing
streamlit run src/app.py      # leave the launch terminal visible
```

---

## 0:00 – 0:25 · The problem (talking head or title card)

> "In Singapore, **82.83% of family caregivers who know respite care exists
> have never used it** — not because there aren't enough services, but because
> matching one to your exact hours, budget and care needs takes time and
> certainty a stretched caregiver doesn't have. Respite Navigator is the
> matching layer that closes that gap. It finds the one option that fits, and
> drafts the first message — it never books anything for you."

## 0:25 – 1:15 · The happy path (screen: chat + terminal)

Type as Belinda:

> "My mother has Alzheimer's and gets agitated in the afternoons. I lecture
> part-time and need weekday daytime cover near Toa Payoh. I can't really go
> above $85 a session."

While it runs, **point at the terminal**:

> "Every step is logged. Intake pulls a typed profile — need, budget, area.
> Match & Rank does semantic retrieval, then a **hard filter on the $85 budget
> and the Toa Payoh area**, then a rerank. Explain & Draft writes the enquiry.
> Consent gate, then Output."

Back to the chat, walk through the result:

> "Three options ranked, each with a *why this fits* that's grounded in the
> service's real description — expand 'Where this comes from' to see the source
> passage. And a drafted enquiry — subject and body — in an editable box.
> **There is no send button.** Belinda copies this into her own email."

## 1:15 – 1:55 · Budget is a real filter (screen: chat)

New conversation ("Start over"). Type:

> "Same situation but my ceiling is $30 a session."

> "Now it returns **no matches** — and says so honestly, suggesting she widen
> the budget or area or accept a different care type. It does **not** quietly
> show a $60 option and hope she doesn't notice. Budget is a first-class
> filter, never an afterthought."

## 1:55 – 2:30 · Guardrails (screen: `src/guardrails.py` + a test run)

> "The 'responsible AI' story is enforced by mechanism, not prompt wording."

Show the three, fast:

- `ALLOWED_TOOLS` — a closed set of two tools. No send, no book, no call.
- `scan_source_for_action_tools()` — greps the codebase for a forbidden tool
  definition; run `pytest -q tests/test_guardrails.py` and show it green.
- `llm.complete(..., response_schema=…)` — every model output is
  Pydantic-validated before use; a bad one raises and aborts the run.
- `MAX_ITERATIONS` — hard cap, checked after every node.

## 2:30 – 3:00 · Evaluation (screen: `docs/EVALUATION.md`)

```bash
python src/evaluate.py
```

> "Eight scripted caregiver conversations, run end-to-end against the real
> model and index. **Schema validation 100%. Tool calls 100%. Task completion
> 8 out of 8** — including the ones that *should* return no-match or ask a
> clarifying question. Zero answer-fidelity issues: every shown reason traces
> to a real passage. Retrieval recall@3 is 0.80. The harness writes this table
> straight to the repo — it's our evaluation slide."

Close:

> "Respite Navigator — finds the fit, drafts the ask, and stops there."

---

## If a live run misbehaves

The model is non-deterministic. If Match & Rank returns fewer than 3 options,
that's correct behaviour when only 1–2 services genuinely fit — say so rather
than re-rolling on camera. If a run errors, the UI shows a retry message (not a
stack trace); just resubmit. Pre-record the happy path as a fallback clip.
