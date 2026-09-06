# Respite Navigator — Project Brief

This is the persistent context file referenced by Strategy §0. It is the
condensed brief; the full build guide is
[Respite_Navigator_Hackathon_Strategy.md](Respite_Navigator_Hackathon_Strategy.md),
and the source material is in [reference/](reference/).

## One-line pitch

A friendly chat that helps a caregiver in Singapore find a few trustworthy
hours of respite care — matched to their situation, budget and schedule —
without filling in another form.

## The problem is discovery, not supply

Singapore already has respite services (day-care centres, home-help,
short-stay). Only **50.09%** of informal caregivers know respite care exists
for them; of those who do, **82.83%** have never used it. Even Australia's
fully-funded, nationally advertised Carer Gateway reached only **6%** uptake.
So this is built as a **matching and activation layer**, not a listing site.

### Supporting evidence (for the slide)

- **210,000+** caregivers in Singapore today (SMU 2020 survey, cited by MOH)
- **50.09%** aware respite care exists (SMU ROSA Research Brief, Mar 2025)
- **82.83%** of those aware have never used it (SMU ROSA, Mar 2025)
- **S$56,877/yr** average income loss for caregivers who changed work
  arrangements because of caregiving (SMU ROSA, Mar 2025)
- **6%** uptake of Australia's Carer Gateway by June 2023 (Australian Govt
  Dept. of Health, Disability and Ageing, 2025) — proof this is a
  matching gap, not a supply gap

## POV problem statement

> A family caregiver in Singapore who has already heard that respite care
> exists needs a fast, judgment-free way to find and act on the one option
> that actually fits their week and their wallet — because 82.83% of
> caregivers who are aware respite services exist have still never used one
> (SMU ROSA Research Brief, Mar 2025), not for lack of options, but because
> matching the right service to their specific hours, budget and care needs
> takes more time and certainty than a caregiver already stretched thin has
> to spare.

### Persona

Belinda Seet, 63, a part-time lecturer and primary caregiver for her
89-year-old mother who has Alzheimer's, says she cannot afford respite care
priced at S$20–S$40 an hour — the cost alone rules it out before she can even
assess whether a service would fit her mother's needs.
*(Reported in Malay Mail/CNA, "The Big Read," 2021.)*

## Why agentic AI, not just a directory (separate slide)

A static directory already exists and hasn't closed the gap. The bottleneck
is translating one caregiver's specific, messy situation (unpaid time
available, budget, a parent's needs) into a short, ranked, *explained*
shortlist, and lowering the activation energy to reach out. That is a
planning-and-matching task with fuzzy natural-language input: the agent
extracts structured constraints from an unstructured conversation, reasons
about fit rather than keyword overlap, and adapts its next question to what
the caregiver has already said.

## How it works (V1)

1. **Natural-language intake** — caregiver describes their situation in their
   own words (who they care for, what would help, when they're free, what
   they can spend). No form, no jargon, no login.
2. **Matching against a curated knowledge base** — ~15 hand-checked local
   services, filtered/ranked by budget, schedule fit, proximity, care type.
3. **Plain-language recommendations** — 2–3 options, each with a short
   "why this fits" tied to what the caregiver said.
4. **Human-in-the-loop consent** — the agent drafts the first enquiry
   message; nothing is booked, called, or sent without explicit approval.

## Service record schema

```json
{
  "name": "Sunshine Corner Day Centre",
  "care_type": "day-care",
  "days": "Mon-Fri",
  "hours": "9am-1pm",
  "cost_per_session": 45,
  "area": "Toa Payoh",
  "distance_km": 2.1,
  "description": "Write this the way a caregiver would read it, not as a form field. The rich free-text description carries the real semantic signal for retrieval."
}
```

## Roadmap (mention, don't build)

- **V2 — Living Directory:** partnered data feeds keep the list current;
  adds a `book_service()` tool behind the same consent gate.
- **V3 — Companion:** proactive check-ins ("it's been six weeks, want me to
  look again?") and hand-off to a human support line when the issue is beyond
  logistics.

## Coding standard

Comment every function and non-trivial line so the logic stays auditable for
hackathon judges and for debugging. Folder structure: `src/`, `data/`,
`docs/`, `tests/`. Secrets in `.env`; dependencies in `requirements.txt`.
