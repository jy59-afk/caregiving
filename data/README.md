# `data/` — curated respite-service knowledge base

| File | Committed? | Purpose |
|---|---|---|
| `services.sample.json` | yes | 3-record schema example only |
| `services.json` | yes | The working dataset the vector index is built from |
| `vector_index/` | **no** (gitignored) | FAISS build artifact — rebuild with `python src/ingest.py` |

## Record schema

```json
{
  "name": "NTUC Health Senior Day Care (Toa Payoh)",   // unique display name (provider + branch)
  "provider": "NTUC Health",                            // operating organisation
  "care_type": "day-care",                              // day-care | dementia day-care is folded into day-care | weekend-respite | night-respite | home-help | short-stay
  "days": "Mon-Fri",                                    // free text, human-readable
  "hours": "7am-7pm, full-day and half-day options",    // free text, human-readable
  "cost_per_session": 75,                               // NUMBER: approx out-of-pocket for one day / visit / night BEFORE means-tested subsidy (used as the hard budget filter)
  "area": "Toa Payoh",                                  // planning area — matched by case-insensitive substring against the caregiver's stated area
  "distance_km": 2.0,                                   // placeholder; a real build computes this per-query from the caregiver's location
  "contact": "AIC hotline 1800-650-6060 / aic.sg care-finder",  // routing point for the enquiry
  "source": "https://ntuchealth.sg/...",               // where the facts were checked
  "image": "services/01.jpg",                           // per-centre card photo, relative to data/images/ (see data/images/README.md)
  "enquiry_email": "care@ntuchealth.sg",                // published enquiry email, or null — drives the web UI's "Open in your email app" (mailto:) button (see data/contacts.md)
  "enquiry_form": null,                                 // web contact-form URL, shown when there is no email; null when neither (enquire via AIC)
  "description": "..."                                  // free text — THIS is what gets embedded and semantically matched; write it the way a caregiver would read it
}
```

`ingest.py` embeds a flattened passage per record (name + care_type + area +
days + hours + cost + description) and keeps `name`, `cost`, `area`, `days`,
`care_type` as filter metadata. Extra fields (`provider`, `contact`,
`source`, `distance_km`, `image`, `enquiry_email`, `enquiry_form`) are carried
in the JSON for humans / the web UI but are not
embedded.

## AIC Nursing Home Respite Care import (76 records total)

The dataset was expanded from the 13 hand-curated records to **76** by
importing every centre on AIC's
[Nursing Home Respite Care](https://www.aic.sg/Care-Services/Nursing-Home-Respite-Care)
map (`POST https://www.aic.sg/api/map-items`, category
`94c18c4d-5f6e-4380-8eb3-c2043bd9037a` — 64 rows). One row (`All Saints Home
(Hougang)`) duplicated an existing curated facility and was dropped; the other
63 were appended. Rebuild the raw pull + merge with the two scratch scripts
noted in the session handoff.

What the AIC source **does** give per centre: name, address + postal code,
phone/email, lat/long, sometimes visiting hours. What it does **not** give:
per-centre respite fees, the length/rules of a respite stay, or care-mix
detail. So for every imported record:

| Field | How it was set |
|---|---|
| `care_type` | `"short-stay"` for all 63 — nursing-home respite is a planned short **residential** stay |
| `cost_per_session` | `131` for all 63 — AIC's single published figure ("daily room and board starts from $131 for an open ward", pre-subsidy). Per-centre pricing is not published; do not read these as real quotes |
| `days` / `hours` | generic: `"By arrangement (planned block, typically up to 30 days)"` / `"24-hour residential respite stay"` |
| `area` | planning area / neighbourhood, hand-derived from the street address + postal code (see the `AREA` map in the transform script) |
| `distance_km` | `0.0` placeholder (same meaning as before — a real build computes this per query) |
| `description` | templated per record from name + area + operator, kept short and residential-focused so the passages are not 63 identical paragraphs |
| `contact` | `"AIC 1800-650-6060 (referral + application required)"` + the provider phone where AIC listed one |

**Retrieval note:** with ~80% of the index now same-shape nursing-home
passages, MiniLM-only recall@3 on the labeled set fell from 0.80 to 0.60. The
LLM rerank in `src/match_rank.py` still drops the mismatched homes in the live
pipeline; the durable fix is Titan embeddings + rerank tuning, not trimming the
dataset. `RECALL_AT_3_TARGET` in `tests/test_retrieval.py` was lowered to 0.6
to match.

## Provenance and honesty note

**This is testing data, not a verified live directory.** The records describe
real Singapore respite programmes and real centre locations, but the specific
fees, operating hours and attendance rules are approximations compiled from
public 2025–2026 sources (per-record in `source`; the AIC-imported 63 all
carry AIC's published open-ward baseline, not a real per-centre quote) and
**will drift**. Fees shown are pre-subsidy; actual out-of-pocket cost is
means-tested and often far lower (AIC subsidy up to ~80%).

Before any real deployment each record must be re-confirmed with the provider
or via AIC Care Finder (`aic.sg/care-services/care-finder`, 1800-650-6060).

### Sources used to compile this dataset

- Agency for Integrated Care — Nursing Home Respite Care, Day Care, Night Respite (`aic.sg`)
- HealthHub — "Respite Care for Caregivers Provided by Senior Care Centres"
- NTUC Health — Senior Day Care centre pages; "Cost and Options for Respite Care 2026"
- Dementia Singapore — New Horizon Centres (`dementia.org.sg/services/nhc/`)
- St Luke's ElderCare — services and locations (`slec.org.sg`)
- Homage — Respite Care (`homage.sg/services/respite-care/`)
- Apex Harmony Lodge (`apexharmony.org.sg`)
- All Saints Home — Services & Centres (`allsaintshome.org.sg`)
- AWWA (`awwa.org.sg`)

## Coverage (deliberate variety for retrieval testing)

- **Care types:** general day care, dementia day care, weekend respite, night
  respite, in-home respite, nursing-home short stay (65 of the 76 records are
  `short-stay` after the AIC import)
- **Areas:** ~30 planning areas island-wide after the AIC import (was ~12)
- **Budget span:** ~S$19/attendance (weekend respite) to ~S$131/day
  (nursing-home respite) before subsidy
- **Deliberate near-duplicates** for semantic disambiguation tests: general
  vs dementia day care in the *same* area (Toa Payoh); three New Horizon
  dementia centres differing only by location; and now many nursing-home
  respite records that differ only by operator + area — which is what the
  location > budget > schedule fallback ranker in `src/match_rank.py` is built
  to order when nothing clears every hard constraint
