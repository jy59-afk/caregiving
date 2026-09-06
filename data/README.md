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
  "description": "..."                                  // free text — THIS is what gets embedded and semantically matched; write it the way a caregiver would read it
}
```

`ingest.py` embeds a flattened passage per record (name + care_type + area +
days + hours + cost + description) and keeps `name`, `cost`, `area`, `days`,
`care_type` as filter metadata. Extra fields (`provider`, `contact`,
`source`, `distance_km`) are carried in the JSON for humans but are not
embedded.

## Provenance and honesty note

**This is testing data, not a verified live directory.** The 13 records
describe real Singapore respite programmes and real centre locations, but the
specific fees, operating hours and attendance rules are approximations
compiled from public 2025–2026 sources (listed per-record in `source`) and
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
  respite, in-home respite, nursing-home short stay
- **Areas:** Toa Payoh, Bishan, Bukit Batok, Bedok, Tampines, Pasir Ris,
  Choa Chu Kang, Jurong, Hougang, Yishun, Bukit Merah, plus one islandwide
  home-visit service
- **Budget span:** ~S$19/attendance (weekend respite) to ~S$120/day
  (nursing-home respite) before subsidy
- **Deliberate near-duplicates** for semantic disambiguation tests: general
  vs dementia day care in the *same* area (Toa Payoh); three New Horizon
  dementia centres differing only by location
