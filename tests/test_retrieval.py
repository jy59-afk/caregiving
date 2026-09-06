"""
test_retrieval.py — retrieval-layer checks.

Two things are verified here:
  1. Hard filters (budget / area) actually exclude candidates — a pure unit
     test with a fake store, so it runs without building the index.
  2. recall@3 on a small labeled query set — the graded RAG metric. Skipped
     automatically if the FAISS index hasn't been built yet
     (`python src/ingest.py`).

Run:  pytest tests/test_retrieval.py -v
"""

import pytest  # test framework + skip helpers

from ingest import DEFAULT_INDEX_PATH  # so we can check whether the index exists before running recall@3
import retrieval_tool  # module under test (src/ is on sys.path via conftest.py)


# ---------------------------------------------------------------------------
# 1. Hard-filter unit test — no index required
# ---------------------------------------------------------------------------

class _FakeDoc:
    """Minimal stand-in for a LangChain Document: just metadata + page_content."""

    def __init__(self, name, cost, area):
        self.metadata = {"name": name, "cost": cost, "area": area}  # fields _hard_filter reads
        self.page_content = f"{name} passage"                        # not asserted on here


def test_hard_filter_excludes_over_budget_and_wrong_area():
    """A candidate over budget, or outside the requested area, must be dropped."""
    candidates = [
        (_FakeDoc("In budget, right area", cost=40, area="Toa Payoh"), 0.10),   # keep
        (_FakeDoc("Over budget", cost=120, area="Toa Payoh"), 0.05),             # drop: cost > budget
        (_FakeDoc("Wrong area", cost=30, area="Jurong"), 0.02),                  # drop: area mismatch
    ]
    profile = {"budget": 50, "area": "Toa Payoh"}  # constraints the caregiver stated

    kept = retrieval_tool._hard_filter(candidates, profile)  # exercise the filter directly

    names = [doc.metadata["name"] for doc, _ in kept]  # what survived
    assert names == ["In budget, right area"]          # only the one that satisfies every hard constraint


def test_hard_filter_passes_everything_when_no_constraints():
    """With no budget/area stated, hard filters must not drop anything."""
    candidates = [
        (_FakeDoc("A", cost=999, area="Anywhere"), 0.1),
        (_FakeDoc("B", cost=1, area="Elsewhere"), 0.2),
    ]
    kept = retrieval_tool._hard_filter(candidates, {"budget": None, "area": None})  # empty constraints
    assert len(kept) == 2  # nothing filtered out


# ---------------------------------------------------------------------------
# 2. recall@3 — the graded RAG metric. Needs the built index.
# ---------------------------------------------------------------------------

# Each case: a caregiver-style free-text need + the service that should surface,
# labeled against the curated data/services.json dataset.
LABELED_QUERIES = [
    {
        # dementia + wandering + secure outdoor space -> Apex Harmony Lodge
        "query": "my father has advanced dementia and keeps trying to walk out the door, I need somewhere he can't wander off from",
        "expected": "Apex Harmony Lodge Dementia Day Care Centre (Pasir Ris)",
    },
    {
        # up all night / sundowning -> night respite
        "query": "mum is awake and disoriented every night and I haven't slept properly in weeks",
        "expected": "NTUC Health Night Respite / Staycay@Henderson",
    },
    {
        # refuses to leave home / bedbound -> in-home
        "query": "my husband is bedbound and refuses to go to any centre, I just need a few hours of cover at home",
        "expected": "Homage Home-Based Respite Care",
    },
    {
        # extended absence / overseas travel -> nursing home respite
        "query": "I have to fly overseas for three weeks and there's no one else to look after my dad",
        "expected": "NTUC Health Nursing Home Respite (Jurong)",
    },
    {
        # weekday cover fine, weekend gap, tight budget -> weekend respite
        "query": "my helper covers weekdays but I get no break on saturday and sunday and money is really tight",
        "expected": "St Luke's ElderCare Weekend Respite Care (Bishan)",
    },
    {
        # post-stroke rehab + break -> St Luke's day care + rehab
        "query": "my mother is recovering from a stroke and needs physiotherapy plus somewhere to go during the day",
        "expected": "St Luke's ElderCare Senior Care Centre (Teck Whye)",
    },
    {
        # general frailty, mobile, not dementia, working caregiver in central area
        "query": "dad is frail and lonely at home while I'm at work, he's still mobile and his memory is basically fine",
        "expected": "NTUC Health Senior Day Care (Toa Payoh)",
    },
    {
        # early dementia, newly diagnosed, west side
        "query": "my wife was just diagnosed with early dementia, we live in bukit batok and want a weekday programme for her",
        "expected": "New Horizon Centre (Bukit Batok) - Dementia Singapore",
    },
    {
        # dementia day care, north of Singapore
        "query": "looking for a dementia day centre near yishun so I don't have to travel across the island every morning",
        "expected": "AWWA Dementia Day Care Centre (Yishun)",
    },
    {
        # post-hospital step-down, north-east, week-plus stay
        "query": "dad is being discharged from hospital next week and needs a short nursing stay near hougang before coming home",
        "expected": "All Saints Home Respite Care (Hougang)",
    },
]

RECALL_AT_3_TARGET = 0.7  # minimum acceptable recall@3 on the labeled set; raise as the dataset/tuning improves


@pytest.mark.skipif(
    not (DEFAULT_INDEX_PATH / "index.faiss").exists(),
    reason="vector index not built yet — run `python src/ingest.py`",
)
def test_recall_at_3_meets_target():
    """The correct service should appear in the top 3 retrieved for most labeled queries."""
    hits = 0  # count of queries whose expected service was in the top 3
    for case in LABELED_QUERIES:
        profile = {"needs_description": case["query"], "budget": None, "area": None}  # semantic-only for this metric
        results = retrieval_tool.search_services(profile, k=8)                         # retrieve, then top-3 after filtering
        if any(r["name"] == case["expected"] for r in results):                       # expected service surfaced?
            hits += 1

    recall_at_3 = hits / len(LABELED_QUERIES)  # the graded number
    assert recall_at_3 >= RECALL_AT_3_TARGET, f"recall@3 ={recall_at_3:.2f} below target {RECALL_AT_3_TARGET}"
