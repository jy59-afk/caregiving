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

# Each case: a caregiver-style free-text need + the service that should surface.
# Expand to 8-10 cases covering the real dataset's variety before relying on the number.
LABELED_QUERIES = [
    {
        "query": "dad gets confused during the day and needs someone patient supervising him",
        "expected": "Sunshine Corner Day Centre",
    },
    {
        "query": "mum won't leave the house and needs help moving around at home",
        "expected": "Golden Years Home Respite",
    },
    {
        "query": "I need to travel for a week — somewhere safe for my father to stay overnight",
        "expected": "Harmony Short-Stay Centre",
    },
]

RECALL_AT_3_TARGET = 0.66  # minimum acceptable recall@3 on the labeled set; raise as the dataset/tuning improves


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
