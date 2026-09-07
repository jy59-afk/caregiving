"""
test_graph.py — the LangGraph pipeline end to end, with the model and the
retrieval tool mocked. No network, no FAISS index.

Covers:
  * happy path: Intake -> Match & Rank -> Explain & Draft -> Output, outcome
    "matches_ready", a draft in the final response
  * clarifying-question branch: routes Intake -> Output, rerank never called
  * empty-shortlist branch: routes Match & Rank -> Output, draft never called
  * iteration cap: a low MAX_ITERATIONS forces an early Output with
    outcome "stopped_iteration_cap"
  * schema-validation failure in a node aborts the whole run (not swallowed)
  * the compiled graph registers no send/book node

Run:  pytest tests/test_graph.py -v
"""

import pytest

import guardrails
import graph as graph_mod
import llm
import match_rank
from state import (
    CaregiverProfile,
    DraftedMessage,
    RankedMatch,
    RankedMatches,
)

# A transcript with enough detail that Intake would not need to ask anything.
_FULL_TRANSCRIPT = [
    {
        "role": "user",
        "content": (
            "My mother has Alzheimer's and gets agitated in the afternoons. "
            "I need weekday daytime cover in Bishan, and I can't go above $40 a session."
        ),
    }
]


def _fake_retrieval(*names):
    """A stand-in for search_services: one dict per service name."""
    return [
        {
            "name": n,
            "grounding_passage": f"{n} runs a weekday small-group dementia day programme in Bishan with transport.",
            "similarity_score": 0.1 + i * 0.05,
        }
        for i, n in enumerate(names)
    ]


def _complete_router(*, profile=None, matches=None, draft=None):
    """
    Build a fake `llm.complete` that answers by which response_schema it was
    handed — mirroring how the three nodes call it.
    """
    prof = profile or CaregiverProfile(
        needs_description="mother has Alzheimer's, afternoon agitation, needs weekday daytime supervision",
        budget=40,
        area="Bishan",
        schedule="weekday daytime",
        relationship="mother",
    )
    ranked = matches if matches is not None else RankedMatches(
        matches=[RankedMatch(name="Bishan Wellness Day Centre", why="Weekday dementia day programme in Bishan.", grounding_passage="x")]
    )
    drafted = draft or DraftedMessage(
        service_name="Bishan Wellness Day Centre",
        subject="Enquiry about weekday dementia day care",
        body="Hello, I care for my mother who has Alzheimer's. I'm looking for weekday daytime cover in Bishan. "
             "Could you share availability, fees after subsidy, and whether transport is included?",
    )

    def fake_complete(messages, *, model_role, response_schema=None, **kwargs):
        if response_schema is CaregiverProfile:
            return prof
        if response_schema is RankedMatches:
            return ranked
        if response_schema is DraftedMessage:
            return drafted
        raise AssertionError(f"unexpected response_schema {response_schema!r}")

    return fake_complete


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_runs_all_nodes_and_produces_a_draft(monkeypatch):
    monkeypatch.setattr(match_rank, "search_services", lambda profile, k=8: _fake_retrieval("Bishan Wellness Day Centre"))
    monkeypatch.setattr(llm, "complete", _complete_router())

    final = graph_mod.run(_FULL_TRANSCRIPT)

    assert final["outcome"] == "matches_ready"
    assert final["caregiver_profile"].area == "Bishan"
    assert [m.name for m in final["candidate_matches"]] == ["Bishan Wellness Day Centre"]
    assert final["drafted_message"].service_name == "Bishan Wellness Day Centre"
    # the grounding passage on the reconciled match came from retrieval, not the model's "x"
    assert "weekday small-group dementia day programme" in final["candidate_matches"][0].grounding_passage
    assert "Bishan Wellness Day Centre" in final["final_response"]
    assert "Subject:" in final["final_response"]
    # every node bumped the counter: intake, match_rank, explain_draft
    assert final["iteration_count"] == 3


# ---------------------------------------------------------------------------
# Clarifying-question branch
# ---------------------------------------------------------------------------

def test_clarifying_question_short_circuits_to_output(monkeypatch):
    """Intake sets clarifying_question -> straight to Output; rerank never runs."""
    thin_profile = CaregiverProfile(
        needs_description="",
        clarifying_question="What kind of care does your mother need, and when do you need a break?",
    )

    def boom_search(*a, **k):
        raise AssertionError("search_services must not be called when Intake needs clarification")

    monkeypatch.setattr(match_rank, "search_services", boom_search)
    monkeypatch.setattr(llm, "complete", _complete_router(profile=thin_profile))

    final = graph_mod.run([{"role": "user", "content": "I need help with my mum"}])

    assert final["outcome"] == "needs_clarification"
    assert final["final_response"] == thin_profile.clarifying_question
    assert final["candidate_matches"] == []
    assert final["drafted_message"] is None


# ---------------------------------------------------------------------------
# Empty-shortlist branch -> the next-best fallback list (no draft)
# ---------------------------------------------------------------------------

_FAKE_SERVICES = [
    {"name": "In-Budget Bishan Day Care", "care_type": "day-care", "days": "Mon-Fri", "hours": "8am-6pm",
     "cost_per_session": 35, "area": "Bishan", "distance_km": 1.0, "description": "weekday day care in Bishan"},
    {"name": "Over-Budget Jurong Home", "care_type": "short-stay", "days": "Daily", "hours": "24-hour",
     "cost_per_session": 150, "area": "Jurong", "distance_km": 8.0, "description": "residential nursing stay"},
    {"name": "Over-Budget Bishan Night Respite", "care_type": "night-respite", "days": "Nightly", "hours": "overnight",
     "cost_per_session": 110, "area": "Bishan", "distance_km": 2.0, "description": "overnight respite in Bishan"},
]


def test_no_match_falls_back_to_next_best_before_draft(monkeypatch):
    monkeypatch.setattr(match_rank, "search_services", lambda profile, k=8: [])
    monkeypatch.setattr(match_rank, "load_services", lambda: _FAKE_SERVICES)

    calls = {"draft": 0}

    def fake_complete(messages, *, model_role, response_schema=None, **kwargs):
        if response_schema is CaregiverProfile:
            return CaregiverProfile(needs_description="weekday dementia day care in Bishan", budget=40,
                                    area="Bishan", schedule="weekday daytime")
        if response_schema is DraftedMessage:
            calls["draft"] += 1
            return DraftedMessage(service_name="x", subject="x", body="x")
        raise AssertionError(f"unexpected schema {response_schema!r}")

    monkeypatch.setattr(llm, "complete", fake_complete)

    final = graph_mod.run(_FULL_TRANSCRIPT)

    assert final["outcome"] == "fallback_matches"
    assert final["candidate_matches"] == []
    assert final["drafted_message"] is None
    assert calls["draft"] == 0
    # location > budget > schedule: the two Bishan options come before the
    # Jurong one; within Bishan, the in-budget day care leads.
    assert [m.name for m in final["fallback_matches"]] == [
        "In-Budget Bishan Day Care",
        "Over-Budget Bishan Night Respite",
        "Over-Budget Jurong Home",
    ]
    assert "closest options" in final["final_response"].lower()
    assert "In-Budget Bishan Day Care" in final["final_response"]


def test_no_match_still_drafts_when_the_caregiver_asks(monkeypatch):
    """Follow-up 'draft me the email' on the fallback path -> Explain & Draft runs
    for the closest option; outcome stays fallback_matches but a draft is attached."""
    monkeypatch.setattr(match_rank, "search_services", lambda profile, k=8: [])
    monkeypatch.setattr(match_rank, "load_services", lambda: _FAKE_SERVICES)

    def fake_complete(messages, *, model_role, response_schema=None, **kwargs):
        if response_schema is CaregiverProfile:
            return CaregiverProfile(needs_description="weekday dementia day care in Bishan", budget=40,
                                    area="Bishan", schedule="weekday daytime", wants_draft=True)
        if response_schema is DraftedMessage:
            return DraftedMessage(service_name="In-Budget Bishan Day Care",
                                  subject="Enquiry", body="Hello, I know this may not fit my hours, but...")
        raise AssertionError(f"unexpected schema {response_schema!r}")

    monkeypatch.setattr(llm, "complete", fake_complete)

    final = graph_mod.run(_FULL_TRANSCRIPT + [{"role": "user", "content": "draft me the email"}])

    assert final["outcome"] == "fallback_matches"          # still a near-miss list
    assert final["candidate_matches"] == []
    assert final["drafted_message"] is not None            # ...but a draft was produced anyway
    assert final["drafted_message"].service_name == "In-Budget Bishan Day Care"
    assert "I've drafted an enquiry to In-Budget Bishan Day Care" in final["final_response"]


# ---------------------------------------------------------------------------
# Iteration cap
# ---------------------------------------------------------------------------

def test_iteration_cap_forces_early_output(monkeypatch):
    """With MAX_ITERATIONS=1, Intake alone trips the cap and the graph stops at Output."""
    monkeypatch.setattr(guardrails.settings, "max_iterations", 1)

    def boom_search(*a, **k):
        raise AssertionError("cap should have routed to Output before Match & Rank")

    monkeypatch.setattr(match_rank, "search_services", boom_search)
    monkeypatch.setattr(llm, "complete", _complete_router())

    final = graph_mod.run(_FULL_TRANSCRIPT)

    assert final["outcome"] == "stopped_iteration_cap"
    assert final["iteration_count"] == 1
    assert "nothing was sent" in final["final_response"].lower()


# ---------------------------------------------------------------------------
# Schema-validation failure aborts the run
# ---------------------------------------------------------------------------

def test_schema_error_in_a_node_aborts_the_run(monkeypatch):
    """A node raising LLMSchemaError propagates out of graph.run — no unvalidated output."""
    monkeypatch.setattr(match_rank, "search_services", lambda profile, k=8: _fake_retrieval("Some Centre"))

    def fake_complete(messages, *, model_role, response_schema=None, **kwargs):
        if response_schema is CaregiverProfile:
            return CaregiverProfile(needs_description="weekday dementia day care", budget=40, area="Bishan")
        if response_schema is RankedMatches:
            raise llm.LLMSchemaError("RankedMatches validation failed for model output")
        raise AssertionError("draft should never be reached")

    monkeypatch.setattr(llm, "complete", fake_complete)

    with pytest.raises(llm.LLMSchemaError):
        graph_mod.run(_FULL_TRANSCRIPT)


# ---------------------------------------------------------------------------
# Topology guardrail
# ---------------------------------------------------------------------------

def test_compiled_graph_registers_only_the_expected_nodes():
    """No send/book/call node is wired into the graph — the node set is closed."""
    g = graph_mod.build_graph()
    node_ids = set(g.get_graph().nodes) - {"__start__", "__end__"}
    assert node_ids == {"intake", "match_rank", "explain_draft", "consent_gate", "output"}
    for forbidden in ("send", "book", "submit", "dispatch", "call"):
        assert not any(forbidden in nid for nid in node_ids)
