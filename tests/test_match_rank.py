"""
test_match_rank.py — unit tests for the Match & Rank node, in isolation.

No network, no vector index. `match_rank.search_services` and `llm.complete`
are both monkeypatched. Under test:
  * retrieval output is passed through to the rerank prompt correctly
  * the empty-retrieval shortcut (no LLM call)
  * reconciliation: model sets order + "why"; retrieval owns the facts;
    invented names and duplicates are dropped
  * the profile reaches search_services in the dict shape it expects (so the
    budget/area hard filters actually run against it)

Run:  pytest tests/test_match_rank.py -v
"""

import pytest

import llm
import match_rank
from match_rank import run_match_and_rank
from state import CaregiverProfile, RankedMatch, RankedMatches, new_state


def _retrieved(*names_scores) -> list[dict]:
    """Build a fake search_services result: [(name, score), ...] -> list of dicts."""
    return [
        {
            "name": name,
            "grounding_passage": f"{name} is a real service. Authoritative passage.",
            "similarity_score": score,
        }
        for name, score in names_scores
    ]


def _state_with_profile(**profile_kw):
    """A state whose Intake step has already run."""
    profile_kw.setdefault("needs_description", "weekday daytime dementia care")
    state = new_state([{"role": "user", "content": "..."}])
    state["caregiver_profile"] = CaregiverProfile(**profile_kw)
    return state


# ---------------------------------------------------------------------------
# Empty retrieval — no model call
# ---------------------------------------------------------------------------

def test_empty_retrieval_returns_no_matches_without_calling_llm(monkeypatch):
    monkeypatch.setattr(match_rank, "search_services", lambda profile, k=8: [])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")))

    state = _state_with_profile()
    state["iteration_count"] = 4
    out = run_match_and_rank(state)

    assert out["candidate_matches"] == []
    assert out["iteration_count"] == 5


# ---------------------------------------------------------------------------
# Missing profile — run out of order
# ---------------------------------------------------------------------------

def test_raises_if_profile_not_set():
    with pytest.raises(ValueError):
        run_match_and_rank(new_state([{"role": "user", "content": "hi"}]))


# ---------------------------------------------------------------------------
# Profile shape handed to retrieval
# ---------------------------------------------------------------------------

def test_profile_passed_to_search_services_as_dict_with_budget_and_area(monkeypatch):
    seen = {}

    def capture(profile, k=8):
        seen["profile"] = profile
        return []

    monkeypatch.setattr(match_rank, "search_services", capture)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: RankedMatches(matches=[]))

    run_match_and_rank(_state_with_profile(budget=55, area="Yishun", needs_description="dementia day care near yishun"))

    p = seen["profile"]
    assert isinstance(p, dict)
    assert p["budget"] == 55
    assert p["area"] == "Yishun"
    assert p["needs_description"] == "dementia day care near yishun"


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def test_model_sets_order_retrieval_owns_facts(monkeypatch):
    """Model reorders B before A and writes 'why'; passages/scores come from retrieval."""
    retrieved = _retrieved(("Service A", 0.10), ("Service B", 0.20), ("Service C", 0.30))
    monkeypatch.setattr(match_rank, "search_services", lambda profile, k=8: retrieved)

    model_out = RankedMatches(matches=[
        RankedMatch(name="Service B", why="Best fit for weekends.", grounding_passage="MODEL-INVENTED TEXT", similarity_score=0.99),
        RankedMatch(name="Service A", why="Also plausible.", grounding_passage="MODEL-INVENTED TEXT", similarity_score=0.99),
    ])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: model_out)

    out = run_match_and_rank(_state_with_profile())
    matches = out["candidate_matches"]

    assert [m.name for m in matches] == ["Service B", "Service A"]          # model's order kept
    assert matches[0].why == "Best fit for weekends."                        # model's rationale kept
    assert matches[0].grounding_passage == "Service B is a real service. Authoritative passage."  # from retrieval
    assert matches[0].similarity_score == 0.20                               # from retrieval, not the model's 0.99


def test_invented_and_duplicate_names_are_dropped(monkeypatch):
    retrieved = _retrieved(("Real One", 0.1), ("Real Two", 0.2))
    monkeypatch.setattr(match_rank, "search_services", lambda profile, k=8: retrieved)

    model_out = RankedMatches(matches=[
        RankedMatch(name="Real One", why="ok", grounding_passage="x"),
        RankedMatch(name="Totally Made Up Centre", why="hallucinated", grounding_passage="x"),
        RankedMatch(name="Real One", why="listed twice", grounding_passage="x"),
        RankedMatch(name="Real Two", why="ok", grounding_passage="x"),
    ])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: model_out)

    out = run_match_and_rank(_state_with_profile())
    assert [m.name for m in out["candidate_matches"]] == ["Real One", "Real Two"]


def test_reconcile_caps_at_three(monkeypatch):
    retrieved = _retrieved(("S1", 0.1), ("S2", 0.2), ("S3", 0.3), ("S4", 0.4))
    monkeypatch.setattr(match_rank, "search_services", lambda profile, k=8: retrieved)
    model_out = RankedMatches(matches=[
        RankedMatch(name=n, why="ok", grounding_passage="x") for n in ("S1", "S2", "S3", "S4")
    ])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: model_out)

    out = run_match_and_rank(_state_with_profile())
    assert len(out["candidate_matches"]) == 3


# ---------------------------------------------------------------------------
# Rerank prompt contents
# ---------------------------------------------------------------------------

def test_rerank_prompt_carries_candidate_passages_and_uses_reasoning_model(monkeypatch):
    retrieved = _retrieved(("Service A", 0.1))
    monkeypatch.setattr(match_rank, "search_services", lambda profile, k=8: retrieved)

    seen = {}

    def spy(messages, **kwargs):
        seen["messages"] = messages
        seen["kwargs"] = kwargs
        return RankedMatches(matches=[RankedMatch(name="Service A", why="fit", grounding_passage="x")])

    monkeypatch.setattr(llm, "complete", spy)
    run_match_and_rank(_state_with_profile())

    user_turn = seen["messages"][-1]["content"]
    assert "Service A is a real service." in user_turn        # the grounding passage is in the prompt
    assert seen["kwargs"]["model_role"] == "reasoning"        # stronger model for ranking nuance
    assert seen["kwargs"]["response_schema"] is RankedMatches
