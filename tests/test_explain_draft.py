"""
test_explain_draft.py — unit tests for the Explain & Draft node, in isolation.

No network. `llm.complete` is monkeypatched. Under test:
  * the no-matches shortcut (returns drafted_message=None, no model call)
  * the happy path returns the validated DraftedMessage and bumps iteration
  * the top match's grounding passage is what reaches the prompt
  * there is no send/book behaviour — the node only ever returns draft text

Run:  pytest tests/test_explain_draft.py -v
"""

import pytest

import llm
from explain_draft import run_explain_and_draft
from state import CaregiverProfile, DraftedMessage, RankedMatch, new_state


def _state_with_matches(*matches: RankedMatch):
    state = new_state([{"role": "user", "content": "..."}])
    state["caregiver_profile"] = CaregiverProfile(needs_description="weekday dementia day care in Bishan", area="Bishan")
    state["candidate_matches"] = list(matches)
    return state


def _match(name="Bishan Day Centre") -> RankedMatch:
    return RankedMatch(
        name=name,
        why="Runs a weekday dementia programme in Bishan.",
        grounding_passage=f"{name} runs a small-group weekday dementia day programme with transport within 5km.",
        similarity_score=0.12,
    )


# ---------------------------------------------------------------------------
# No matches — nothing to draft
# ---------------------------------------------------------------------------

def test_no_matches_returns_none_without_calling_llm(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")))

    state = _state_with_matches()   # empty
    state["iteration_count"] = 7
    out = run_explain_and_draft(state)

    assert out["drafted_message"] is None
    assert out["iteration_count"] == 8


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_returns_validated_draft_and_bumps_iteration(monkeypatch):
    canned = DraftedMessage(
        service_name="Bishan Day Centre",
        subject="Enquiry about weekday dementia day care",
        body="Hello, I care for my mother who has dementia. I am looking for weekday daytime cover in Bishan. "
             "Could you tell me about availability, fees after subsidy, and whether transport is provided?",
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: canned)

    out = run_explain_and_draft(_state_with_matches(_match()))

    assert out["drafted_message"] is canned
    assert out["iteration_count"] == 1


def test_prompt_is_grounded_in_top_match_passage_only(monkeypatch):
    seen = {}

    def spy(messages, **kwargs):
        seen["messages"] = messages
        seen["kwargs"] = kwargs
        return DraftedMessage(service_name="A", subject="s", body="b")

    monkeypatch.setattr(llm, "complete", spy)

    top = _match("Top Match Centre")
    other = _match("Second Centre")
    run_explain_and_draft(_state_with_matches(top, other))

    user_turn = seen["messages"][-1]["content"]
    assert "Top Match Centre runs a small-group weekday dementia day programme" in user_turn  # top match passage present
    assert "Second Centre" not in user_turn                                                    # only the top match is drafted for
    assert seen["kwargs"]["model_role"] == "reasoning"
    assert seen["kwargs"]["response_schema"] is DraftedMessage


# ---------------------------------------------------------------------------
# Guardrail: node surface is draft-only
# ---------------------------------------------------------------------------

def test_node_returns_only_draft_keys(monkeypatch):
    """The node's contract is draft text + iteration count — nothing that could send."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: DraftedMessage(service_name="A", subject="s", body="b"))
    out = run_explain_and_draft(_state_with_matches(_match()))
    assert set(out.keys()) == {"drafted_message", "iteration_count"}
