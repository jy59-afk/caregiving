"""
test_intake.py — unit tests for the Intake node, in isolation.

No network. `llm.complete` is monkeypatched to return a canned
`CaregiverProfile` (or to raise), so what's under test is the node's own
logic: transcript rendering, the empty-conversation shortcut, iteration
counting, and that a guardrail failure propagates rather than being swallowed.

Run:  pytest tests/test_intake.py -v
"""

import pytest

import llm  # so we can monkeypatch llm.complete (intake calls it as llm.complete)
from intake import run_intake
from state import CaregiverProfile, new_state


def _profile(**kw) -> CaregiverProfile:
    """A valid CaregiverProfile with sensible defaults, overridable per test."""
    kw.setdefault("needs_description", "mum has moderate dementia and needs daytime supervision")
    return CaregiverProfile(**kw)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_run_intake_returns_validated_profile_and_bumps_iteration(monkeypatch):
    """The node returns whatever validated profile llm.complete produced, and counts one step."""
    canned = _profile(budget=40, area="Bishan")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: canned)

    state = new_state([{"role": "user", "content": "I need day care for my mum in Bishan, budget about $40."}])
    state["iteration_count"] = 2

    out = run_intake(state)

    assert out["caregiver_profile"] is canned          # passed straight through
    assert out["iteration_count"] == 3                  # 2 -> 3


def test_run_intake_passes_relabelled_transcript_without_system_turns(monkeypatch):
    """System turns are dropped; user->Caregiver, assistant->Assistant in the prompt."""
    seen = {}

    def spy(messages, **kwargs):
        seen["messages"] = messages
        seen["kwargs"] = kwargs
        return _profile()

    monkeypatch.setattr(llm, "complete", spy)

    state = new_state([
        {"role": "system", "content": "orchestration scaffolding — must not appear"},
        {"role": "user", "content": "my father keeps wandering"},
        {"role": "assistant", "content": "How many hours of cover do you need?"},
        {"role": "user", "content": "weekdays while I work"},
    ])
    run_intake(state)

    user_prompt = seen["messages"][-1]["content"]           # the transcript turn
    assert "scaffolding" not in user_prompt                 # system turn excluded
    assert "Caregiver: my father keeps wandering" in user_prompt
    assert "Assistant: How many hours of cover do you need?" in user_prompt
    assert seen["kwargs"]["model_role"] == "extract"        # cheap model for parsing
    assert seen["kwargs"]["response_schema"] is CaregiverProfile  # the guardrail is wired


# ---------------------------------------------------------------------------
# Empty-conversation shortcut
# ---------------------------------------------------------------------------

def test_run_intake_with_no_messages_skips_llm_and_asks_a_question(monkeypatch):
    """Nothing to extract from: return an empty profile with a clarifying question, no model call."""
    called = False

    def boom(*a, **k):
        nonlocal called
        called = True
        raise AssertionError("llm.complete should not be called on an empty transcript")

    monkeypatch.setattr(llm, "complete", boom)

    out = run_intake(new_state([]))

    assert called is False
    profile = out["caregiver_profile"]
    assert profile.needs_description == ""
    assert profile.clarifying_question                       # a non-empty question was set
    assert out["iteration_count"] == 1                       # still counts as a step


def test_run_intake_treats_whitespace_only_transcript_as_empty(monkeypatch):
    """A transcript that renders to only whitespace takes the same shortcut."""
    def boom(*a, **k):
        raise AssertionError("llm.complete should not be called")

    monkeypatch.setattr(llm, "complete", boom)
    out = run_intake(new_state([{"role": "user", "content": "   \n  "}]))
    assert out["caregiver_profile"].clarifying_question


# ---------------------------------------------------------------------------
# Guardrail failure must propagate
# ---------------------------------------------------------------------------

def test_run_intake_propagates_schema_error(monkeypatch):
    """If llm.complete raises LLMSchemaError (bad extraction), the node does not swallow it."""
    def raise_schema(*a, **k):
        raise llm.LLMSchemaError("CaregiverProfile validation failed")

    monkeypatch.setattr(llm, "complete", raise_schema)

    with pytest.raises(llm.LLMSchemaError):
        run_intake(new_state([{"role": "user", "content": "help"}]))
