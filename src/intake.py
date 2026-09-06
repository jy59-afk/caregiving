"""
intake.py — the first worker node of the pipeline.

Job: read the conversation so far and distil it into a `CaregiverProfile`
(the free-text need + budget + area + schedule the rest of the graph needs).
It is the first real consumer of the `llm.complete(..., response_schema=...)`
guardrail: the extracted profile is Pydantic-validated inside `llm.py` before
this node ever sees it, so a malformed extraction raises rather than flowing
downstream.

Tested in isolation in `tests/test_intake.py` (the LLM call is mocked).
"""

from __future__ import annotations

import llm  # the provider-agnostic model seam; call as llm.complete so tests can monkeypatch it
from state import CaregiverProfile, RespiteState, render_transcript  # state contract + schema + transcript helper

# The extraction instruction. Kept terse: the field-by-field guidance lives in
# `CaregiverProfile`'s Field(description=...) values, which `llm.complete`
# injects as the JSON-schema contract. This message only sets the task and the
# non-negotiable rules.
_INTAKE_SYSTEM = (
    "You are the intake step of a service that matches Singapore family "
    "caregivers to respite-care options. Read the transcript and extract a "
    "structured profile of the caregiver's need.\n"
    "Rules:\n"
    "- Use only what the caregiver actually said or clearly implied. Never "
    "guess a budget, an area, or a diagnosis that isn't there — leave the "
    "field null instead.\n"
    "- `needs_description` should read like the caregiver describing the care "
    "recipient, in their words, focused on the care need rather than logistics.\n"
    "- Only set `clarifying_question` when a recommendation would be unreliable "
    "without the missing fact; otherwise leave it null."
)


def run_intake(state: RespiteState) -> dict:
    """
    LangGraph node: `state -> partial state update`.

    Reads `state["messages"]`, returns `{"caregiver_profile": CaregiverProfile,
    "iteration_count": <bumped>}`. Raises `llm.LLMSchemaError` (via
    `llm.complete`) if the model's extraction does not validate — we want that
    loud, not swallowed.
    """
    transcript = render_transcript(state.get("messages", []))  # drop system turns, relabel speakers

    # Guard: nothing to extract from. Return an empty-but-valid profile so the
    # graph can route to a "ask the caregiver something first" branch instead
    # of calling the model with a blank prompt.
    if not transcript.strip():
        empty = CaregiverProfile(
            needs_description="",                                   # nothing said yet
            clarifying_question="What kind of care does the person you look after need, and when do you need a break?",
        )
        return {
            "caregiver_profile": empty,
            "iteration_count": state.get("iteration_count", 0) + 1,  # this node still counts as one step
        }

    # The real extraction. `response_schema=CaregiverProfile` makes llm.complete
    # force JSON, parse it, and validate it — so `profile` below is always a
    # well-formed CaregiverProfile or an exception was already raised.
    profile = llm.complete(
        [
            {"role": "system", "content": _INTAKE_SYSTEM},          # task + rules
            {"role": "user", "content": f"Transcript:\n{transcript}"},  # the conversation to distil
        ],
        model_role="extract",                                       # cheap/fast model — this is parsing, not reasoning
        response_schema=CaregiverProfile,                           # the guardrail: validated before we get it
        max_tokens=512,                                             # generous: gpt-oss/qwen spend hidden tokens first
    )

    return {
        "caregiver_profile": profile,                               # validated CaregiverProfile
        "iteration_count": state.get("iteration_count", 0) + 1,     # one more step taken
    }
