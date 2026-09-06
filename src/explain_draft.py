"""
explain_draft.py — the third worker node.

Job: for the single top-ranked service, draft the enquiry message the
caregiver would send. Grounded in that match's `grounding_passage` — the draft
states only facts the caregiver gave plus questions to ask; it never invents
fees, availability, or a confirmed booking.

Hard rule (CLAUDE.md): there is no send/book tool in V1. This node returns
draft text and nothing else. The caregiver reviews, edits, and sends it.

Tested in isolation in `tests/test_explain_draft.py` (the LLM call is mocked).
"""

from __future__ import annotations

import llm  # model seam; call as llm.complete so tests can monkeypatch it
from state import DraftedMessage, RankedMatch, RespiteState

# Drafting instruction. The shape of `body`/`subject` is pinned by the
# DraftedMessage field descriptions that llm.complete injects; this message
# sets voice, grounding, and the no-booking rule.
_DRAFT_SYSTEM = (
    "You draft a first-contact enquiry message that a Singapore family "
    "caregiver will send to a respite-care service. Write in the caregiver's "
    "first-person voice, warm and plain, not salesy.\n"
    "Rules:\n"
    "- Ground every statement about the service in the provided passage. Do not "
    "state fees, subsidy amounts, availability, or transport as facts — ask "
    "about them instead.\n"
    "- Include only details about the care recipient that appear in the "
    "caregiver's profile.\n"
    "- This is a draft for the caregiver to review and send themselves. Do not "
    "write as though a place is already booked or confirmed."
)


def run_explain_and_draft(state: RespiteState) -> dict:
    """
    LangGraph node: `state -> {"drafted_message": DraftedMessage | None,
    "iteration_count": <bumped>}`.

    Uses `state["candidate_matches"][0]` — the top match from Match & Rank. If
    there are no matches, returns `drafted_message=None` without calling the
    model (nothing to draft about).
    """
    matches: list[RankedMatch] = state.get("candidate_matches", [])   # ordered shortlist from the previous node
    bumped = state.get("iteration_count", 0) + 1                       # this node counts as one step either way

    if not matches:                                                   # Match & Rank found nothing that fit
        return {"drafted_message": None, "iteration_count": bumped}

    top = matches[0]                                                  # draft only for the single best fit
    profile = state.get("caregiver_profile")                          # the caregiver's stated need, for context
    profile_json = profile.model_dump_json(indent=2) if profile is not None else "{}"

    draft = llm.complete(
        [
            {"role": "system", "content": _DRAFT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Service name: {top.name}\n"
                    f"Service passage (the only source of fact about the service):\n{top.grounding_passage}\n\n"
                    f"Why it was matched: {top.why}\n\n"
                    f"Caregiver profile:\n{profile_json}\n\n"
                    "Write the enquiry message."
                ),
            },
        ],
        model_role="reasoning",                                       # drafting quality matters — use the stronger model
        response_schema=DraftedMessage,                               # guardrail: validated JSON only
        max_tokens=1024,                                              # room for the body + hidden reasoning tokens
    )

    return {"drafted_message": draft, "iteration_count": bumped}
