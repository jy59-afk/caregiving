"""
explain_draft.py — the third worker node.

Job: for one shortlisted service, draft the enquiry message the caregiver would
send. Grounded in that match's `grounding_passage` — the draft states only
facts the caregiver gave plus questions to ask; it never invents fees,
availability, or a confirmed booking.

Which service: the top `candidate_matches` by default; the caregiver's
`selected_option` if they picked one from the list; and on the no-match path
(only `fallback_matches`), the closest / selected near-miss — with the draft
worded as a first question because it doesn't fit everything they asked for.

Hard rule (CLAUDE.md): there is no send/book tool in V1. This node returns
draft text and nothing else. The caregiver reviews, edits, and sends it.

Tested in isolation in `tests/test_explain_draft.py` (the LLM call is mocked).
"""

from __future__ import annotations

import re  # normalising a service name for the `selected_option` match

import llm  # model seam; call as llm.complete so tests can monkeypatch it
from config import settings  # DRAFT_MODEL_ROLE — lets a rate-limited Bedrock demo draft on the cheap model
from state import CaregiverProfile, DraftedMessage, RankedMatch, RespiteState


def _norm(s: str) -> str:
    """Lowercase + drop punctuation/extra space — for loose service-name matching."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def _pick_target(matches: list[RankedMatch], profile: CaregiverProfile | None) -> RankedMatch:
    """
    Which service to draft for: the caregiver's `selected_option` if it names or
    numbers one of `matches`, else the top of the list.
    """
    if profile is None or not profile.selected_option:
        return matches[0]
    sel = profile.selected_option.strip()

    # "2" / "option 3" / "the first one" -> a 1-based index into the list
    words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    m = re.search(r"\b([1-9])\b", sel) or None
    idx = int(m.group(1)) if m else next((n for w, n in words.items() if w in sel.lower()), None)
    if idx is not None and 1 <= idx <= len(matches):
        return matches[idx - 1]

    # name / partial-name match
    sel_n = _norm(sel)
    if sel_n:
        for cand in matches:
            cand_n = _norm(cand.name)
            if sel_n in cand_n or cand_n in sel_n or _overlap(sel_n, cand_n) >= 0.6:
                return cand
    return matches[0]  # couldn't resolve — draft for the top, better than nothing


def _overlap(a: str, b: str) -> float:
    """Fraction of `a`'s words that also appear in `b` (crude token similarity)."""
    aw, bw = set(a.split()), set(b.split())
    return len(aw & bw) / len(aw) if aw else 0.0

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

    Drafts for one service: the caregiver's `selected_option` if they named one,
    else the top of the list. The list is `candidate_matches` on the normal
    path; on the no-match path it's `fallback_matches` (the graph only routes
    here then when a draft was explicitly requested) and the draft is worded as
    a first question, since it doesn't fit everything the caregiver asked for.
    Returns `drafted_message=None` (no model call) when there is nothing to
    draft for.
    """
    bumped = state.get("iteration_count", 0) + 1                       # this node counts as one step either way

    candidates: list[RankedMatch] = state.get("candidate_matches") or []
    fallbacks: list[RankedMatch] = state.get("fallback_matches") or []
    matches = candidates or fallbacks                                 # whichever the matcher produced
    if not matches:                                                   # nothing to draft about
        return {"drafted_message": None, "iteration_count": bumped}

    is_near_miss = not candidates                                     # drafting off the fallback list
    profile = state.get("caregiver_profile")
    target = _pick_target(matches, profile)                           # honour `selected_option`, else the top
    profile_json = profile.model_dump_json(indent=2) if profile is not None else "{}"

    near_miss_note = (
        "\nNOTE: this service does NOT match everything the caregiver asked for "
        "(see 'Why it was matched' — it may be over budget, wrong hours, or out "
        "of area). Word the message as a first question to check whether they "
        "can help at all; acknowledge the gap plainly (e.g. 'I know this may be "
        "above my budget, but...'). Do not imply it is a good fit."
        if is_near_miss else ""
    )

    draft = llm.complete(
        [
            {"role": "system", "content": _DRAFT_SYSTEM + near_miss_note},
            {
                "role": "user",
                "content": (
                    f"Service name: {target.name}\n"
                    f"Service passage (the only source of fact about the service):\n{target.grounding_passage}\n\n"
                    f"Why it was matched: {target.why}\n\n"
                    f"Caregiver profile:\n{profile_json}\n\n"
                    "Write the enquiry message."
                ),
            },
        ],
        model_role=settings.draft_model_role,                         # "reasoning" normally; "extract" to dodge a Sonnet rate limit
        response_schema=DraftedMessage,                               # guardrail: validated JSON only
        max_tokens=1024,                                              # room for the body + hidden reasoning tokens
    )

    return {"drafted_message": draft, "iteration_count": bumped}
