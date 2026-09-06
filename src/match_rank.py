"""
match_rank.py — the second worker node.

Job: turn a `CaregiverProfile` into an ordered shortlist of at most three
services, each with a grounded one-line "why this fits".

Two stages:
  1. Retrieval (already built) — `retrieval_tool.search_services` does hybrid
     search: semantic similarity, then hard filters on budget and area. It
     returns <=3 candidates. Budget/area are enforced here, not by the model.
  2. Rerank (LLM, `model_role="reasoning"`) — reorders the candidates and may
     drop ones that don't genuinely fit, and writes the "why" line. Its output
     is validated against `RankedMatches`.

Guardrail detail: after the rerank, every match is reconciled against the
retrieval output. A name the model invented is dropped; `grounding_passage`
and `similarity_score` are always taken from retrieval, never from the model —
so "why this fits" can be checked against real service text.

Tested in isolation in `tests/test_match_rank.py` (retrieval + LLM both mocked).
"""

from __future__ import annotations

import json  # to hand the candidate list to the rerank model as compact JSON

import llm  # model seam; call as llm.complete so tests can monkeypatch it
from retrieval_tool import search_services  # hybrid retrieval — semantic search then budget/area hard filter
from state import CaregiverProfile, RankedMatch, RankedMatches, RespiteState

# Rerank instruction. The model only reorders/drops and writes "why"; it must
# not add services or edit the facts. Field-level rules ride along in the
# RankedMatches JSON-schema contract that llm.complete injects.
_RERANK_SYSTEM = (
    "You are the ranking step of a Singapore respite-care matcher. You are "
    "given the caregiver's profile and a short list of candidate services that "
    "already passed the budget and area filters. Your job:\n"
    "- Reorder the candidates best-fit-first for THIS caregiver's stated need.\n"
    "- Drop any candidate that does not genuinely fit (wrong care type, wrong "
    "schedule, mismatched acuity). Returning fewer than three is correct when "
    "only one or two fit. Never add a service that is not in the list.\n"
    "- For each kept service write one sentence, addressed to the caregiver, on "
    "why it fits — using only what the candidate's passage supports. No claims "
    "about fees, subsidy amounts, or availability that aren't in the passage.\n"
    "- Copy each service `name` exactly as given."
)


def _format_candidates(candidates: list[dict]) -> str:
    """
    Render the retrieval output as compact JSON for the prompt: name + the
    grounding passage the model must reason from. The similarity score is left
    out — it is retrieval bookkeeping, not something the model should weigh.
    """
    slim = [
        {"name": c["name"], "passage": c["grounding_passage"]}  # only what the rerank needs to judge fit
        for c in candidates
    ]
    return json.dumps(slim, indent=2, ensure_ascii=False)       # readable in the prompt, still valid JSON


def _reconcile(model_matches: list[RankedMatch], retrieved: list[dict]) -> list[RankedMatch]:
    """
    Trust the model for ORDER and for the "why" line; trust retrieval for
    everything factual. Drop any match whose name isn't in the retrieved set
    (the model invented or mangled it), and overwrite grounding_passage /
    similarity_score from retrieval so they are always real.
    """
    by_name = {c["name"]: c for c in retrieved}   # fast lookup of the authoritative record per service name
    out: list[RankedMatch] = []                   # reconciled shortlist, in the model's order
    seen: set[str] = set()                        # guard against the model listing the same service twice

    for m in model_matches:                       # walk the model's ranking in order
        source = by_name.get(m.name)              # the retrieval record this match claims to be
        if source is None or m.name in seen:      # invented name, or a duplicate — drop it
            continue
        seen.add(m.name)
        out.append(
            RankedMatch(
                name=m.name,                                     # verified against retrieval
                why=m.why.strip(),                               # the model's contribution, kept
                grounding_passage=source["grounding_passage"],   # authoritative text, from retrieval
                similarity_score=source.get("similarity_score"),  # authoritative score, from retrieval
            )
        )
        if len(out) == 3:                         # never return more than the top 3
            break
    return out


def run_match_and_rank(state: RespiteState) -> dict:
    """
    LangGraph node: `state -> {"candidate_matches": [RankedMatch, ...],
    "iteration_count": <bumped>}`.

    If retrieval returns nothing (everything filtered out on budget/area), the
    model is not called at all and the node returns an empty shortlist — the
    graph can then route to a "loosen your constraints" branch.
    """
    profile: CaregiverProfile | None = state.get("caregiver_profile")  # set by the Intake node
    if profile is None:                                                # defensive: node run out of order
        raise ValueError("run_match_and_rank called before caregiver_profile was set")

    bumped = state.get("iteration_count", 0) + 1                        # this node counts as one step either way

    # Stage 1: hybrid retrieval. Pass the profile as a plain dict — that is the
    # shape search_services expects, and model_dump gives budget/area/None keys.
    retrieved = search_services(profile.model_dump(), k=8)

    if not retrieved:                                                  # nothing survived the hard filters
        return {"candidate_matches": [], "iteration_count": bumped}

    # Stage 2: LLM rerank. response_schema=RankedMatches -> validated before we see it.
    ranked = llm.complete(
        [
            {"role": "system", "content": _RERANK_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Caregiver profile:\n{profile.model_dump_json(indent=2)}\n\n"   # the stated need, verbatim
                    f"Candidate services:\n{_format_candidates(retrieved)}"          # name + passage per candidate
                ),
            },
        ],
        model_role="reasoning",                                        # ranking nuance is worth the stronger model
        response_schema=RankedMatches,                                 # guardrail: validated JSON only
        max_tokens=1024,                                               # room for 3 rationales + hidden reasoning tokens
    )

    # Guardrail: never let the model's factual fields or invented names through.
    reconciled = _reconcile(ranked.matches, retrieved)

    return {"candidate_matches": reconciled, "iteration_count": bumped}
