"""
match_rank.py — the second worker node.

Job: turn a `CaregiverProfile` into an ordered shortlist of at most four
services, each with a grounded one-line "why this fits".

Two stages:
  1. Retrieval (already built) — `retrieval_tool.search_services` does hybrid
     search: semantic similarity, then hard filters on budget and area. It
     returns <=4 candidates. Budget/area are enforced here, not by the model.
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
import re  # tokenising the caregiver's stated schedule for the fallback ranker

import llm  # model seam; call as llm.complete so tests can monkeypatch it
from ingest import load_services, to_document_text  # raw records + the same passage-flattener the index is built from — for the no-match fallback
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
    "- Keep the candidates that plausibly fit — aim to return the top 4 so the "
    "caregiver has real options to compare. Drop a candidate only when it "
    "clearly does not fit (wrong care type, wrong schedule, or acuity the "
    "service plainly can't handle). Returning fewer than four is right only "
    "when the rest are clear mismatches. Never add a service not in the list.\n"
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
    similarity_score from retrieval so they are always real. Caps at 4.
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
        if len(out) == 4:                         # never return more than the top 4
            break
    return out


# ==========================================================================
# No-match fallback — the "next 4 best" list
# ==========================================================================
#
# When retrieval + rerank leave the shortlist empty, it means no service
# clears every constraint the caregiver gave (care need, schedule, hours, and
# the budget/area hard filters). Rather than tell them to loosen something, we
# still hand them the four closest services, ranked in a fixed priority:
#
#     1. location  — a service in the stated area beats one outside it. The
#                    caregiver is already stretched for time; a nearby option
#                    they can actually get to beats a cheaper one across town.
#     2. budget    — among services equally near, one within the stated ceiling
#                    beats one over it; among those, the cheaper wins.
#     3. schedule  — a service whose days/hours plausibly cover the stated
#                    timing beats one that doesn't (crude keyword check).
#
# (If the caregiver stated no area, location is neutral and this reduces to
# budget-then-schedule.)
#
# These are shown with an explicit caveat and NO draft — the caregiver has to
# check what each place can actually accommodate. The ranking is deterministic
# (no model call), so it is auditable and cheap.

# Short, common words to ignore when matching the schedule text against a
# service's days/hours — they carry no timing signal.
_SCHEDULE_STOPWORDS = frozenset({
    "need", "needs", "cover", "care", "while", "during", "some", "week",
    "with", "them", "have", "want", "just", "only", "from", "time", "hours",
})

# Timing words a caregiver uses -> extra strings that mean the same thing in a
# service's days/hours text. Keeps the crude matcher from flagging, say, a
# "Mon-Fri" day centre as not covering "weekday daytime".
_SCHEDULE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "weekday": ("mon-fri", "mon - fri", "monday", "weekdays"),
    "weekdays": ("mon-fri", "mon - fri", "weekday"),
    "daytime": ("day-care", "day care", "am-", "am -", "daycare"),
    "daily": ("mon-fri", "mon-sat", "mon-sun", "everyday"),
    "weekend": ("sat", "sun", "saturday", "sunday"),
    "weekends": ("sat", "sun", "saturday", "sunday"),
    "overnight": ("night", "24-hour", "24 hour", "residential", "nightly"),
    "night": ("overnight", "24-hour", "nightly", "residential"),
    "nights": ("overnight", "24-hour", "nightly", "residential"),
    "block": ("30 days", "short-stay", "residential", "up to 30"),
    "weeks": ("30 days", "short-stay", "residential"),
    "residential": ("short-stay", "24-hour", "30 days"),
}


def _schedule_mismatch(service: dict, schedule_text: str) -> int:
    """
    0 if any meaningful word from the caregiver's stated schedule (or a known
    synonym of it) also appears in the service's days/hours/care_type text,
    else 1. Deliberately crude — it is a tie-breaker and a caveat hint, not a
    classifier. No stated schedule -> never a mismatch.
    """
    if not schedule_text:                                     # caregiver gave no timing — don't penalise anyone
        return 0
    # days/hours carry the structured timing; the description often spells it
    # out in the caregiver's own words ("weekday", "overnight", "weekend").
    haystack = f"{service['days']} {service['hours']} {service['care_type']} {service.get('description', '')}".lower()
    tokens = {                                                # content words from the stated schedule
        t for t in re.findall(r"[a-z]+", schedule_text.lower())
        if len(t) > 3 and t not in _SCHEDULE_STOPWORDS
    }
    for tok in tokens:                                        # direct hit or a synonym hit both count
        if tok in haystack or any(syn in haystack for syn in _SCHEDULE_SYNONYMS.get(tok, ())):
            return 0
    return 1                                                  # no overlap at all == flag it


def _fallback_sort_key(service: dict, profile: CaregiverProfile) -> tuple:
    """
    The location > budget > schedule priority, as a tuple that sorts ascending
    with 0 = best. Trailing `cost` and `distance_km` break ties inside each band.
    """
    cost = service["cost_per_session"]
    # 1. location: 0 if the service sits in the stated area (substring match),
    #    else 1. Neutral (always 0) when the caregiver named no area.
    area_key = 0
    if profile.area:
        area_key = 0 if profile.area.lower() in service["area"].lower() else 1
    # 2. budget: 0 while within the ceiling, else dollars over it.
    budget_key = 0.0 if profile.budget is None else max(0.0, cost - profile.budget)
    # 3. schedule: 0 if the days/hours plausibly cover the stated timing.
    schedule_key = _schedule_mismatch(service, profile.schedule or "")
    return (area_key, budget_key, schedule_key, cost, service.get("distance_km", 0.0))


def _fallback_why(service: dict, profile: CaregiverProfile) -> str:
    """
    A plain, honest one-liner for a fallback match: what it costs, where it is,
    and that it is only a near-miss. Built from the record's own fields, so it
    is trivially checkable against the grounding passage.
    """
    cost = service["cost_per_session"]
    bits: list[str] = []
    # location first — it's the top ranking factor, so lead the caveat with it.
    if profile.area and profile.area.lower() not in service["area"].lower():
        bits.append(f"in {service['area']}, not {profile.area}")
    elif profile.area:
        bits.append(f"in {service['area']}")
    if profile.budget is not None and cost > profile.budget:
        bits.append(f"about S${cost:g}/session, over your S${profile.budget:g} budget")
    elif profile.budget is not None:
        bits.append(f"about S${cost:g}/session, within budget")
    else:
        bits.append(f"about S${cost:g}/session")
    if profile.schedule and _schedule_mismatch(service, profile.schedule):
        bits.append(f"provides {service['care_type'].replace('-', ' ')}, not the '{profile.schedule}' cover you described")
    return (
        "Closest available option — " + "; ".join(bits) +
        ". Shown because nothing matched every constraint; check what they can take before enquiring."
    )


def _fallback_shortlist(profile: CaregiverProfile, exclude: set[str]) -> list[RankedMatch]:
    """
    The four next-best services by the location > budget > schedule priority,
    ignoring every hard filter. `exclude` drops names already tried (there are
    none today, but the rerank could in future return a partial list).
    """
    services = [s for s in load_services() if s["name"] not in exclude]  # whole curated dataset
    ranked = sorted(services, key=lambda s: _fallback_sort_key(s, profile))  # apply the fixed priority
    return [
        RankedMatch(
            name=s["name"],                          # real record name
            why=_fallback_why(s, profile),           # deterministic caveat, grounded in the record
            grounding_passage=to_document_text(s),   # same flattened passage the index uses
            similarity_score=None,                   # not a semantic hit — there is no score
        )
        for s in ranked[:4]                          # exactly the next 4
    ]


def run_match_and_rank(state: RespiteState) -> dict:
    """
    LangGraph node: `state -> {"candidate_matches": [RankedMatch, ...],
    "fallback_matches": [RankedMatch, ...], "iteration_count": <bumped>}`.

    Normal path: hybrid retrieval -> LLM rerank -> reconciled shortlist in
    `candidate_matches`.

    No-match path: if retrieval finds nothing that clears the hard filters, or
    the rerank drops every candidate, `candidate_matches` stays empty and
    `fallback_matches` carries the four next-best services (budget > schedule >
    location priority, filters ignored) for the Output node to surface with a
    caveat. No draft is made for a fallback match.
    """
    profile: CaregiverProfile | None = state.get("caregiver_profile")  # set by the Intake node
    if profile is None:                                                # defensive: node run out of order
        raise ValueError("run_match_and_rank called before caregiver_profile was set")

    bumped = state.get("iteration_count", 0) + 1                        # this node counts as one step either way

    # Stage 1: hybrid retrieval. Pass the profile as a plain dict — that is the
    # shape search_services expects, and model_dump gives budget/area/None keys.
    retrieved = search_services(profile.model_dump(), k=8)

    reconciled: list[RankedMatch] = []
    if retrieved:                                                      # something cleared the hard filters
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
            model_role="reasoning",                                    # ranking nuance is worth the stronger model
            response_schema=RankedMatches,                             # guardrail: validated JSON only
            max_tokens=3000,                                           # room for up to 4 rationales + gpt-oss hidden reasoning tokens
        )
        # Guardrail: never let the model's factual fields or invented names through.
        reconciled = _reconcile(ranked.matches, retrieved)

    if reconciled:                                                     # normal success path
        return {"candidate_matches": reconciled, "iteration_count": bumped}

    # No-match path: retrieval empty, or the rerank kept nothing. Hand back the
    # four next-best services instead of a dead end.
    fallback = _fallback_shortlist(profile, exclude={m.name for m in reconciled})
    return {
        "candidate_matches": [],
        "fallback_matches": fallback,
        "iteration_count": bumped,
    }
