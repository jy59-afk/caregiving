"""
state.py — the single typed object that flows through the LangGraph pipeline,
plus the Pydantic schemas every model output is validated against.

Why this module exists:
  * CLAUDE.md's state contract lives in exactly one place. Nodes read/write
    `RespiteState`; they never invent ad-hoc dict keys.
  * Every schema here is a guardrail. A node that calls the LLM passes one of
    these as `response_schema=`, so `llm.complete()` refuses to return
    anything that does not validate (see `src/llm.py`).
  * Heavy objects (the transcript, the candidate matches, the draft) live in
    the state, never in a prompt — keeping prompts small and auditable.

Nothing here talks to a network or a model; it is pure data definition, so it
is cheap to import from tests.
"""

from __future__ import annotations  # allow `str | None` style hints everywhere

from typing import Literal, TypedDict  # LangGraph merges partial-dict updates into a TypedDict cleanly

from pydantic import BaseModel, Field  # schema container + per-field metadata used in prompts


# ==========================================================================
# 1. Intake — what the caregiver told us, distilled
# ==========================================================================

class CaregiverProfile(BaseModel):
    """
    The structured need, extracted from the free-text conversation by the
    Intake node. Only `needs_description` is required — a caregiver may not
    have stated a budget or an area yet, and the retrieval tool treats a
    missing value as "no constraint" rather than failing.

    The field descriptions are written for the model: they are what the
    extraction prompt leans on to decide what goes where.
    """

    needs_description: str = Field(
        description=(
            "One or two plain sentences describing the CARE RECIPIENT's needs and "
            "situation in the caregiver's own framing: their condition, behaviours, "
            "mobility, and what kind of cover the caregiver is looking for. This is "
            "the text that gets semantically matched against services, so keep it "
            "about the care need, not about logistics."
        )
    )
    budget: float | None = Field(
        default=None,
        description=(
            "The caregiver's stated ceiling on cost, as a number in Singapore "
            "dollars. If they gave a per-hour figure, leave this null unless they "
            "also implied a per-session or per-day cap. Null when no budget was stated."
        ),
    )
    area: str | None = Field(
        default=None,
        description=(
            "The Singapore town / neighbourhood the caregiver wants the service "
            "near (e.g. 'Toa Payoh', 'Bishan'). Null if they did not say."
        ),
    )
    schedule: str | None = Field(
        default=None,
        description=(
            "When the caregiver needs cover, in plain words: e.g. 'weekday "
            "daytime while at work', 'weekends only', 'overnight', 'a three-week "
            "block in December'. Null if not stated."
        ),
    )
    relationship: str | None = Field(
        default=None,
        description=(
            "Who the caregiver is caring for, e.g. 'mother', 'husband', 'father'. "
            "Null if not stated."
        ),
    )
    clarifying_question: str | None = Field(
        default=None,
        description=(
            "If — and only if — a critical fact is missing that would make a "
            "recommendation unreliable (no sense of the care need at all, or no "
            "budget when cost is clearly the caregiver's main worry), the single "
            "most useful question to ask next. Otherwise null."
        ),
    )
    wants_draft: bool = Field(
        default=False,
        description=(
            "True if the caregiver's LATEST message explicitly asks to draft, "
            "write, prepare, or send an enquiry / email / message to a service "
            "(e.g. 'draft me the email', 'write to them', 'contact St Luke's'). "
            "False otherwise — the pipeline drafts on a normal match anyway; "
            "this flag matters only for the near-miss list."
        ),
    )
    selected_option: str | None = Field(
        default=None,
        description=(
            "If the caregiver's LATEST message picks one specific service the "
            "assistant just listed — by full or partial name ('St Luke's Teck "
            "Whye'), or by position ('the first one', 'option 2', 'number 3') — "
            "the EXACT service name as it appeared in that list. If they picked "
            "by position, still resolve it to the exact name. Null if they did "
            "not pick a listed option."
        ),
    )


# ==========================================================================
# 2. Match & Rank — the <=4 services we are recommending, in order
# ==========================================================================

class RankedMatch(BaseModel):
    """
    One recommended service after the rerank pass. `grounding_passage` and
    `similarity_score` are copied verbatim from the retrieval tool's output —
    the model is not allowed to author them — so the "why this fits" line in
    `why` can always be checked against real service content.
    """

    name: str = Field(description="Exact service name, copied from the retrieved candidate — never paraphrased or invented.")
    why: str = Field(
        description=(
            "One sentence, addressed to the caregiver, on why this service fits "
            "their stated need. Must be supported by the grounding passage — no "
            "claims about fees, availability or outcomes that aren't in it."
        )
    )
    grounding_passage: str = Field(description="The retrieved service passage this recommendation is grounded in. Set by the node, not the model.")
    similarity_score: float | None = Field(
        default=None,
        description="Retrieval distance (lower = closer). For logging/eval only; set by the node, not the model.",
    )


class RankedMatches(BaseModel):
    """The rerank node's whole output: an ordered shortlist, best first, at most 4."""

    matches: list[RankedMatch] = Field(
        description="Between 0 and 4 services, ordered best fit first. Drop any candidate that does not genuinely fit rather than padding to four.",
    )


# ==========================================================================
# 3. Explain & Draft — the message the caregiver would send
# ==========================================================================

class DraftedMessage(BaseModel):
    """
    A ready-to-review enquiry message for the top-matched service. This is a
    DRAFT: V1 has no tool that sends it. The caregiver reads it, edits it, and
    sends it themselves.
    """

    service_name: str = Field(description="The service this message is addressed to (the top match).")
    subject: str = Field(description="A short subject line for an email or contact-form enquiry.")
    body: str = Field(
        description=(
            "The message body, in the caregiver's first-person voice. 4-8 short "
            "sentences: who they care for and the key needs, the cover they are "
            "looking for (days/hours), any budget context, and 2-3 concrete "
            "questions (availability, fees after subsidy, transport, assessment "
            "process). Only facts the caregiver actually gave — no invented "
            "details. Do not promise or assume a booking."
        )
    )


# ==========================================================================
# 4. The graph state — CLAUDE.md's contract, verbatim
# ==========================================================================

class RespiteState(TypedDict, total=False):
    """
    The object LangGraph threads through Intake -> Match & Rank -> Explain &
    Draft -> (consent) -> Output. `total=False` so a node can return just the
    keys it produces and LangGraph merges them in.

    Keys:
      messages          — the conversation so far, chat-completions shape
                          [{"role": "user"|"assistant"|"system", "content": str}].
      caregiver_profile — CaregiverProfile once Intake has run, else absent/None.
      candidate_matches — the ordered RankedMatch shortlist from Match & Rank.
      fallback_matches  — set by Match & Rank ONLY when candidate_matches is
                          empty: the 4 next-best services (budget > schedule >
                          location priority, hard filters ignored), shown with
                          a caveat and no draft.
      drafted_message   — DraftedMessage from Explain & Draft, else None.
      consent_given     — HITL gate: the caregiver has seen the draft and
                          approved showing/using it. Nothing auto-sends regardless.
      iteration_count   — incremented by each node; the guardrail slice caps it
                          at settings.max_iterations.

    Terminal-output keys (written only by the Output node in `src/graph.py` —
    they are the graph's result surface, not something the worker nodes touch):
      outcome           — which terminal branch the run ended on.
      final_response    — the plain-text message the UI shows the caregiver.
    """

    messages: list[dict[str, str]]
    caregiver_profile: CaregiverProfile | None
    candidate_matches: list[RankedMatch]
    fallback_matches: list[RankedMatch]
    drafted_message: DraftedMessage | None
    consent_given: bool
    iteration_count: int
    outcome: Outcome | None
    final_response: str | None


# The ways a graph run can terminate. `matches_ready` is the success path; the
# rest are all legitimate, non-error endings the UI renders differently.
Outcome = Literal[
    "matches_ready",         # >=1 grounded match + a draft the caregiver can review
    "fallback_matches",      # nothing cleared every constraint — the 4 next-best shown with a caveat, no draft
    "no_matches",            # not reachable while the dataset is non-empty; kept as a defensive end state
    "needs_clarification",   # Intake couldn't proceed without asking the caregiver something
    "stopped_iteration_cap",  # the hard iteration cap tripped before a result was ready
]


def new_state(messages: list[dict[str, str]] | None = None) -> RespiteState:
    """
    Build a fresh state with every key present and sensibly empty. Nodes and
    tests start from this rather than half-populating a bare dict.
    """
    return RespiteState(
        messages=list(messages or []),   # copy so the caller's list isn't aliased into state
        caregiver_profile=None,          # set by the Intake node
        candidate_matches=[],            # set by the Match & Rank node
        fallback_matches=[],             # set by Match & Rank only on the no-match path
        drafted_message=None,            # set by the Explain & Draft node
        consent_given=False,             # HITL gate starts closed
        iteration_count=0,               # bumped by each node that runs
        outcome=None,                    # set once, by the Output node
        final_response=None,             # set once, by the Output node
    )


def render_transcript(messages: list[dict[str, str]]) -> str:
    """
    Flatten the message list into a plain-text transcript for an extraction
    prompt. `system` turns are dropped — they are orchestration scaffolding,
    not something the caregiver said. Speakers are relabelled so the model
    reads it as a dialogue, not as its own chat history.
    """
    lines: list[str] = []                                  # collects one "Speaker: text" line per turn
    for msg in messages:                                   # walk the conversation in order
        role = msg.get("role", "")                         # "user" / "assistant" / "system"
        if role == "system":                               # skip scaffolding turns
            continue
        content = msg.get("content", "").strip()           # trim stray whitespace per turn
        if not content:                                    # skip blank turns — they add no signal
            continue
        speaker = "Caregiver" if role == "user" else "Assistant"   # human-readable label
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)                                # newline-separated transcript
