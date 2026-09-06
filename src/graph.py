"""
graph.py — wires the three worker nodes into one LangGraph state graph, behind
the guardrails from `src/guardrails.py`.

Flow (CLAUDE.md's V1 architecture):

    Intake ─▶ Match & Rank ─▶ Explain & Draft ─▶ Consent gate ─▶ Output ─▶ END
       │            │
       │            └─(no candidates)──────────────────────────▶ Output
       └─(clarifying question, or iteration cap)────────────────▶ Output

Guardrails enforced at this layer:
  * Iteration cap — `guardrails.iteration_cap_reached` is checked in the
    conditional edge after every worker node; when it trips, routing goes
    straight to Output with `outcome="stopped_iteration_cap"`.
  * Tool allow-list — the graph only ever wires `Match & Rank` (which calls
    `search_services`) and `Explain & Draft` (the `draft_message` equivalent).
    No node here sends, books, or contacts anyone. `guardrails.ALLOWED_TOOLS`
    and its source scan are the machine-checkable statement of that.
  * Schema validation — already inside `llm.complete`; a node that raises
    `LLMSchemaError` propagates it out of `run(...)`, aborting the run rather
    than emitting unvalidated text (there is no try/except swallowing it here).

The Consent gate is the HITL checkpoint. In V1 there is no sender, so the gate
does not branch on consent — it is the seam where a future "release the draft"
decision would live, and it records the (default-False) consent flag so the
Output node can word itself as "here is a draft for you to send" rather than
implying anything was sent.

Build the compiled graph once with `build_graph()`; run a transcript through it
with `run(messages)`.
"""

from __future__ import annotations

import logging  # node-transition channel — CLAUDE.md's "log node transitions + tool-call outcomes"
import time  # per-node wall-clock, for the exit log line

from langgraph.graph import END, START, StateGraph  # minimal state-machine primitives

import guardrails  # iteration cap + allow-list live here; imported as a module so tests can patch settings
from explain_draft import run_explain_and_draft  # node 3 — draft the enquiry message (no send)
from intake import run_intake  # node 1 — transcript -> CaregiverProfile
from match_rank import run_match_and_rank  # node 2 — profile -> ranked, grounded shortlist
from state import RespiteState, new_state  # state contract + fresh-state builder

# One named logger for the whole orchestration layer. Callers decide where it
# goes: `python src/graph.py` and the Streamlit UI attach an INFO handler; the
# test suite leaves it at WARNING so runs stay silent. Nothing here prints
# directly — that keeps the trace opt-in and out of pytest output.
log = logging.getLogger("respite.graph")


# ==========================================================================
# Observability — one wrapper that traces every node
# ==========================================================================

def _traced(name: str, fn):
    """
    Wrap a node function so each entry and exit is logged on one line.

    Wrapping here rather than editing every node module means all four callers
    — `run()`, `python src/graph.py`, the eval harness and the Streamlit UI —
    get an identical node-by-node trace for free, while the worker nodes stay
    free of logging noise. The exit line names the state keys the node wrote,
    so the route taken is legible after the fact (e.g. `intake` writing only
    `caregiver_profile`/`iteration_count` then jumping to `output` == a
    clarifying question was raised).
    """
    def _wrapped(state: RespiteState) -> dict:  # same signature LangGraph expects of a node
        step = state.get("iteration_count", 0)  # steps already taken when this node starts
        log.info("-> %-13s  (step %d)", name, step)  # entering the node
        started = time.perf_counter()  # start the node's stopwatch
        result = fn(state)  # run the real node
        elapsed_ms = (time.perf_counter() - started) * 1000  # node duration in ms
        log.info(  # leaving the node
            "<- %-13s  %4.0f ms  wrote %s",
            name, elapsed_ms, sorted((result or {}).keys()),
        )
        return result  # hand the node's partial update straight back to LangGraph
    return _wrapped


# ==========================================================================
# Routing helpers — the conditional edges
# ==========================================================================

def _route_after_intake(state: RespiteState) -> str:
    """
    After Intake: stop early if the cap tripped or if Intake decided it needs
    to ask the caregiver something before a recommendation is worth making;
    otherwise proceed to Match & Rank.
    """
    if guardrails.iteration_cap_reached(state):  # guardrail 1 — never loop past the cap
        log.info("   route: intake -> output  (iteration cap)")  # why we short-circuit
        return "output"
    profile = state.get("caregiver_profile")  # Intake always sets this
    if profile is not None and profile.clarifying_question:  # a critical fact is missing
        log.info("   route: intake -> output  (clarifying question)")  # missing a must-have fact
        return "output"  # Output will surface the question to the caregiver
    log.info("   route: intake -> match_rank")  # enough to go on
    return "match_rank"  # enough to go on — retrieve + rank


def _route_after_match(state: RespiteState) -> str:
    """
    After Match & Rank: stop if the cap tripped or if nothing survived
    retrieval + rerank (Output then suggests loosening budget/area);
    otherwise draft for the top match.
    """
    if guardrails.iteration_cap_reached(state):  # guardrail 1
        log.info("   route: match_rank -> output  (iteration cap)")  # why we short-circuit
        return "output"
    if not state.get("candidate_matches"):  # empty shortlist — no point drafting
        log.info("   route: match_rank -> output  (empty shortlist)")  # nothing survived filter + rerank
        return "output"
    log.info("   route: match_rank -> explain_draft")  # at least one match to draft for
    return "explain_draft"


# ==========================================================================
# Terminal nodes — Consent gate + Output
# ==========================================================================

def run_consent_gate(state: RespiteState) -> dict:
    """
    HITL checkpoint. V1 has no tool that sends the draft, so this gate cannot
    "approve a send" — there is nothing to send. It exists to:
      * make the human-in-the-loop step explicit in the graph topology, and
      * normalise `consent_given` to a real bool for the Output node.
    It performs no side effect and calls no model.
    """
    return {"consent_given": bool(state.get("consent_given", False))}  # pass-through, coerced to bool


def _format_matches(state: RespiteState) -> str:
    """Render the ranked shortlist as a short, plain-text list for the caregiver."""
    lines: list[str] = []  # one "1. Name — why" line per match
    for i, match in enumerate(state.get("candidate_matches", []), start=1):  # 1-based for humans
        lines.append(f"{i}. {match.name} — {match.why}")  # name + the grounded one-liner
    return "\n".join(lines)


def run_output(state: RespiteState) -> dict:
    """
    Terminal node: decide which of the four outcomes the run ended on and write
    the caregiver-facing `final_response`. Writes nothing else of substance —
    this is the graph's result surface, not another processing step.

    Order of checks matters: the iteration cap is reported even if, say, a
    clarifying question is also set, because "we stopped early" is the more
    important thing to tell the caregiver.
    """
    profile = state.get("caregiver_profile")  # may be None only if the graph was mis-built
    matches = state.get("candidate_matches", [])  # ranked shortlist, possibly empty
    draft = state.get("drafted_message")  # DraftedMessage or None

    if guardrails.iteration_cap_reached(state):  # guardrail 1 tripped somewhere upstream
        outcome = "stopped_iteration_cap"
        response = (
            "I stopped before finishing because this request hit the built-in "
            "step limit. Nothing was sent. Please try again with a bit more "
            "detail about the care need, budget and timing so I can get there faster."
        )
    elif profile is not None and profile.clarifying_question:  # Intake needs an answer first
        outcome = "needs_clarification"
        response = profile.clarifying_question  # ask exactly what Intake asked for
    elif not matches:  # retrieval + rerank found nothing that fits
        outcome = "no_matches"
        response = (
            "I couldn't find a respite service that fits those constraints. "
            "Widening the budget or the area — or allowing a different type of "
            "care (day care, in-home, short-stay) — would give me something to match."
        )
    else:  # the success path — at least one grounded match, and a draft to review
        outcome = "matches_ready"
        parts = [
            "Here are the respite options that best fit what you described:",
            _format_matches(state),
        ]
        if draft is not None:  # Explain & Draft ran and produced a message
            parts.append(
                f"\nI've drafted an enquiry to {draft.service_name} for you to review, "
                f"edit and send yourself:\n\n"
                f"Subject: {draft.subject}\n\n{draft.body}"
            )
        response = "\n".join(parts)

    log.info("   outcome: %s", outcome)  # the one line that says how the run ended
    return {"outcome": outcome, "final_response": response}  # the only keys Output writes


# ==========================================================================
# Graph assembly
# ==========================================================================

def build_graph():
    """
    Assemble and compile the state graph. Cheap to call; nodes do no work until
    the compiled graph is invoked. Kept as a function (not a module-level
    singleton) so a test can rebuild it after patching settings.
    """
    builder = StateGraph(RespiteState)  # the state schema drives partial-update merging

    # --- nodes -----------------------------------------------------------
    # Every node is wrapped in `_traced(...)` so the compiled graph emits a
    # node-by-node log line no matter who invokes it.
    builder.add_node("intake", _traced("intake", run_intake))  # transcript -> profile
    builder.add_node("match_rank", _traced("match_rank", run_match_and_rank))  # profile -> ranked shortlist (calls search_services)
    builder.add_node("explain_draft", _traced("explain_draft", run_explain_and_draft))  # top match -> draft message (no send)
    builder.add_node("consent_gate", _traced("consent_gate", run_consent_gate))  # HITL checkpoint (no side effect)
    builder.add_node("output", _traced("output", run_output))  # decide outcome + write final_response

    # --- edges -----------------------------------------------------------
    builder.add_edge(START, "intake")  # every run starts at Intake

    builder.add_conditional_edges(  # guardrail 1 + clarifying-question branch
        "intake",
        _route_after_intake,
        {"match_rank": "match_rank", "output": "output"},
    )
    builder.add_conditional_edges(  # guardrail 1 + empty-shortlist branch
        "match_rank",
        _route_after_match,
        {"explain_draft": "explain_draft", "output": "output"},
    )

    builder.add_edge("explain_draft", "consent_gate")  # draft always goes through the HITL seam
    builder.add_edge("consent_gate", "output")  # then to the result surface
    builder.add_edge("output", END)  # Output is terminal

    return builder.compile()  # -> a runnable graph


def run(messages: list[dict[str, str]], *, consent_given: bool = False) -> RespiteState:
    """
    Convenience entry point: build a fresh state from `messages`, run it through
    the compiled graph, and return the final state (which carries `outcome` and
    `final_response`).

    `consent_given` is accepted so a caller/UI can pre-set the HITL flag; it
    changes nothing that sends, because nothing sends.
    """
    state = new_state(messages)  # every key present-and-empty
    state["consent_given"] = consent_given  # seed the HITL flag from the caller
    graph = build_graph()  # compile fresh (see build_graph docstring)
    return graph.invoke(state)  # LangGraph threads the state through to END


if __name__ == "__main__":
    # Manual smoke run — needs a real LLM backend configured and the FAISS index
    # built (`python src/ingest.py`). Unit tests mock both and don't need either.
    import sys  # only needed here — to force UTF-8 on the Windows console

    # Model drafts contain smart quotes / non-breaking hyphens (‑); the
    # default Windows cp1252 stdout raises UnicodeEncodeError on them.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    logging.basicConfig(  # attach an INFO handler so the node trace is visible on the console
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    for _noisy in ("httpx", "huggingface_hub", "faiss", "sentence_transformers"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)  # keep the trace readable — mute library chatter
    demo_messages = [
        {
            "role": "user",
            "content": (
                "My mother has Alzheimer's and gets agitated in the afternoons. "
                "I lecture part-time and need weekday daytime cover near Toa Payoh. "
                "I can't really go above $85 a session."
            ),
        }
    ]
    # Note: Toa Payoh + ~$85 lands on the dementia day-care records. Narrowing
    # this to an area/budget with no genuine fit (e.g. weekday cover in Bishan,
    # which only has weekend respite) correctly returns outcome="no_matches".
    final = run(demo_messages)  # Intake -> Match & Rank -> Explain & Draft -> Output
    print(f"outcome: {final.get('outcome')}\n")  # which branch we ended on
    print(final.get("final_response"))  # what the caregiver would see
