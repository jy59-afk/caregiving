"""
app.py — the thinnest possible chat front-end over the LangGraph pipeline.

Deliberately NOT the graded component (CLAUDE.md): it owns no matching logic,
no prompts, no schema. Every user turn is handed to `graph` as the full
transcript; whatever `graph` puts in `final_response` / `outcome` /
`candidate_matches` / `drafted_message` is what the page renders.

The one hard rule this UI must visibly honour: it never sends anything. When a
draft exists it is shown in a copy-friendly block for the caregiver to send
themselves — there is no "send" button because there is no send tool.

Run:  streamlit run src/app.py   (needs LLM_BACKEND configured in .env and the
FAISS index built via `python src/ingest.py` — the page guards for the latter).
"""

from __future__ import annotations

import logging  # so the graph's node-transition trace (see graph._traced) surfaces in the launch terminal

import streamlit as st  # the whole UI toolkit for this file

import graph  # the compiled pipeline; imported as a module so we call graph.run() and share its logger config
from config import settings  # typed view of .env — backend name + index path for the pre-flight check
from llm import LLMError  # provider/guardrail failures we want to show as a message, not a stack trace

# --------------------------------------------------------------------------
# Logging — the graph emits one line per node (graph._traced); `streamlit run`
# surfaces stdout/stderr in the terminal it was launched from, so configuring
# the root logger here is enough for a screen-capture of the trace.
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # INFO is our node-transition channel (graph._traced logs here)
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",  # timestamped + named so transitions are legible on video
)
for _noisy in ("httpx", "huggingface_hub", "faiss", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)  # mute library chatter so the node trace stands out
log = logging.getLogger("respite.app")  # named logger so it can be filtered from library noise


# --------------------------------------------------------------------------
# The one place this file touches the graph. A thin bookend around
# `graph.run()`: the graph itself logs every node transition (graph._traced),
# so this only adds a start/end line naming the turn and the backend.
# --------------------------------------------------------------------------
def run_pipeline(messages: list[dict[str, str]]) -> dict:
    """
    Run the full transcript through the compiled graph and return the final
    state (carrying `outcome` and `final_response`).

    All matching, ranking and drafting — and the per-node trace — live in
    `graph`. This wrapper exists only to frame each turn in the log.
    """
    log.info("turn start — %d message(s), backend=%s", len(messages), settings.llm_backend)
    final = graph.run(messages, consent_given=False)  # HITL flag stays closed; V1 has nothing that flips it
    log.info("turn end — outcome=%s", final.get("outcome"))
    return final


# --------------------------------------------------------------------------
# Pre-flight — Match & Rank loads the FAISS index lazily on first call and
# raises deep in faiss if it was never built. Catch that here and tell the
# caregiver-facing operator what to run, instead of dumping a traceback.
# --------------------------------------------------------------------------
def index_is_built() -> bool:
    """True when `python src/ingest.py` has produced a loadable FAISS index."""
    return (settings.index_path / "index.faiss").exists()  # the file retrieval_tool.FAISS.load_local opens first


# --------------------------------------------------------------------------
# Rendering helpers — pure presentation, no logic. Each takes the final state.
# --------------------------------------------------------------------------
def md_safe(text: str) -> str:
    """
    Escape the one Markdown metacharacter that bites us here: a lone `$`.
    Streamlit's `st.markdown` treats `$…$` as LaTeX, so a draft that mentions
    "S$80 per session" renders as garbled math. Service passages and money
    figures are full of dollar signs, so escape them before any markdown call.
    """
    return text.replace("$", "\\$")  # backslash-escape; harmless in text that had no math


def render_matches(state: dict) -> None:
    """Show the ranked shortlist: name, the grounded 'why', and a peek at the source passage."""
    matches = state.get("candidate_matches", [])  # list[RankedMatch], best first, <=3
    if not matches:  # defensive — caller only invokes this on the matches_ready path
        return
    st.markdown("#### Options that fit")  # section heading
    for i, m in enumerate(matches, start=1):  # 1-based numbering for humans
        with st.container(border=True):  # visually group each option
            st.markdown(f"**{i}. {md_safe(m.name)}**")  # exact service name (reconciled against retrieval)
            st.markdown(md_safe(m.why))  # one sentence, addressed to the caregiver, grounded in the passage
            with st.expander("Where this comes from"):  # keep the raw passage available but out of the way
                st.caption(md_safe(m.grounding_passage))  # the retrieved service text the 'why' must be supported by


def render_draft(state: dict) -> None:
    """Show the enquiry draft in a copy-friendly block. The caregiver sends it — the app never does."""
    draft = state.get("drafted_message")  # DraftedMessage or None
    if draft is None:  # Explain & Draft may not have run (e.g. no matches)
        return
    st.markdown(f"#### Draft enquiry to {md_safe(draft.service_name)}")  # whom the message is addressed to
    st.caption("Review and edit this, then send it yourself. This tool does not contact any service for you.")
    # A text area is the most reliable 'select-all and copy' surface in Streamlit
    # and makes it obvious the text is the deliverable, not a sent message. No
    # `key=` on purpose: a keyed widget would keep its first value and ignore a
    # newer draft on a later turn — we want `value=` to win every rerun.
    st.text_area(
        label="Subject",  # labelled so screen readers announce the field
        value=draft.subject,  # short subject line from the draft
        height=68,  # one line's worth
    )
    st.text_area(
        label="Message",  # the body the caregiver will paste into an email / contact form
        value=draft.body,  # first-person, grounded, no assumed booking
        height=260,  # room for the 4-8 sentence body without scrolling
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Respite Navigator", page_icon="🧭")  # browser-tab identity

st.title("Respite Navigator")  # the product name
st.caption(
    "Tell me about the care situation and what kind of break you need. "
    "I'll match Singapore respite options to your hours and budget, and draft "
    "an enquiry you can send yourself."
)  # sets expectations: matching + drafting, not booking

# --- sidebar: unobtrusive run metadata --------------------------------------
with st.sidebar:  # kept out of the main flow — operators glance here, caregivers don't need to
    st.subheader("Session")
    st.write(f"Model backend: `{settings.llm_backend}`")  # groq (dev) / bedrock (demo)
    st.write(f"Max steps: {settings.max_iterations}")  # the hard iteration cap in force
    if st.session_state.get("last_outcome"):  # populated after the first exchange
        st.write(f"Last outcome: `{st.session_state['last_outcome']}`")  # which branch the last run ended on
    if st.button("Start over"):  # clear the transcript without restarting the server
        st.session_state.clear()  # wipe messages + cached render data
        st.rerun()  # redraw the now-empty page

# --- pre-flight: index must exist or Match & Rank will blow up --------------
if not index_is_built():  # the FAISS artifact is gitignored and rebuilt locally
    st.error(
        "The service index hasn't been built yet. Run this once from the repo "
        "root, then reload:\n\n```\npython src/ingest.py\n```"
    )
    st.stop()  # nothing below can work without the index

# --- conversation state ---------------------------------------------------
if "messages" not in st.session_state:  # first load of this browser session
    st.session_state.messages = []  # chat-completions shape: [{"role", "content"}, ...]

# Replay the transcript so far (Streamlit reruns top-to-bottom on every input).
for msg in st.session_state.messages:  # user + assistant turns, in order
    with st.chat_message(msg["role"]):  # renders the correct avatar/alignment
        st.markdown(md_safe(msg["content"]))  # md_safe: a drafted "S$80" must not render as LaTeX

# On the assistant's most recent turn we may also have structured extras to
# show (shortlist + draft). They're stashed on session_state, not in the
# transcript, so they render once under the latest reply and don't get replayed
# as plain text above.
if st.session_state.get("last_outcome") == "matches_ready":  # only the success path has extras
    render_matches(st.session_state.get("last_state", {}))  # the ranked options
    render_draft(st.session_state.get("last_state", {}))  # the enquiry draft

# --- the input --------------------------------------------------------------
prompt = st.chat_input("Describe the care need, your hours, and your budget…")  # bottom-pinned chat box

if prompt:  # the caregiver submitted a turn
    # 1. record + show their message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(md_safe(prompt))

    # 2. run the whole pipeline on the full transcript. Passing every turn means
    #    a follow-up answer to a clarifying question is just re-extracted with
    #    the extra context — the graph is stateless between calls by design.
    try:
        with st.spinner("Matching options…"):  # the graph makes 1-3 LLM calls; give feedback
            final_state = run_pipeline(list(st.session_state.messages))  # copy so state can't alias our list
    except LLMError as e:  # provider unreachable, or a schema-validation guardrail fired
        # Don't append to the transcript — let them retry the same turn.
        st.session_state.messages.pop()  # remove the user turn we just added so history stays clean
        st.error(
            "I couldn't complete that safely — the model call failed or returned "
            f"something that didn't pass validation. Please try rephrasing.\n\n`{e}`"
        )
        st.stop()

    # 3. the caregiver-facing reply is whatever the Output node wrote
    reply = final_state.get("final_response") or "(no response produced)"  # fallback should never trigger
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # 4. stash the structured extras + outcome for the render pass above
    st.session_state.last_outcome = final_state.get("outcome")  # sidebar + extras gate read this
    st.session_state.last_state = final_state  # render_matches / render_draft read straight from it

    # 5. rerun so the new assistant turn + any extras draw through the normal path
    st.rerun()
