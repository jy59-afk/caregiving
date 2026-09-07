"""
api.py — the HTTP surface the web frontend talks to.

This is the second interface over the same LangGraph pipeline (the first is the
Streamlit `src/app.py`). Like that one, it is NOT the graded component: it owns
no matching logic, no prompts, no schema. It:

  * runs the full transcript through `graph.run()` on every turn, and
  * enriches the returned shortlist with the extra fields the Airbnb-style
    page needs — a photo per care type (`src/geo` has no say here) and a map
    coordinate per service (`geo.locate`) — by joining each `RankedMatch`
    back to its full record in `data/services.json`.

The guardrails still hold: nothing here sends or books anything, and the draft
(when there is one) is returned as text for the caregiver to send themselves.

Run:  uvicorn api:app --app-dir src --port 8000
      (needs LLM_BACKEND configured in .env and the FAISS index built via
      `python src/ingest.py` — the /api/chat route reports the latter clearly.)
"""

from __future__ import annotations

import logging  # surface the graph's per-node trace in the uvicorn console
import random  # a random sample of services for the pre-chat 'featured' view
from pathlib import Path  # locate the bundled frontend + image assets

from fastapi import FastAPI, HTTPException  # tiny web framework + its error type
from fastapi.responses import FileResponse  # to hand back index.html at "/"
from fastapi.staticfiles import StaticFiles  # serve the frontend + photos as-is
from pydantic import BaseModel, Field  # request/response contracts

import graph  # the compiled pipeline — imported as a module so we share its logger config
from config import DATA_DIR, PROJECT_ROOT, settings  # repo-anchored paths + typed .env view
from geo import locate  # service record -> {"lat", "lng"} for the map
from ingest import load_services  # the full curated records, to enrich matches by name
from llm import LLMError  # provider / schema-guardrail failures -> a clean 502, not a stack trace

# --------------------------------------------------------------------------
# Logging — same setup as src/app.py so `uvicorn api:app` shows the node trace
# (-> intake / <- match_rank / route: / outcome:) in its console.
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # INFO is the graph's node-transition channel (graph._traced)
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
for _noisy in ("httpx", "huggingface_hub", "faiss", "sentence_transformers", "watchfiles"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)  # mute library chatter so the trace stands out
log = logging.getLogger("respite.api")

# --------------------------------------------------------------------------
# Static asset locations
# --------------------------------------------------------------------------
WEB_DIR = PROJECT_ROOT / "web"          # index.html + styles.css + app.js
IMAGES_DIR = DATA_DIR / "images"        # bundled stock photos: services/NN.jpg per centre + <care_type>.jpg fallbacks

# Care types that have a generic fallback photo, used only if a record has no
# per-centre `image` (data/images/services/NN.jpg). `default.jpg` covers the rest.
_CARE_TYPES_WITH_PHOTO = frozenset(
    {"day-care", "home-help", "short-stay", "night-respite", "weekend-respite"}
)

# Join key: service name -> the full record. Built once at import; the dataset
# is static for the life of the process (rebuilding the index restarts it).
_SERVICES_BY_NAME: dict[str, dict] = {s["name"]: s for s in load_services()}


# ==========================================================================
# Request / response contracts
# ==========================================================================

class ChatTurn(BaseModel):
    """One message in the running transcript, chat-completions shape."""

    role: str = Field(description="'user' or 'assistant'.")
    content: str = Field(description="The message text.")


class ChatRequest(BaseModel):
    """The whole transcript so far. The graph is stateless between calls, so
    the client always sends every turn and the pipeline re-extracts with the
    fuller context."""

    messages: list[ChatTurn] = Field(description="Full conversation so far, oldest first.")


class MatchOut(BaseModel):
    """One ranked (or fallback) service, enriched for the card + map."""

    rank: int = Field(description="1-based position in the shortlist.")
    name: str
    provider: str | None = None
    care_type: str = Field(description="e.g. 'day-care'; drives which photo the card shows.")
    area: str
    cost_per_session: float | None = None
    contact: str | None = None
    why: str = Field(description="The grounded one-line 'why this fits', straight from the matcher.")
    grounding_passage: str = Field(description="The retrieved service text `why` must be supported by.")
    lat: float = Field(description="Approximate map latitude (town centroid + deterministic jitter).")
    lng: float = Field(description="Approximate map longitude.")
    photo_url: str = Field(description="Path under /images for this card's photo.")
    is_fallback: bool = Field(description="True when this is a next-best near-miss, not a true match.")


class DraftOut(BaseModel):
    """
    The enquiry draft — present only on the matches_ready outcome. The app never
    sends it; the frontend offers a `mailto:` button (or the centre's web form)
    so the caregiver sends it from their own mail client.
    """

    service_name: str
    subject: str
    body: str
    enquiry_email: str | None = Field(default=None, description="The centre's published enquiry email, for a mailto: link. None if it only takes a web form.")
    enquiry_form: str | None = Field(default=None, description="The centre's web contact-form URL, shown when there is no email. None if neither (enquire via AIC).")


class ProfileOut(BaseModel):
    """The distilled need, echoed back so the header pill can show what's being matched on."""

    needs_description: str | None = None
    area: str | None = None
    budget: float | None = None
    schedule: str | None = None
    relationship: str | None = None


class ChatResponse(BaseModel):
    """Everything the page needs to redraw itself after a turn."""

    reply: str = Field(description="The caregiver-facing text (the Output node's final_response).")
    outcome: str = Field(description="Which terminal branch the run ended on.")
    profile: ProfileOut | None = Field(default=None, description="What Intake extracted this turn.")
    matches: list[MatchOut] = Field(description="Up to 4 services — true matches, or the fallback near-misses.")
    draft: DraftOut | None = Field(default=None, description="Enquiry draft to review and send yourself, or null.")


# ==========================================================================
# Enrichment — join a RankedMatch back to its full record, add photo + coords
# ==========================================================================

def _photo_url(record: dict) -> str:
    """
    Path under /images for a service card. Prefer the record's own per-centre
    photo (`image: "services/NN.jpg"`); fall back to a generic per-care-type
    photo, then `default.jpg`, so a card never 404s.
    """
    image = record.get("image")  # e.g. "services/07.jpg" — one distinct photo per centre
    if image:
        return f"/images/{image}"
    care_type = record.get("care_type", "")
    slug = care_type if care_type in _CARE_TYPES_WITH_PHOTO else "default"
    return f"/images/{slug}.jpg"


def _first_sentence(text: str, *, max_len: int = 180) -> str:
    """First sentence of a service description, for a card blurb when there is
    no matcher-written 'why' (the pre-chat 'featured' view)."""
    text = (text or "").strip()
    dot = text.find(". ")
    snippet = text[: dot + 1] if 0 < dot < max_len else text[:max_len].rstrip()
    return snippet


def _build_row(
    record: dict, rank: int, *, why: str, is_fallback: bool, grounding_passage: str = ""
) -> MatchOut:
    """One enriched card row: display + map fields from a `data/services.json`
    record, plus a caller-supplied `why`/`grounding_passage`."""
    coords = locate(record or {"name": record.get("name", "")})  # town centroid + jitter; SG centre if unknown
    return MatchOut(
        rank=rank,
        name=record.get("name", ""),
        provider=record.get("provider"),
        care_type=record.get("care_type", "respite"),
        area=record.get("area", "Singapore"),
        cost_per_session=record.get("cost_per_session"),
        contact=record.get("contact"),
        why=why,
        grounding_passage=grounding_passage,
        lat=coords["lat"],
        lng=coords["lng"],
        photo_url=_photo_url(record),
        is_fallback=is_fallback,
    )


def _enrich(matches: list, *, is_fallback: bool) -> list[MatchOut]:
    """
    Turn the pipeline's `RankedMatch` list into `MatchOut` rows: keep the
    matcher's `name` / `why` / `grounding_passage` verbatim, and add the
    display + map fields from the joined `data/services.json` record.
    """
    rows: list[MatchOut] = []  # collected, in the matcher's order
    for i, m in enumerate(matches, start=1):  # 1-based rank for the card badge
        record = _SERVICES_BY_NAME.get(m.name) or {"name": m.name}  # authoritative record; {} only if the dataset drifted
        rows.append(
            _build_row(
                record, i,
                why=m.why,  # the grounded rationale — matcher's, untouched
                grounding_passage=m.grounding_passage,  # matcher's, untouched
                is_fallback=is_fallback,
            )
        )
    return rows


# ==========================================================================
# App
# ==========================================================================

app = FastAPI(
    title="Respite Navigator API",
    summary="HTTP surface for the Airbnb-style caregiver frontend, over the LangGraph pipeline.",
)


def _index_ready() -> bool:
    """True once `python src/ingest.py` has produced a loadable FAISS index."""
    return (settings.index_path / "index.faiss").exists()  # the file retrieval_tool.FAISS.load_local opens first


@app.get("/api/health")
def health() -> dict:
    """Liveness + a pointer to fix the one thing that commonly isn't ready."""
    index_built = _index_ready()  # retrieval needs this file
    return {
        "ok": True,
        "backend": settings.llm_backend,  # groq (dev) / bedrock (demo)
        "index_built": index_built,  # False -> run `python src/ingest.py`
        "services": len(_SERVICES_BY_NAME),  # how many records the join map holds
    }


@app.get("/api/featured", response_model=list[MatchOut])
def featured(n: int = 4) -> list[MatchOut]:
    """
    A random sample of services for the page's initial view — before the
    caregiver has said anything, so there is nothing to rank. Purely for
    "here's what's out there"; the chat replaces it with real matches. No model
    call, no index. `why` is the first sentence of the record's description.
    """
    n = max(1, min(n, len(_SERVICES_BY_NAME)))  # clamp to the dataset size
    picks = random.sample(list(_SERVICES_BY_NAME.values()), n)  # fresh random draw each load
    return [
        _build_row(rec, i, why=_first_sentence(rec.get("description", "")), is_fallback=False)
        for i, rec in enumerate(picks, start=1)
    ]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Run the transcript through the pipeline and return the redraw payload.

    Every turn re-runs the whole graph on the full transcript — so as the
    caregiver adds detail, the shortlist (and therefore the cards + map)
    updates on the next response with no extra plumbing.
    """
    # Pre-flight: Match & Rank loads the FAISS index lazily and blows up deep
    # in faiss if it was never built. Catch it here with an actionable message.
    if not _index_ready():
        raise HTTPException(
            status_code=503,
            detail="The service index isn't built yet. Run `python src/ingest.py` from the repo root, then retry.",
        )

    messages = [  # drop blank turns; the graph's transcript renderer skips them anyway
        {"role": t.role, "content": t.content} for t in req.messages if t.content.strip()
    ]
    if not messages:  # nothing to work with
        raise HTTPException(status_code=422, detail="Send at least one non-empty message.")

    log.info("POST /api/chat — %d message(s), backend=%s", len(messages), settings.llm_backend)
    try:
        final = graph.run(messages, consent_given=False)  # Intake -> Match & Rank -> Explain & Draft -> Output
    except LLMError as e:  # provider unreachable, or a schema-validation guardrail fired
        log.warning("pipeline LLMError: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"The model call failed or returned something that didn't pass validation. Try rephrasing. ({e})",
        )
    except RuntimeError as e:  # e.g. the index was built with a different embedding model than EMBEDDING_BACKEND wants
        log.warning("pipeline RuntimeError: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # never leak a raw 500 with an unparseable body to the frontend
        log.exception("pipeline failed")
        raise HTTPException(status_code=500, detail=f"The pipeline hit an unexpected error: {type(e).__name__}: {e}")

    # candidate_matches is the success path; fallback_matches is the near-miss
    # list shown (with a caveat, no draft) when nothing cleared every constraint.
    true_matches = final.get("candidate_matches") or []
    fallback = final.get("fallback_matches") or []
    matches = true_matches if true_matches else fallback
    enriched = _enrich(matches, is_fallback=not true_matches and bool(fallback))

    draft_obj = final.get("drafted_message")  # DraftedMessage | None
    draft = None
    if draft_obj is not None:
        rec = _SERVICES_BY_NAME.get(draft_obj.service_name, {})  # to attach the centre's enquiry email / form
        draft = DraftOut(
            **draft_obj.model_dump(),
            enquiry_email=rec.get("enquiry_email"),
            enquiry_form=rec.get("enquiry_form"),
        )

    # The Output node inlines the full draft text into `final_response` (the
    # Streamlit UI shows it that way). The web UI renders the draft in its own
    # editable block, so strip the inlined copy from the chat reply to avoid
    # showing the same message twice.
    reply = final.get("final_response") or ""
    if draft is not None:
        marker = "I've drafted an enquiry to "
        head, _, _ = reply.partition(marker)
        reply = head.rstrip() or reply  # keep the shortlist prose, drop the draft dump

    profile_obj = final.get("caregiver_profile")  # CaregiverProfile | None
    profile = (
        ProfileOut(**profile_obj.model_dump(include=set(ProfileOut.model_fields)))
        if profile_obj is not None
        else None
    )

    log.info("  -> outcome=%s, %d card(s)", final.get("outcome"), len(enriched))
    return ChatResponse(
        reply=reply,
        outcome=final.get("outcome") or "",
        profile=profile,
        matches=enriched,
        draft=draft,
    )


# --------------------------------------------------------------------------
# Static files. Order matters: the /images mount and the API routes above are
# registered first, so the catch-all frontend mount below never shadows them.
# --------------------------------------------------------------------------
if IMAGES_DIR.is_dir():  # only mount if the folder exists (photos are added in a later step)
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(WEB_DIR / "index.html")


# Everything else under / (styles.css, app.js, any future asset) is served
# straight from web/. Registered last so it can't shadow /api or /images.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
